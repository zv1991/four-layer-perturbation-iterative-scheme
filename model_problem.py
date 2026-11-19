# -*- coding: utf-8 -*-
"""
model_problem.py -- the model problem of Section 7 and its two examples.

The problem (7.1) of the article,

    du/dt - d^2u/dx^2 + beta(x, t) u + gamma(t) int_0^1 exp(x - y) u(y, t) dy = f(x, t),
    u(0, t) = u(1, t) = 0,    u(x, 0) = u_0(x),    0 < x < 1,  0 < t <= T,

with beta(x, t) = 5 sin(2 pi t) cos(pi x) and gamma(t) = 3 exp(-t), has the form
(2.1)-(2.2) in H = L_2(0, 1):

    A    = -d^2/dx^2 with homogeneous Dirichlet conditions
           (self-adjoint, positive definite, nu = pi^2);
    B(t) = multiplication by beta(., t) + gamma(t) * integral operator
           (bounded, neither self-adjoint nor of constant sign, does not
           commute with A).

Space discretization (Section 7).  On the uniform grid x_i = i h,
i = 1, ..., N, h = 1/(N + 1), the second derivative is replaced by the
central difference quotient and the integral by the rectangle rule.  This
gives a problem of the same form in H_h = R^N with the inner product
(v, w)_h = h sum_i v_i w_i:

    A_h = h^{-2} tridiag(-1, 2, -1)     (symmetric positive definite),
    (B_h(t) v)_i = beta(x_i, t) v_i + gamma(t) h sum_j exp(x_i - x_j) v_j.

Hence the semi-discrete model problem is a MatrixEvolutionProblem
(evolution_problem.py).  The eigenvectors of A_h are known explicitly:
A_h = Q diag(lambda) Q with the symmetric orthogonal matrix of the discrete
sine transform,

    Q_ij = sqrt(2h) sin(pi h i j),   lambda_j = 4 h^{-2} sin^2(pi h j / 2),

so that the resolvent (I + c A_h)^{-1} = Q diag(1/(1 + c lambda)) Q is
computed once for every c without an eigenvalue solver.

Manufactured solutions.  In Examples 1 and 2 the right-hand side is formed
on the grid, f_h(t) = U'(t) + (A_h + B_h(t)) U(t), where
U(t) = (u(x_i, t))_{i=1}^N for a given function

    u(x, t) = sum_m g_m(t) sin(m pi x).

Thus U(t) is the exact solution of the semi-discrete problem, and all
errors are errors of the time discretization measured in the norm of H_h.

    Example 1:  u = t^7 exp(-2t) sin(pi x) + (1/2) t^7 exp(-t) sin(2 pi x),
    Example 2:  u = exp(-t) sin(pi x) + cos(3t) sin(2 pi x).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from evolution_problem import ExactSolution, MatrixEvolutionProblem, Vector


# ======================================================================
#  Temporal factors g_m(t) of the manufactured solutions
# ======================================================================
class ExponentialFactor:
    """g(t) = c exp(a t)."""

    def __init__(self, c: float, a: float):
        self.c, self.a = float(c), float(a)

    def value(self, t: float) -> float:
        return self.c * math.exp(self.a * t)

    def derivative(self, t: float) -> float:
        return self.c * self.a * math.exp(self.a * t)


class CosineFactor:
    """g(t) = c cos(omega t)."""

    def __init__(self, c: float, omega: float):
        self.c, self.omega = float(c), float(omega)

    def value(self, t: float) -> float:
        return self.c * math.cos(self.omega * t)

    def derivative(self, t: float) -> float:
        return -self.c * self.omega * math.sin(self.omega * t)


class FlatFactor:
    """g(t) = c t^m exp(a t); all derivatives of order < m vanish at t = 0."""

    def __init__(self, c: float, m: int, a: float):
        self.c, self.m, self.a = float(c), int(m), float(a)

    def value(self, t: float) -> float:
        return self.c * t ** self.m * math.exp(self.a * t)

    def derivative(self, t: float) -> float:
        return self.c * (self.m * t ** (self.m - 1) + self.a * t ** self.m) * math.exp(self.a * t)


class ModeSolution(ExactSolution):
    """
    u(x_i, t) = sum_m g_m(t) sin(m pi x_i) on the grid nodes x_i.

    Every sin(m pi x_i) is an eigenvector of A_h; the solution is smooth in t
    if the factors g_m are smooth.
    """

    def __init__(self, x: np.ndarray, modes: Iterable[Tuple[int, object]]):
        self.modes: List[Tuple[int, object]] = [(int(m), g) for m, g in modes]
        self._shapes = [np.sin(m * math.pi * x) for m, _ in self.modes]

    def value(self, t: float) -> Vector:
        return sum(g.value(t) * s for (_, g), s in zip(self.modes, self._shapes))

    def derivative(self, t: float) -> Vector:
        return sum(g.derivative(t) * s for (_, g), s in zip(self.modes, self._shapes))


# ======================================================================
#  Semi-discrete model problem
# ======================================================================
class ModelProblem(MatrixEvolutionProblem):
    """
    Semi-discrete model problem U' + (A_h + B_h(t)) U = f_h(t) in H_h = R^N.

    Parameters
    ----------
    n_nodes : int
        Number N of interior grid nodes; h = 1/(N + 1).
    modes : sequence of (m, g_m)
        Manufactured solution u(x, t) = sum_m g_m(t) sin(m pi x).
    name : str
    """

    def __init__(self, n_nodes: int = 127, modes: Sequence[Tuple[int, object]] = (),
                 name: str = ""):
        N = int(n_nodes)
        h = 1.0 / (N + 1)
        x = h * np.arange(1, N + 1)
        self.n_nodes, self.h, self.x = N, h, x

        # A_h: central difference quotient of -d^2/dx^2 (Dirichlet conditions)
        A = (2.0 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)) / h ** 2
        # its eigenpairs (discrete sine transform)
        j = np.arange(1, N + 1)
        eigenvalues = 4.0 / h ** 2 * np.sin(0.5 * math.pi * j * h) ** 2
        Q = math.sqrt(2.0 * h) * np.sin(math.pi * h * np.outer(j, j))

        # ingredients of B_h(t)
        self._cos = np.cos(math.pi * x)                       # beta(x_i, t) / beta_amplitude(t)
        self._kernel = h * np.exp(x[:, None] - x[None, :])    # rectangle rule for the integral

        super().__init__(A, weight=h, eigen=(eigenvalues, Q),
                         exact_solution=ModeSolution(x, modes), name=name)

    # -- coefficients of the lower-order terms ----------------------------
    @staticmethod
    def beta_amplitude(t: float) -> float:
        """beta(x, t) = beta_amplitude(t) cos(pi x)."""
        return 5.0 * math.sin(2.0 * math.pi * t)

    @staticmethod
    def gamma(t: float) -> float:
        return 3.0 * math.exp(-t)

    # -- the operator B_h(t) ------------------------------------------------
    def apply_B(self, t: float, v: Vector) -> Vector:
        """(B_h(t) v)_i = beta(x_i, t) v_i + gamma(t) h sum_j exp(x_i - x_j) v_j."""
        return self.beta_amplitude(t) * self._cos * v + self.gamma(t) * (self._kernel @ v)

    def matrix_B(self, t: float) -> np.ndarray:
        return np.diag(self.beta_amplitude(t) * self._cos) + self.gamma(t) * self._kernel

    def apply_A(self, v: Vector) -> Vector:
        """A_h v by the three-point stencil (the same as the matrix product)."""
        w = 2.0 * v
        w[1:] -= v[:-1]
        w[:-1] -= v[1:]
        return w / self.h ** 2


# ======================================================================
#  The examples of Section 7
# ======================================================================
def example_1(n_nodes: int = 127) -> ModelProblem:
    """Example 1: u = t^7 exp(-2t) sin(pi x) + (1/2) t^7 exp(-t) sin(2 pi x)."""
    modes = [(1, FlatFactor(1.0, 7, -2.0)), (2, FlatFactor(0.5, 7, -1.0))]
    return ModelProblem(n_nodes, modes, name="Example 1")


def example_2(n_nodes: int = 127) -> ModelProblem:
    """Example 2: u = exp(-t) sin(pi x) + cos(3t) sin(2 pi x)."""
    modes = [(1, ExponentialFactor(1.0, -1.0)), (2, CosineFactor(1.0, 3.0))]
    return ModelProblem(n_nodes, modes, name="Example 2")
