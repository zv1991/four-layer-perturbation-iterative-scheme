# -*- coding: utf-8 -*-
"""
evolution_problem.py -- the evolutionary problem with a variable operator.

The article considers, in a Hilbert space H, the Cauchy problem

    du/dt + L(t) u = f(t),   t in (0, T],   u(0) = u_0,            (2.1)
    L(t) = A + B(t),                                                (2.2)

where A is a self-adjoint positive definite operator,

    (A u, u) >= nu ||u||^2,   nu > 0,                               (2.3)

and B(t) is a bounded operator depending on time,

    b = max_{0 <= t <= T} ||B(t)||.                                 (2.4)

The algorithms of the article (perturbation_algorithm.py,
four_layer_scheme.py, layer_iteration.py) access a problem *only* through
the interface EvolutionProblem defined below.  The interface contains
exactly the operations that the iterative realization of Section 3 needs:

    norm of H                                   norm(v)
    the resolvent  S = (I + c A)^{-1},  c > 0   resolvent(c)
    application of B(t) to a known vector       apply_B(t, v)
    the data f(t) and u_0                       f(t), initial_value()
    the constants nu of (2.3) and b of (2.4)        nu, bound_B(t_final)

The operator A is never inverted together with B(t), and B(t) is never
inverted at all: this is the point of moving the term with B(t_k) to the
right-hand side (Section 3).

Implementations
---------------
EvolutionProblem          abstract base class (any Hilbert space, any
                          representation of the vectors);
MatrixEvolutionProblem    H = R^N with the weighted inner product
                          (v, w) = omega * sum_i v_i w_i, a symmetric positive
                          definite matrix A and a matrix-valued function B(t);
                          the resolvent is computed from the spectral
                          decomposition A = Q diag(lambda) Q^T;
ExactSolution             optional known solution u(t); if a problem is given
                          an exact solution and no right-hand side, then
                          f(t) = u'(t) + (A + B(t)) u(t) and u_0 = u(0)
                          (method of manufactured solutions).

The model problem of Section 7 (model_problem.py) is a particular case of
MatrixEvolutionProblem.  Other problems are obtained either by passing
matrices to MatrixEvolutionProblem or by subclassing EvolutionProblem
(for example, with a sparse factorization in resolvent()).
"""

from __future__ import annotations

import abc
import math
from typing import Callable, Optional, Tuple

import numpy as np

Vector = np.ndarray
Operator = Callable[[Vector], Vector]


# ======================================================================
#  Known solution (optional)
# ======================================================================
class ExactSolution(abc.ABC):
    """A known solution u(t) of problem (2.1) and its derivative u'(t)."""

    @abc.abstractmethod
    def value(self, t: float) -> Vector:
        """u(t)."""

    @abc.abstractmethod
    def derivative(self, t: float) -> Vector:
        """u'(t)."""


# ======================================================================
#  Abstract problem
# ======================================================================
class EvolutionProblem(abc.ABC):
    """
    Abstract Cauchy problem (2.1)-(2.4):  u' + (A + B(t)) u = f(t),  u(0) = u_0.

    Subclasses implement the operations listed in the module docstring.
    Vectors of H are NumPy arrays of the same shape as initial_value().
    """

    #: label used in reports
    name: str = ""
    #: optional known solution (used to measure errors)
    exact_solution: Optional[ExactSolution] = None

    # ------------------------------------------------------------------
    #  Hilbert space H
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def inner(self, v: Vector, w: Vector) -> float:
        """Inner product (v, w) of H."""

    def norm(self, v: Vector) -> float:
        """Norm ||v|| = (v, v)^{1/2} of H."""
        return math.sqrt(max(self.inner(v, v), 0.0))

    # ------------------------------------------------------------------
    #  Operators
    # ------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def nu(self) -> float:
        """Lower bound nu > 0 of A, inequality (2.3)."""

    @abc.abstractmethod
    def apply_A(self, v: Vector) -> Vector:
        """A v (used only for manufactured right-hand sides and checks)."""

    @abc.abstractmethod
    def resolvent(self, c: float) -> Operator:
        """
        The operator S = (I + c A)^{-1}, c > 0, as a function g -> S g.

        This is the only operator that the algorithms invert: c = tau for the
        two-layer schemes (2.12)-(2.14) and c = 6 tau / 11 for the four-layer
        scheme (2.9).  Implementations should factorize I + c A once for every
        value of c and reuse the factorization.
        """

    @abc.abstractmethod
    def apply_B(self, t: float, v: Vector) -> Vector:
        """B(t) v; B(t) is only applied to known vectors."""

    @abc.abstractmethod
    def bound_B(self, t_final: float) -> float:
        """An upper bound b of ||B(t)|| on [0, t_final], definition (2.4)."""

    # ------------------------------------------------------------------
    #  Data
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def f(self, t: float) -> Vector:
        """Right-hand side f(t) of equation (2.1)."""

    @abc.abstractmethod
    def initial_value(self) -> Vector:
        """Initial vector u_0."""

    # ------------------------------------------------------------------
    #  Known solution
    # ------------------------------------------------------------------
    @property
    def has_exact_solution(self) -> bool:
        return self.exact_solution is not None

    def exact(self, t: float) -> Vector:
        """The exact solution u(t) (only if an ExactSolution is attached)."""
        if self.exact_solution is None:
            raise NotImplementedError("the problem %r has no known exact solution" % self.name)
        return self.exact_solution.value(t)

    def error(self, t: float, v: Vector) -> float:
        """||u(t) - v||."""
        return self.norm(v - self.exact(t))


# ======================================================================
#  Finite-dimensional problem with matrices
# ======================================================================
class MatrixEvolutionProblem(EvolutionProblem):
    """
    Problem (2.1) in H = R^N with the inner product (v, w) = omega * sum_i v_i w_i.

    Parameters
    ----------
    A : (N, N) array
        Symmetric positive definite matrix (self-adjoint in H, since the
        weight omega is a constant).
    B : callable t -> (N, N) array, optional
        The matrix B(t).  Subclasses may instead override apply_B() and
        matrix_B() (e.g. for a structured B(t)).
    f : callable t -> (N,) array, optional
        Right-hand side.  If it is omitted, exact_solution must be given and
        f(t) = u'(t) + A u(t) + B(t) u(t) is used (manufactured solution).
    u0 : (N,) array, optional
        Initial vector; by default u(0) of the exact solution.
    weight : float
        The weight omega of the inner product (e.g. the mesh size h).
    eigen : (eigenvalues, eigenvectors), optional
        Spectral decomposition A = Q diag(lambda) Q^T with Q orthogonal.  If it
        is omitted, it is computed by numpy.linalg.eigh.  A known
        decomposition (e.g. the discrete sine transform) can be supplied.
    exact_solution : ExactSolution, optional
    bound_samples : int
        Number of equally spaced points t in [0, T] at which ||B(t)|| is
        evaluated in bound_B().
    name : str
    """

    def __init__(self, A: np.ndarray,
                 B: Optional[Callable[[float], np.ndarray]] = None,
                 f: Optional[Callable[[float], Vector]] = None,
                 u0: Optional[Vector] = None, *,
                 weight: float = 1.0,
                 eigen: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 exact_solution: Optional[ExactSolution] = None,
                 bound_samples: int = 401,
                 name: str = ""):
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        if not np.allclose(A, A.T, rtol=1e-12, atol=1e-12 * np.abs(A).max()):
            raise ValueError("A must be symmetric (self-adjoint)")
        if f is None and exact_solution is None:
            raise ValueError("either the right-hand side f or an exact solution must be given")
        self.name = name
        self.dimension = A.shape[0]
        self.weight = float(weight)
        self._A = A
        self._B = B
        self._f = f
        self._u0 = None if u0 is None else np.asarray(u0, dtype=float)
        self.exact_solution = exact_solution
        self.bound_samples = int(bound_samples)

        # Spectral decomposition A = Q diag(lambda) Q^T; it gives nu (2.3) and
        # the resolvents (I + c A)^{-1} = Q diag(1 / (1 + c lambda)) Q^T.
        if eigen is None:
            eigenvalues, eigenvectors = np.linalg.eigh(A)
        else:
            eigenvalues, eigenvectors = (np.asarray(e, dtype=float) for e in eigen)
        if eigenvalues.min() <= 0.0:
            raise ValueError("A must be positive definite")
        self.eigenvalues = eigenvalues
        self._Q = eigenvectors
        self._QT = np.ascontiguousarray(eigenvectors.T)
        self._resolvents = {}

    # -- Hilbert space ---------------------------------------------------
    def inner(self, v: Vector, w: Vector) -> float:
        return self.weight * float(np.dot(v, w))

    def norm(self, v: Vector) -> float:
        return math.sqrt(self.weight) * float(np.linalg.norm(v))

    def zeros(self) -> Vector:
        return np.zeros(self.dimension)

    # -- operators ---------------------------------------------------------
    @property
    def nu(self) -> float:
        return float(self.eigenvalues.min())

    @property
    def matrix_A(self) -> np.ndarray:
        return self._A

    def apply_A(self, v: Vector) -> Vector:
        return self._A @ v

    def resolvent_matrix(self, c: float) -> np.ndarray:
        """The matrix of (I + c A)^{-1}; computed once for every value of c."""
        key = float(c)
        if key not in self._resolvents:
            self._resolvents[key] = (self._Q / (1.0 + key * self.eigenvalues)) @ self._QT
        return self._resolvents[key]

    def resolvent(self, c: float) -> Operator:
        matrix = self.resolvent_matrix(c)
        return matrix.__matmul__

    def matrix_B(self, t: float) -> np.ndarray:
        if self._B is None:
            raise NotImplementedError("B(t) is not given; override matrix_B() and apply_B()")
        return self._B(t)

    def apply_B(self, t: float, v: Vector) -> Vector:
        return self.matrix_B(t) @ v

    def bound_B(self, t_final: float) -> float:
        """
        b = max_t ||B(t)||, definition (2.4), evaluated at bound_samples equally
        spaced points of [0, t_final].  Since the weight of the inner product is
        a constant, the operator norm of H is the spectral norm of the matrix.
        """
        return max(np.linalg.norm(self.matrix_B(t), 2)
                   for t in np.linspace(0.0, t_final, self.bound_samples))

    # -- data --------------------------------------------------------------
    def f(self, t: float) -> Vector:
        if self._f is not None:
            return self._f(t)
        # manufactured right-hand side  f = u' + (A + B(t)) u
        u = self.exact_solution.value(t)
        return self.exact_solution.derivative(t) + self.apply_A(u) + self.apply_B(t, u)

    def initial_value(self) -> Vector:
        if self._u0 is not None:
            return self._u0.copy()
        return self.exact_solution.value(0.0)

    # -- reference solution of one layer equation --------------------------
    def direct_layer_solve(self, c: float, t: float, F: Vector) -> Vector:
        """
        Solution of (I + c (A + B(t))) w = F by Gaussian elimination.  It is
        used only as a reference in the experiments (Figure 7.3); the
        algorithms never invert the operator A + B(t).
        """
        matrix = np.eye(self.dimension) + c * (self._A + self.matrix_B(t))
        return np.linalg.solve(matrix, F)
