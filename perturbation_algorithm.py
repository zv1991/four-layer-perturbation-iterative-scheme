# -*- coding: utf-8 -*-
"""
perturbation_algorithm.py -- the perturbation algorithm (Sections 2 and 3).

The purely implicit four-layer semi-discrete scheme of the third order
(the three-step backward differentiation formula, BDF3)

    (11/6 u_k - 3 u_{k-1} + 3/2 u_{k-2} - 1/3 u_{k-3}) / tau + (A + B_k) u_k = f_k   (2.9)

is embedded in the one-parameter family

    delta u_k + L_k u_k + (eps/2) delta^2 u_k + (eps^2/3) delta^3 u_k = f_k,           (2.10)

which coincides with (2.9) for eps = tau.  Substituting the expansion
u_k = sum_j eps^j u_k^(j) (2.11) and equating the coefficients of eps^0, eps^1,
eps^2 gives three two-layer (implicit Euler) schemes:

    (u0_k - u0_{k-1}) / tau + (A + B_k) u0_k = f_k,         u0_0 = u_0,  k >= 1,   (2.12)
    (u1_k - u1_{k-1}) / tau + (A + B_k) u1_k = -1/2 delta^2 u0_k,          k >= 2,   (2.13)
    (u2_k - u2_{k-1}) / tau + (A + B_k) u2_k = -1/2 delta^2 u1_k
                                               -1/3 delta^3 u0_k,          k >= 3,   (2.14)

and the approximate solution of problem (2.1) is

    v_k = u0_k + tau u1_k + tau^2 u2_k,   u(t_k) ~ v_k,   k = 2, ..., n.               (2.15)

Starting vectors.  As in [Rogava, Gulua, Georgian Math. J. 25 (2018)], the
starting vectors of (2.13) and (2.14) are defined from given approximations
v_1 ~ u(t_1) and v_2 ~ u(t_2) (accuracy O(tau^3)) by

    v_1 = u0_1 + tau u1_1,                  i.e.  u1_1 = (v_1 - u0_1) / tau,
    v_2 = u0_2 + tau u1_2 + tau^2 u2_2,     i.e.  u2_2 = (v_2 - u0_2 - tau u1_2) / tau^2.

Realization.  The three grid functions are computed layer by layer: on
the layer t_k, first u0_k, then u1_k (it needs u0 on the layers k-2, ..., k),
then u2_k (it needs u1 on k-2, ..., k and u0 on k-3, ..., k).  On every layer
only equations of two-layer schemes are solved, each of the form (3.1) with
c = tau and

    F_k = u^(j)_{k-1} + tau g_k^(j),
    g^(0)_k = f_k,  g^(1)_k = -1/2 delta^2 u0_k,  g^(2)_k = -1/2 delta^2 u1_k - 1/3 delta^3 u0_k,   (3.2)

by the iterations (3.4) of Section 3 (layer_iteration.py) with the initial
approximation w^{[0]} = u^(j)_{k-1} (Remark 3.3).  Thus, in all iterations, on
all layers and for all three schemes, one and the same operator I + tau A is
inverted.

Error estimate (Theorem 6.3).  If the solution of (2.1) is smooth enough, the
starting values are of accuracy O(tau^3), and the quantities M_0, M_1, M_2 of
(5.3) are bounded uniformly with respect to tau, then ||u(t_k) - v_k|| = O(tau^3).
The intermediate approximations u0_k and u0_k + tau u1_k are of the first and
second order, respectively.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from evolution_problem import EvolutionProblem, Vector
from grid_functions import TimeGrid, backward_difference
from layer_iteration import LayerIteration


class PerturbationSolution:
    """
    Grid functions produced by the perturbation algorithm.

    Attributes
    ----------
    grid : TimeGrid
    u0, u1, u2 : arrays (n + 1, ...)
        Solutions of the two-layer schemes (2.12), (2.13), (2.14); u1_0 = 0 and
        u2_0 = u2_1 = 0 (these layers do not belong to the schemes).
    v : array (n + 1, ...)
        Approximate solution (2.15) on the layers k >= 2; v_0 = u_0 and v_1 is
        the given starting value.
    mean_sweeps : float
        Mean number of iterations (inversions of I + tau A) per two-layer solve.
    q : float or None
        Ratio of the geometric progression of the iterations, (3.5).
    """

    def __init__(self, grid: TimeGrid, u0, u1, u2, v, mean_sweeps: float, q: Optional[float]):
        self.grid = grid
        self.u0, self.u1, self.u2, self.v = u0, u1, u2, v
        self.mean_sweeps = mean_sweeps
        self.q = q

    @property
    def tau(self) -> float:
        return self.grid.tau

    @property
    def t(self) -> np.ndarray:
        return self.grid.t

    def first_order(self):
        """u0_k: approximation of the first order."""
        return self.u0

    def second_order(self):
        """u0_k + tau u1_k: approximation of the second order."""
        return self.u0 + self.tau * self.u1

    def third_order(self):
        """v_k of (2.15): approximation of the third order."""
        return self.v


class PerturbationAlgorithm:
    """
    Perturbation algorithm for the four-layer scheme (2.9).

    Parameters
    ----------
    problem : EvolutionProblem
    t_final : float
        Length T of the time interval.
    n_steps : int
        Number n of time steps, tau = T / n.
    bound_B : float, optional
        Upper bound b of ||B(t)||, definition (2.4).  If it is omitted and the
        stopping criterion is used, it is computed by problem.bound_B(T).
    tolerance : float
        Tolerance tol of the stopping criterion of the iterations (Remark 3.3).
    sweeps : int, optional
        Prescribed number of iterations on every layer of each scheme
        (instead of the stopping criterion).

    Example
    -------
    >>> algorithm = PerturbationAlgorithm(problem, 1.0, 160, bound_B=b)
    >>> solution = algorithm.run(v1, v2)          # v1 ~ u(t_1), v2 ~ u(t_2)
    >>> solution.v[-1]                            # approximation of u(T)
    """

    def __init__(self, problem: EvolutionProblem, t_final: float, n_steps: int, *,
                 bound_B: Optional[float] = None, tolerance: float = 1.0e-13,
                 sweeps: Optional[int] = None, max_sweeps: int = 500):
        self.problem = problem
        self.grid = TimeGrid(t_final, n_steps)
        if bound_B is None and sweeps is None:
            bound_B = problem.bound_B(t_final)
        self.bound_B = bound_B
        # one solver for the three schemes: all of them invert I + tau A
        self.solver = LayerIteration(problem, self.grid.tau, bound_B, tolerance=tolerance,
                                     sweeps=sweeps, max_sweeps=max_sweeps)

    # ------------------------------------------------------------------
    def _two_layer_step(self, t: float, w_prev: Vector, g: Vector) -> Vector:
        """
        One layer of a two-layer scheme  (w - w_prev) / tau + (A + B(t)) w = g,
        i.e. equation (3.1) with F = w_prev + tau g, solved by the iterations
        (3.4) with the initial approximation w^{[0]} = w_prev (Remark 3.3).
        """
        return self.solver.solve(t, w_prev + self.grid.tau * g, w_prev)

    # ------------------------------------------------------------------
    def run(self, v1: Vector, v2: Vector, n_layers: Optional[int] = None) -> PerturbationSolution:
        """
        Run the algorithm with the starting values v1 ~ u(t_1), v2 ~ u(t_2).

        n_layers
            If given, only the layers 0, ..., n_layers are computed.
        """
        problem, grid, tau = self.problem, self.grid, self.grid.tau
        t = grid.t
        last = grid.n if n_layers is None else int(n_layers)
        if not 1 <= last <= grid.n:
            raise ValueError("n_layers must be between 1 and n")

        initial = problem.initial_value()
        shape = (grid.n + 1,) + np.shape(initial)
        u0, u1, u2 = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        u0[0] = initial

        self.solver.reset_statistics()
        for k in range(1, last + 1):
            # ---- scheme (2.12): implicit Euler scheme, first order ----------
            u0[k] = self._two_layer_step(t[k], u0[k - 1], problem.f(t[k]))
            if k == 1:
                # starting vector of scheme (2.13):  v_1 = u0_1 + tau u1_1
                u1[1] = (v1 - u0[1]) / tau
                continue

            # ---- scheme (2.13): right-hand side -1/2 delta^2 u0_k -----------
            g1 = -0.5 * backward_difference(u0, k, 2, tau)
            u1[k] = self._two_layer_step(t[k], u1[k - 1], g1)
            if k == 2:
                # starting vector of scheme (2.14):  v_2 = u0_2 + tau u1_2 + tau^2 u2_2
                u2[2] = (v2 - u0[2] - tau * u1[2]) / tau ** 2
                continue

            # ---- scheme (2.14): right-hand side -1/2 delta^2 u1_k - 1/3 delta^3 u0_k
            g2 = -0.5 * backward_difference(u1, k, 2, tau) - backward_difference(u0, k, 3, tau) / 3.0
            u2[k] = self._two_layer_step(t[k], u2[k - 1], g2)

        # ---- approximate solution (2.15) ------------------------------------
        v = u0 + tau * u1 + tau ** 2 * u2
        v[1] = v1
        if last < grid.n:
            v[last + 1:] = 0.0
        return PerturbationSolution(grid, u0, u1, u2, v, self.solver.mean_sweeps, self.solver.q)


def extrapolated_starting_values(problem: EvolutionProblem, tau: float, bound_B: float, *,
                                 substeps: int = 4, tolerance: float = 1.0e-13):
    """
    Starting values v_1 ~ u(t_1), v_2 ~ u(t_2) with an error O(tau^3) for a
    problem whose solution is not known (the article assumes that such
    values are given; this is one way to obtain them).

    The implicit Euler scheme (2.12) is applied on [0, 2 tau] with the steps
    h, h/2, h/4, h = tau / substeps (each step realized by the iterations of
    Section 3), and the results are combined by Richardson extrapolation,

        v = (8 E_{h/4} - 6 E_{h/2} + E_h) / 3,

    which eliminates the terms of order h and h^2 of the error expansion of
    the implicit Euler scheme for a smooth solution.
    """
    results = []
    for refinement in (1, 2, 4):
        m = int(substeps) * refinement                  # steps per tau
        h = tau / m
        solver = LayerIteration(problem, h, bound_B, tolerance=tolerance)
        w = problem.initial_value()
        values = []
        for step in range(1, 2 * m + 1):
            t = step * h
            w = solver.solve(t, w + h * problem.f(t), w)       # scheme (2.12) with step h
            if step % m == 0:
                values.append(w)
        results.append(values)
    (e1, e2), (h1, h2), (q1, q2) = results
    return (8.0 * q1 - 6.0 * h1 + e1) / 3.0, (8.0 * q2 - 6.0 * h2 + e2) / 3.0
