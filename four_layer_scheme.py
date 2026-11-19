# -*- coding: utf-8 -*-
"""
four_layer_scheme.py -- direct realization of the four-layer scheme (2.9).

The purely implicit four-layer semi-discrete scheme of the third order of
approximation (the three-step backward differentiation formula, BDF3),

    (11/6 u_k - 3 u_{k-1} + 3/2 u_{k-2} - 1/3 u_{k-3}) / tau + (A + B_k) u_k = f_k,
    k = 3, ..., n,                                                          (2.9)

is obtained from the identity (2.7) by omitting the remainder term tau^3 R_k of
(2.5)-(2.6).  It needs the starting values u_0, u_1, u_2.

Realization (Remark 3.4).  Multiplying (2.9) by 6 tau / 11 gives an equation of
the form (3.1) with c = 6 tau / 11,

    (I + c A + c B_k) u_k = F_k,
    F_k = (18 u_{k-1} - 9 u_{k-2} + 2 u_{k-3}) / 11 + c f_k,

and the term with B_k is moved to the right-hand side as in Section 3:

    (I + 6/11 tau A) u_k^{[s+1]} = 18/11 u_{k-1} - 9/11 u_{k-2} + 2/11 u_{k-3}
                                   + 6/11 tau (f_k - B_k u_k^{[s]}).

The iterations converge at the rate of a geometric progression with the
ratio 6 tau b / (11 + 6 tau nu), provided that this ratio is less than one.
Only the operator I + 6/11 tau A is inverted.  This class is used in
Section 7 for the comparison with the perturbation algorithm.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from evolution_problem import EvolutionProblem, Vector
from grid_functions import TimeGrid
from layer_iteration import LayerIteration


class FourLayerSolution:
    """
    Solution of the four-layer scheme.

    Attributes
    ----------
    grid : TimeGrid
    u : array (n + 1, ...)
        u_0 = u(0), the given starting values u_1, u_2, and u_k from (2.9), k >= 3.
    mean_sweeps : float
        Mean number of iterations (inversions of I + 6/11 tau A) per layer.
    q : float or None
        Ratio 6 tau b / (11 + 6 tau nu) of the iterations.
    """

    def __init__(self, grid: TimeGrid, u, mean_sweeps: float, q: Optional[float]):
        self.grid = grid
        self.u = u
        self.mean_sweeps = mean_sweeps
        self.q = q

    @property
    def tau(self) -> float:
        return self.grid.tau

    @property
    def t(self) -> np.ndarray:
        return self.grid.t


class FourLayerScheme:
    """
    The four-layer scheme (2.9) realized by the iterations of Remark 3.4.

    Parameters
    ----------
    problem : EvolutionProblem
    t_final : float
    n_steps : int
    bound_B : float, optional
        Upper bound b of ||B(t)||; computed by problem.bound_B(T) if it is
        omitted and the stopping criterion is used.
    tolerance : float
        Tolerance of the stopping criterion of the iterations.
    sweeps : int, optional
        Prescribed number of iterations on every layer.
    """

    #: coefficients of (2.9): (11/6 u_k - 3 u_{k-1} + 3/2 u_{k-2} - 1/3 u_{k-3}) / tau
    COEFFICIENTS = (11.0 / 6.0, -3.0, 1.5, -1.0 / 3.0)

    def __init__(self, problem: EvolutionProblem, t_final: float, n_steps: int, *,
                 bound_B: Optional[float] = None, tolerance: float = 1.0e-13,
                 sweeps: Optional[int] = None, max_sweeps: int = 500):
        if n_steps < 3:
            raise ValueError("the four-layer scheme needs at least three steps")
        self.problem = problem
        self.grid = TimeGrid(t_final, n_steps)
        if bound_B is None and sweeps is None:
            bound_B = problem.bound_B(t_final)
        self.bound_B = bound_B
        # c = 6 tau / 11: the operator I + 6/11 tau A is inverted
        self.c = 6.0 * self.grid.tau / 11.0
        self.solver = LayerIteration(problem, self.c, bound_B, tolerance=tolerance,
                                     sweeps=sweeps, max_sweeps=max_sweeps)

    def run(self, u1: Vector, u2: Vector) -> FourLayerSolution:
        """Run the scheme with the starting values u1 ~ u(t_1), u2 ~ u(t_2)."""
        problem, grid, c = self.problem, self.grid, self.c
        t = grid.t
        initial = problem.initial_value()
        u = np.zeros((grid.n + 1,) + np.shape(initial))
        u[0], u[1], u[2] = initial, u1, u2

        self.solver.reset_statistics()
        for k in range(3, grid.n + 1):
            # equation (2.9) multiplied by 6 tau / 11: (I + c A + c B_k) u_k = F_k
            F = (18.0 * u[k - 1] - 9.0 * u[k - 2] + 2.0 * u[k - 3]) / 11.0 + c * problem.f(t[k])
            # iterations of Remark 3.4 with the initial approximation u_{k-1}
            u[k] = self.solver.solve(t[k], F, u[k - 1])
        return FourLayerSolution(grid, u, self.solver.mean_sweeps, self.solver.q)


def four_layer_residual(problem: EvolutionProblem, grid: TimeGrid, w, k: int) -> Vector:
    """
    Residual of scheme (2.9) on a grid function w at the layer k >= 3:

        (11/6 w_k - 3 w_{k-1} + 3/2 w_{k-2} - 1/3 w_{k-3}) / tau + (A + B_k) w_k - f_k.

    For w = v of the perturbation algorithm it equals the residual
    R~_k(tau) of Lemma 5.1, formula (5.2) (up to the errors of the iterations).
    """
    a = FourLayerScheme.COEFFICIENTS
    t = grid.t[k]
    difference = (a[0] * w[k] + a[1] * w[k - 1] + a[2] * w[k - 2] + a[3] * w[k - 3]) / grid.tau
    return difference + problem.apply_A(w[k]) + problem.apply_B(t, w[k]) - problem.f(t)
