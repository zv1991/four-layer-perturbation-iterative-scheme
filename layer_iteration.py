# -*- coding: utf-8 -*-
"""
layer_iteration.py -- iterative realization of an implicit layer (Section 3).

On every time layer t_k, the schemes of the article require the solution of
an equation of the form

    (I + c A + c B_k) w = F_k,          B_k = B(t_k),                  (3.1)

with c = tau for the two-layer schemes (2.12)-(2.14) and c = 6 tau / 11 for the
four-layer scheme (2.9) (Remark 3.4).  The operator I + c A + c B_k depends on k
and, in general, is not self-adjoint.  The term with B_k is moved to the
right-hand side, the terms with A are kept in place,

    (I + c A) w = F_k - c B_k w,                                       (3.3)

and the equation is solved by the iterative method

    (I + c A) w^{[s+1]} = F_k - c B_k w^{[s]},   s = 0, 1, ...          (3.4)

Only the time-independent self-adjoint operator I + c A is inverted, and it
is the same on all layers and in all iterations.  With

    S = (I + c A)^{-1},   q = c b / (1 + c nu),                         (3.5)

Theorem 3.1 states that, if q < 1 (condition (3.7)), equation (3.1) has a
unique solution w and

    ||w^{[s]} - w|| <= q^s ||w^{[0]} - w||,                             (3.9)
    ||w^{[s]} - w|| <= q / (1 - q) ||w^{[s]} - w^{[s-1]}||.              (3.10)

Stopping criterion (Remark 3.3): the iterations are stopped as soon as

    ||w^{[s]} - w^{[s-1]}|| <= (1 - q) tol / q,

which guarantees ||w^{[s]} - w|| <= tol by (3.10).  Alternatively, a
prescribed number s of iterations can be performed on every layer; then no
bound for ||B(t)|| is needed.  The natural initial approximation is the
value on the preceding layer, w^{[0]} = w_{k-1} (Remark 3.3).
"""

from __future__ import annotations

import math
from typing import List, Optional

from evolution_problem import EvolutionProblem, Vector


class IterationDidNotConverge(RuntimeError):
    """Raised if the stopping criterion is not met within max_sweeps iterations."""


class LayerIteration:
    """
    Solver of the layer equation (3.1) by the iterations (3.4).

    Parameters
    ----------
    problem : EvolutionProblem
    c : float
        Coefficient of the layer equation: tau for the two-layer schemes,
        6 tau / 11 for the four-layer scheme.
    bound_B : float or None
        Upper bound b of ||B(t)||, definition (2.4).  It is needed for the ratio q
        of (3.5) and for the stopping criterion; it may be None if a prescribed
        number of iterations is performed.
    tolerance : float
        Tolerance tol of the stopping criterion (absolute, in the norm of H).
    sweeps : int or None
        If given, exactly this number of iterations is performed (no stopping
        criterion).
    max_sweeps : int
        Safeguard for the stopping criterion.
    """

    def __init__(self, problem: EvolutionProblem, c: float, bound_B: Optional[float] = None, *,
                 tolerance: float = 1.0e-13, sweeps: Optional[int] = None,
                 max_sweeps: int = 500):
        if c <= 0.0:
            raise ValueError("the coefficient c must be positive")
        if sweeps is None and bound_B is None:
            raise ValueError("the stopping criterion needs the bound b of ||B(t)||")
        if sweeps is not None and sweeps < 0:
            raise ValueError("the number of iterations must be nonnegative")
        self.problem = problem
        self.c = float(c)
        self.tolerance = float(tolerance)
        self.sweeps = sweeps
        self.max_sweeps = int(max_sweeps)

        # S = (I + c A)^{-1}: the only operator that is inverted, (3.5)
        self._S = problem.resolvent(self.c)

        # ratio of the geometric progression, (3.5); condition (3.7)
        if bound_B is None:
            self.q = None
            self._threshold = None
        else:
            self.q = self.c * bound_B / (1.0 + self.c * problem.nu)
            if sweeps is None and self.q >= 1.0:
                raise ValueError("q = c b / (1 + c nu) = %.3g >= 1: condition (3.7) is violated, "
                                 "the iterations need not converge" % self.q)
            # stopping criterion of Remark 3.3:  ||w^s - w^{s-1}|| <= (1 - q) tol / q
            self._threshold = (math.inf if self.q == 0.0
                               else self.tolerance * (1.0 - self.q) / self.q)

        # statistics
        self.total_sweeps = 0
        self.total_solves = 0

    # ------------------------------------------------------------------
    def solve(self, t: float, F: Vector, initial: Vector,
              history: Optional[List[Vector]] = None) -> Vector:
        """
        Solve (I + c A + c B(t)) w = F by the iterations (3.4).

        Parameters
        ----------
        t : float
            The layer t_k (B is evaluated at t).
        F : vector
            Right-hand side F_k of (3.1).
        initial : vector
            Initial approximation w^{[0]} (normally the value on the preceding
            layer, Remark 3.3).
        history : list or None
            If a list is given, all iterates w^{[0]}, w^{[1]}, ... are appended.
        """
        S, c, problem = self._S, self.c, self.problem
        w = initial
        if history is not None:
            history.append(w.copy())
        if self.sweeps is not None:
            # prescribed number of iterations
            n_sweeps = self.sweeps
            for _ in range(n_sweeps):
                w = S(F - c * problem.apply_B(t, w))                      # (3.4)
                if history is not None:
                    history.append(w.copy())
        else:
            # iterations with the a posteriori stopping criterion (3.10)
            n_sweeps = 0
            while True:
                w_new = S(F - c * problem.apply_B(t, w))                  # (3.4)
                n_sweeps += 1
                increment = problem.norm(w_new - w)
                w = w_new
                if history is not None:
                    history.append(w.copy())
                if increment <= self._threshold:
                    break
                if n_sweeps >= self.max_sweeps:
                    raise IterationDidNotConverge(
                        "the iterations did not converge in %d sweeps (t = %g)" % (n_sweeps, t))
        self.total_sweeps += n_sweeps
        self.total_solves += 1
        return w

    # ------------------------------------------------------------------
    @property
    def mean_sweeps(self) -> float:
        """Mean number of iterations (inversions of I + c A) per layer equation."""
        return self.total_sweeps / max(self.total_solves, 1)

    def reset_statistics(self) -> None:
        self.total_sweeps = 0
        self.total_solves = 0
