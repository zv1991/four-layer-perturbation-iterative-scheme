# -*- coding: utf-8 -*-
"""
grid_functions.py -- time grid, backward difference quotients and errors.

Notation of Section 2 of the article: on [0, T] the grid

    t_k = k tau,   k = 0, 1, ..., n,   tau = T / n,

is introduced; for a grid function w_k the backward difference quotients are

    delta w_k   = (w_k - w_{k-1}) / tau,
    delta^m w_k = delta(delta^{m-1} w)_k
                = tau^{-m} sum_{i=0}^{m} (-1)^i binom(m, i) w_{k-i},   m = 2, 3, ...

A grid function with values in H is stored as an array whose first index
is the layer number k.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np


class TimeGrid:
    """The uniform grid t_k = k tau, k = 0, ..., n, tau = T / n, on [0, T]."""

    def __init__(self, t_final: float, n_steps: int):
        if n_steps < 1:
            raise ValueError("the number of steps must be positive")
        self.t_final = float(t_final)
        self.n = int(n_steps)
        self.tau = self.t_final / self.n
        self.t = self.tau * np.arange(self.n + 1)

    def __len__(self) -> int:
        return self.n + 1

    def __repr__(self) -> str:
        return "TimeGrid(T=%g, n=%d, tau=%g)" % (self.t_final, self.n, self.tau)


def _signed_binomials(order: int) -> List[float]:
    """(-1)^i binom(order, i), i = 0, ..., order."""
    return [float((-1) ** i * math.comb(order, i)) for i in range(order + 1)]


def backward_difference(w, k: int, order: int, tau: float):
    """
    Backward difference quotient delta^order w_k of a grid function w
    (array indexed by the layer number); it uses the layers k - order, ..., k.
    """
    if order == 0:
        return w[k]
    if k < order:
        raise IndexError("delta^%d w_k needs the layers k - %d, ..., k (k = %d)" % (order, order, k))
    coefficients = _signed_binomials(order)
    result = w[k] * coefficients[0]
    for i in range(1, order + 1):
        result = result + coefficients[i] * w[k - i]
    return result / tau ** order


def observed_orders(errors: Sequence[float]) -> List[Optional[float]]:
    """
    Observed orders of convergence log_2(E(tau) / E(tau / 2)) for a sequence
    of errors on successively halved steps; None for the first entry.
    """
    result: List[Optional[float]] = [None]
    for coarse, fine in zip(errors[:-1], errors[1:]):
        result.append(math.log(coarse / fine, 2.0))
    return result


def error_history(problem, grid: TimeGrid, approximation) -> List[float]:
    """||u(t_k) - w_k|| for k = 0, ..., n (the problem must have an exact solution)."""
    return [problem.error(grid.t[k], approximation[k]) for k in range(len(grid))]


def max_error(problem, grid: TimeGrid, approximation, first_layer: int = 1) -> float:
    """max_{first_layer <= k <= n} ||u(t_k) - w_k||."""
    return max(problem.error(grid.t[k], approximation[k]) for k in range(first_layer, len(grid)))


def exact_starting_values(problem, grid: TimeGrid, layers=(1, 2)):
    """The exact values u(t_k) on the given layers (starting values)."""
    return tuple(problem.exact(grid.t[k]) for k in layers)
