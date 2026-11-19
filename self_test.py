# -*- coding: utf-8 -*-
"""
self_test.py -- checks of the implementation (no extra packages needed).

The checks use a *general* problem, not the model problem of Section 7: a
random symmetric positive definite matrix A, a random non-symmetric matrix
function B(t), and a manufactured smooth solution.  This demonstrates that
the classes work for any EvolutionProblem.  The checks are:

  1. the resolvent (I + c A)^{-1} of MatrixEvolutionProblem;
  2. Theorem 3.1: the iterations (3.4) converge to the solution of (3.1), with
     the error reduced at least by the factor q of (3.5) in every iteration;
  3. Lemma 5.1: the approximate solution v of (2.15) satisfies the four-layer
     scheme (2.9) with the residual R~_k(tau) of (5.2);
  4. orders of convergence 1, 2, 3 of u0_k, u0_k + tau u1_k and v_k for a
     solution whose derivatives vanish at t = 0 (then the quantities M_0, M_1,
     M_2 of (5.3) are bounded, as in Example 1), and order 3 of the four-layer
     scheme (2.9); for a general smooth solution (u'(0) != 0) the observed
     order of v increases towards 3 only gradually, as in Example 2;
  5. starting values obtained by extrapolation (for problems without a
     known solution) give the same accuracy as the exact starting values;
  6. the model problem: A_h and its sine-transform eigenpairs are consistent.

Usage:   python self_test.py          (exit code 0 if all checks pass)
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                               # noqa: E402

from diagnostics import residual_identity_defect                                # noqa: E402
from evolution_problem import ExactSolution, MatrixEvolutionProblem              # noqa: E402
from four_layer_scheme import FourLayerScheme                                    # noqa: E402
from grid_functions import exact_starting_values, max_error, observed_orders     # noqa: E402
from layer_iteration import LayerIteration                                       # noqa: E402
from model_problem import example_1                                              # noqa: E402
from perturbation_algorithm import (PerturbationAlgorithm,                       # noqa: E402
                                    extrapolated_starting_values)

FAILURES = []


def check(condition: bool, text: str) -> None:
    print("  [%s] %s" % ("ok" if condition else "FAILED", text))
    if not condition:
        FAILURES.append(text)


# ----------------------------------------------------------------------
#  A general test problem
# ----------------------------------------------------------------------
class SmoothSolution(ExactSolution):
    """u(t) = exp(-t) a + sin(3t) b + (1 + t)^3 c with fixed vectors a, b, c."""

    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def value(self, t):
        return math.exp(-t) * self.a + math.sin(3 * t) * self.b + (1 + t) ** 3 * self.c

    def derivative(self, t):
        return -math.exp(-t) * self.a + 3 * math.cos(3 * t) * self.b + 3 * (1 + t) ** 2 * self.c


class FlatSolution(ExactSolution):
    """u(t) = t^7 (exp(-t) a + cos(2t) b): all derivatives of order < 7 vanish at t = 0."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def value(self, t):
        return t ** 7 * (math.exp(-t) * self.a + math.cos(2 * t) * self.b)

    def derivative(self, t):
        return (7 * t ** 6 * (math.exp(-t) * self.a + math.cos(2 * t) * self.b)
                + t ** 7 * (-math.exp(-t) * self.a - 2 * math.sin(2 * t) * self.b))


def random_problem(dimension=30, seed=1, flat=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((dimension, dimension))
    A = X @ X.T + dimension * np.eye(dimension)              # symmetric positive definite
    A *= 50.0 / np.linalg.eigvalsh(A).max()                  # moderately stiff
    B1, B2 = rng.standard_normal((2, dimension, dimension)) / math.sqrt(dimension)
    vectors = rng.standard_normal((3, dimension))
    exact = FlatSolution(*vectors[:2]) if flat else SmoothSolution(*vectors)
    return MatrixEvolutionProblem(A, B=lambda t: math.sin(2 * t) * B1 + math.cos(t) * B2,
                                  weight=0.5, exact_solution=exact, name="random test problem")


def main() -> int:
    problem = random_problem()
    T = 1.0
    b = problem.bound_B(T)
    print("Test problem: dimension %d, nu = %.3f, b = %.3f" % (problem.dimension, problem.nu, b))

    # 1. resolvent
    c = 0.1
    g = np.arange(problem.dimension, dtype=float)
    direct = np.linalg.solve(np.eye(problem.dimension) + c * problem.matrix_A, g)
    check(np.allclose(problem.resolvent(c)(g), direct, rtol=1e-12, atol=1e-12),
          "resolvent (I + c A)^{-1} agrees with Gaussian elimination")

    # 2. Theorem 3.1
    tau, t = 0.05, 0.3
    F = problem.exact(t)
    solver = LayerIteration(problem, tau, b, tolerance=1e-13)
    history = []
    w = solver.solve(t, F, np.zeros(problem.dimension), history=history)
    reference = problem.direct_layer_solve(tau, t, F)
    errors = [problem.norm(x - reference) for x in history]
    check(problem.norm(w - reference) <= 1e-12, "iterations (3.4) converge to the solution of (3.1)")
    check(all(e1 <= solver.q * e0 * (1 + 1e-9) + 1e-14 for e0, e1 in zip(errors, errors[1:])),
          "error reduced by at least the factor q = %.3f in every iteration (3.9)" % solver.q)

    # 3. Lemma 5.1
    algorithm = PerturbationAlgorithm(problem, T, 80, bound_B=b)
    solution = algorithm.run(*exact_starting_values(problem, algorithm.grid))
    defect, size = residual_identity_defect(problem, solution)
    check(defect <= 1e-6 * size, "Lemma 5.1: residual of (2.9) on v equals R~_k of (5.2) "
          "(defect %.1e, max |R~| %.1e)" % (defect, size))

    # 4. orders of convergence
    steps = (20, 40, 80, 160, 320)
    for flat in (True, False):
        test = random_problem(flat=flat)
        bt = test.bound_B(T)
        e0, e1, e2, e3 = [], [], [], []
        for n in steps:
            algorithm = PerturbationAlgorithm(test, T, n, bound_B=bt)
            start = exact_starting_values(test, algorithm.grid)
            solution = algorithm.run(*start)
            bdf = FourLayerScheme(test, T, n, bound_B=bt).run(*start)
            grid = algorithm.grid
            e0.append(max_error(test, grid, solution.first_order()))
            e1.append(max_error(test, grid, solution.second_order()))
            e2.append(max_error(test, grid, solution.third_order(), first_layer=2))
            e3.append(max_error(test, grid, bdf.u, first_layer=3))
        if flat:
            print("  solution with vanishing derivatives at t = 0:")
            for name, errors, order in (("u0", e0, 1), ("u0 + tau u1", e1, 2), ("v", e2, 3),
                                        ("four-layer scheme", e3, 3)):
                observed = observed_orders(errors)[-1]
                check(abs(observed - order) < 0.15,
                      "order of %s: %.2f (expected %d)" % (name, observed, order))
        else:
            print("  general smooth solution (u'(0) != 0):")
            orders_v = observed_orders(e2)[1:]
            tail = orders_v[-3:]                  # the first order is pre-asymptotic
            check(all(o1 > o0 for o0, o1 in zip(tail, tail[1:])) and 2.5 < tail[-1] < 3.1,
                  "observed orders of v: " + ", ".join("%.2f" % o for o in orders_v)
                  + " (increasing towards 3)")
            observed = observed_orders(e3)[-1]
            check(abs(observed - 3) < 0.15, "order of the four-layer scheme: %.2f (expected 3)" % observed)

    # 5. extrapolated starting values
    test = random_problem(flat=False)
    bt = test.bound_B(T)
    algorithm = PerturbationAlgorithm(test, T, 80, bound_B=bt)
    exact_start = max_error(test, algorithm.grid,
                            algorithm.run(*exact_starting_values(test, algorithm.grid)).v, 2)
    extrapolated = max_error(test, algorithm.grid,
                             algorithm.run(*extrapolated_starting_values(test, algorithm.grid.tau, bt)).v, 2)
    check(abs(extrapolated - exact_start) <= 0.01 * exact_start,
          "extrapolated starting values: error %.3e (exact starting values: %.3e)"
          % (extrapolated, exact_start))

    # 6. model problem
    model = example_1(31)
    Q, lam = model._Q, model.eigenvalues
    check(np.allclose(model.matrix_A @ Q, Q * lam, atol=1e-8 * lam.max()),
          "model problem: sine-transform eigenpairs of A_h")
    v = np.linspace(-1.0, 1.0, model.n_nodes)
    check(np.allclose(model.apply_A(v), model.matrix_A @ v), "model problem: stencil of A_h")
    check(np.allclose(model.apply_B(0.3, v), model.matrix_B(0.3) @ v), "model problem: B_h(t)")

    print()
    if FAILURES:
        print("%d check(s) FAILED." % len(FAILURES))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
