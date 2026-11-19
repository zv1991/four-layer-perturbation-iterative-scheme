# -*- coding: utf-8 -*-
"""
diagnostics.py -- quantities of the analysis of the article (optional).

This module is not needed for the figures and tables of the article; it
evaluates, for a given problem and a computed solution, the quantities that
appear in the hypotheses and in the estimates of Sections 3-6:

    step_conditions()          ratio q (3.5) and condition (3.7) of Theorem 3.1;
                               tau b <= 1/2 (Lemmas 4.2, 4.4, Theorem 5.2);
                               c_0 b tau <= 1/2 (Theorems 6.2, 6.3);
    associated_polynomials_bound()
                               a numerical value of sup_k max_{0<=x<=1} |P_k(x)|,
                               where U_k = P_k(Lambda~) are the operator
                               polynomials (6.2) of Lemma 6.1 (the analysis uses only
                               the existence of the bound c_0, (6.3));
    starting_quantities()      M_0, M_1, M_2 of (5.3) (Theorem 5.2, Theorem 6.3);
    residual_identity_defect() check of Lemma 5.1: the grid function v of (2.15)
                               satisfies the four-layer scheme with the residual
                               R~_k(tau) of (5.2).

Usage:   python diagnostics.py      prints a report for the model problem.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evolution_problem import EvolutionProblem                                   # noqa: E402
from four_layer_scheme import four_layer_residual                                # noqa: E402
from grid_functions import backward_difference                                   # noqa: E402
from perturbation_algorithm import PerturbationSolution                         # noqa: E402


# ======================================================================
#  Hypotheses of the theorems
# ======================================================================
def step_conditions(tau: float, b: float, nu: float, c0: Optional[float] = None) -> Dict[str, object]:
    """
    Step-size conditions of the article for the step tau:

        q = tau b / (1 + tau nu) < 1            Theorem 3.1, (3.5), (3.7)
        tau b <= 1/2                             Lemmas 4.2, 4.4, Theorem 5.2
        c_0 b tau <= 1/2                         Theorems 6.2, 6.3 (if c_0 is given)
    """
    q = tau * b / (1.0 + tau * nu)
    result = {"q": q, "theorem_1": q < 1.0, "lemma_2": tau * b <= 0.5}
    if c0 is not None:
        result["theorem_3"] = c0 * b * tau <= 0.5
    return result


def minimal_step_numbers(t_final: float, b: float, c0: Optional[float] = None) -> Dict[str, int]:
    """Smallest n (tau = T / n) for which tau b <= 1/2 and c_0 b tau <= 1/2 hold."""
    result = {"lemma_2": int(math.ceil(2.0 * b * t_final))}
    if c0 is not None:
        result["theorem_3"] = int(math.ceil(2.0 * c0 * b * t_final))
    return result


# ======================================================================
#  Lemma 6.1: associated operator polynomials
# ======================================================================
def associated_polynomials_bound(n_terms: int = 2000, n_points: int = 20001) -> Tuple[float, int, float]:
    """
    Numerical value of sup_k max_{0 <= x <= 1} |P_k(x)|, where the polynomials

        P_k(x) = 18/11 x P_{k-1}(x) - 9/11 x P_{k-2}(x) + 2/11 x P_{k-3}(x),
        P_0 = 1,  P_{-1} = P_{-2} = 0,

    are the scalar versions of the operator polynomials U_k of (6.2)
    (U_k = P_k(Lambda~) with the self-adjoint operator Lambda~ = (I + 6/11 tau A)^{-1},
    whose spectrum lies in [0, 1]).  Returns (bound, k, x) of the maximum on a
    grid of n_points points and for k <= n_terms.  This is a numerical
    observation, not a proof; the analysis uses only the existence of c_0.
    """
    x = np.linspace(0.0, 1.0, n_points)
    p_km2, p_km1, p_k = np.zeros_like(x), np.zeros_like(x), np.ones_like(x)
    best, best_k, best_x = 1.0, 0, 0.0
    for k in range(1, n_terms + 1):
        p_new = x * (18.0 * p_k - 9.0 * p_km1 + 2.0 * p_km2) / 11.0
        p_km2, p_km1, p_k = p_km1, p_k, p_new
        i = int(np.argmax(np.abs(p_k)))
        if abs(p_k[i]) > best:
            best, best_k, best_x = float(abs(p_k[i])), k, float(x[i])
    return best, best_k, best_x


# ======================================================================
#  Theorem 5.2: quantities M_0, M_1, M_2
# ======================================================================
def starting_quantities(problem: EvolutionProblem, solution: PerturbationSolution) -> Tuple[float, float, float]:
    """
    The quantities (5.3) formed from the first layers of the grid functions:

        M_0 = sum_{j=0}^{6} ||delta^j u0_j||,
        M_1 = sum_{j=0}^{4} ||delta^j u1_{j+1}||,
        M_2 = sum_{j=0}^{2} ||delta^j u2_{j+2}||.

    Theorem 6.3 gives the order tau^3 if they are bounded uniformly in tau.
    """
    tau = solution.tau
    M0 = sum(problem.norm(backward_difference(solution.u0, j, j, tau)) for j in range(7))
    M1 = sum(problem.norm(backward_difference(solution.u1, j + 1, j, tau)) for j in range(5))
    M2 = sum(problem.norm(backward_difference(solution.u2, j + 2, j, tau)) for j in range(3))
    return M0, M1, M2


# ======================================================================
#  Lemma 5.1: residual of the four-layer scheme on v
# ======================================================================
def residual_of_lemma_4(solution: PerturbationSolution, k: int):
    """R~_k(tau) = tau^3 (1/2 delta^2 u2_k + 1/3 delta^3 u1_k + tau/3 delta^3 u2_k), (5.2)."""
    tau = solution.tau
    return tau ** 3 * (0.5 * backward_difference(solution.u2, k, 2, tau)
                       + backward_difference(solution.u1, k, 3, tau) / 3.0
                       + tau / 3.0 * backward_difference(solution.u2, k, 3, tau))


def residual_identity_defect(problem: EvolutionProblem, solution: PerturbationSolution) -> Tuple[float, float]:
    """
    max_{5 <= k <= n} || r_k(v) - R~_k ||  and  max_k ||R~_k||, where r_k(v) is the
    residual of scheme (2.9) on v.  By Lemma 5.1 the first number vanishes up to
    the errors of the iterations and rounding.
    """
    defect, size = 0.0, 0.0
    for k in range(5, solution.grid.n + 1):
        R = residual_of_lemma_4(solution, k)
        r = four_layer_residual(problem, solution.grid, solution.v, k)
        defect = max(defect, problem.norm(r - R))
        size = max(size, problem.norm(R))
    return defect, size


# ======================================================================
#  Report for the model problem
# ======================================================================
def main() -> None:
    from grid_functions import exact_starting_values
    from model_problem import example_1, example_2
    from perturbation_algorithm import PerturbationAlgorithm

    t_final = 1.0
    problem_1, problem_2 = example_1(), example_2()
    b, nu = problem_1.bound_B(t_final), problem_1.nu
    c0, k0, x0 = associated_polynomials_bound()
    print("Model problem (Section 7):  N = %d,  h = 1/%d" % (problem_1.n_nodes, problem_1.n_nodes + 1))
    print("  nu = %.6f,  b = %.6f  (b < nu: the iterations converge for every tau, Remark 3.2)"
          % (nu, b))
    print("  sup_k max_x |P_k(x)| ~ %.6f  (k = %d, x = %.4f)" % (c0, k0, x0))
    steps = minimal_step_numbers(t_final, b, c0)
    print("  tau b <= 1/2      (Lemmas 4.2, 4.4, Theorem 5.2):  n >= %d" % steps["lemma_2"])
    print("  c_0 b tau <= 1/2  (Theorems 6.2, 6.3):           n >= %d" % steps["theorem_3"])
    print()
    print("  %6s  %-34s  %-34s  %s" % ("n", "M_0, M_1, M_2 (Example 1)",
                                       "M_0, M_1, M_2 (Example 2)", "Lemma 5.1 defect / max |R~|"))
    for n in (10, 20, 40, 80, 160, 320):
        row = []
        for problem in (problem_1, problem_2):
            algorithm = PerturbationAlgorithm(problem, t_final, n, bound_B=b)
            solution = algorithm.run(*exact_starting_values(problem, algorithm.grid))
            row.append(starting_quantities(problem, solution))
            if problem is problem_1:
                # Lemma 5.1 is checked in Example 1
                defect, size = residual_identity_defect(problem, solution)
        print("  %6d  %-34s  %-34s  %.1e / %.1e" % (
            n, ", ".join("%.2e" % m for m in row[0]), ", ".join("%.2e" % m for m in row[1]),
            defect, size))


if __name__ == "__main__":
    main()
