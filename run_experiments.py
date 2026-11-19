# -*- coding: utf-8 -*-
"""
run_experiments.py -- numerical experiments of Section 7 of the article

    D. Gulua, J. Rogava, Z. Vashakidze,
    "Perturbation algorithm and iterative realization of a four-layer
     semi-discrete scheme for an evolutionary problem with a variable operator".

The script runs four experiments for the model problem (7.1)
(model_problem.py) and exports the data of all tables and figures of the
article to CSV files in the folder ``data`` (next to this script):

    Experiment 1  Example 1, perturbation algorithm:
                  errors E^(0), E^(1), E^(2) of u0_k, u0_k + tau u1_k, v_k
                  and observed orders                  -> Table 7.1, Figure 7.1
    Experiment 2  Example 2, perturbation algorithm and four-layer scheme (2.9):
                  maximal errors, observed orders, and the errors as
                  functions of t_k for n = 160          -> Table 7.2, Figure 7.2
    Experiment 3  convergence of the iterations (3.4) on the layer t_k = 1/2
                  of Example 2: errors ||w^[s] - w||, the bound q^s ||w^[0] - w||
                  of Theorem 3.1, the ratio q (3.5), the observed ratio, the mean
                  numbers of iterations                 -> Figure 7.3, Table 7.3
    Experiment 4  Example 1 with a prescribed number s of iterations on every
                  layer of each scheme                 -> Table 7.4

The LaTeX sources of the figures and tables of the article read these CSV
files with pgfplots / pgfplotstable.

Usage
-----
    python run_experiments.py                  run all experiments
    python run_experiments.py --list-outputs   print the names of the CSV files
    python run_experiments.py --data-dir DIR   write the CSV files to DIR
    python run_experiments.py --no-progress    no progress bar (plain log)

On Windows, run_Python.bat (in the same folder) checks the requirements and
runs this script.  Requirements: Python >= 3.8 and NumPy (requirements.txt).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                               # noqa: E402

from four_layer_scheme import FourLayerScheme                                    # noqa: E402
from grid_functions import (error_history, exact_starting_values, max_error,     # noqa: E402
                            observed_orders)
from layer_iteration import LayerIteration                                       # noqa: E402
from model_problem import ModelProblem, example_1, example_2                      # noqa: E402
from perturbation_algorithm import PerturbationAlgorithm                         # noqa: E402

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

#: CSV files written by this script (in the order in which they are written)
OUTPUT_FILES = (
    "constants.csv",
    "convergence_example1.csv",
    "convergence_example2.csv",
    "error_history.csv",
    "iteration_decay.csv",
    "iteration_statistics.csv",
    "sweeps_study.csv",
)


# ======================================================================
#  Settings of Section 7
# ======================================================================
@dataclass(frozen=True)
class Settings:
    t_final: float = 1.0                                   # T
    n_nodes: int = 127                                     # N, h = 1/(N + 1) = 1/128
    step_numbers: Sequence[int] = (10, 20, 40, 80, 160, 320, 640, 1280)   # tau = 1/n
    tolerance: float = 1.0e-13                             # tol of the stopping criterion
    history_steps: int = 160                               # Figure 7.2
    decay_steps: Sequence[int] = (10, 20, 40, 80)          # Figure 7.3
    decay_sweeps: int = 6                                  # iterations shown in Figure 7.3
    decay_layer: float = 0.5                               # layer t_k = 1/2 of Figure 7.3
    sweep_numbers: Sequence[int] = (1, 2, 3, 4)            # Table 7.4


# ======================================================================
#  Progress bar and output
# ======================================================================
class ProgressBar:
    """Text progress bar  [##########--------------]  42%  label  (redrawn in place)."""

    def __init__(self, total: int, width: int = 40, enabled: bool = True):
        self.total = max(int(total), 1)
        self.width = int(width)
        self.enabled = enabled
        self.done = 0
        self.label = ""

    def _draw(self) -> None:
        if not self.enabled:
            return
        fraction = min(self.done / self.total, 1.0)
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        line = "  [%s] %3d%%  %s" % (bar, int(round(100 * fraction)), self.label)
        sys.stdout.write("\r" + line[:110].ljust(110))
        sys.stdout.flush()

    def stage(self, label: str) -> None:
        self.label = label
        self._draw()

    def advance(self, units: int = 1) -> None:
        self.done += units
        self._draw()

    def message(self, text: str) -> None:
        """Print a line of the log above the bar."""
        if self.enabled:
            sys.stdout.write("\r" + " " * 110 + "\r")
        sys.stdout.write(text + "\n")
        self._draw()

    def finish(self) -> None:
        self.done = self.total
        self.label = "done"
        self._draw()
        if self.enabled:
            sys.stdout.write("\n")
        sys.stdout.flush()


class Output:
    """Writes the CSV files and reports them."""

    def __init__(self, data_dir: str, progress: ProgressBar):
        self.data_dir = data_dir
        self.progress = progress

    def write(self, name: str, header: Sequence[str], rows) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, name)
        with open(path, "w", newline="") as handle:
            # "\n" line endings on every operating system (identical files on
            # Windows and Linux, no spurious changes under version control)
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            for row in rows:
                writer.writerow([self.format(value) for value in row])
        try:
            shown = os.path.relpath(path)
        except ValueError:          # Windows: path on another drive
            shown = path
        self.progress.message("  written: %s" % shown)

    @staticmethod
    def format(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, np.integer)):
            return "%d" % value
        return "%.6e" % value


# ======================================================================
#  Experiments
# ======================================================================
def experiment_1(problem: ModelProblem, b: float, s: Settings, out: Output,
                 bar: ProgressBar) -> Dict[str, List[float]]:
    """Example 1: orders of u0_k, u0_k + tau u1_k and v_k (Table 7.1, Figure 7.1)."""
    bar.message("Experiment 1: %s, perturbation algorithm" % problem.name)
    e0, e1, e2, sweeps, taus = [], [], [], [], []
    for n in s.step_numbers:
        bar.stage("Experiment 1/4: n = %d" % n)
        algorithm = PerturbationAlgorithm(problem, s.t_final, n, bound_B=b, tolerance=s.tolerance)
        # starting values v_1 = u(t_1), v_2 = u(t_2), as in [Rogava, Gulua 2018]
        solution = algorithm.run(*exact_starting_values(problem, algorithm.grid))
        grid = algorithm.grid
        taus.append(grid.tau)
        e0.append(max_error(problem, grid, solution.first_order()))               # E^(0), k >= 1
        e1.append(max_error(problem, grid, solution.second_order()))              # E^(1), k >= 1
        e2.append(max_error(problem, grid, solution.third_order(), first_layer=2))  # E^(2), k >= 2
        sweeps.append(solution.mean_sweeps)
        bar.advance()
    rows = zip(s.step_numbers, taus, e0, observed_orders(e0), e1, observed_orders(e1),
               e2, observed_orders(e2), sweeps)
    out.write("convergence_example1.csv",
              ["n", "tau", "err0", "ord0", "err1", "ord1", "err2", "ord2", "sweeps"], rows)
    return {"tau": taus, "sweeps": sweeps}


def experiment_2(problem: ModelProblem, b: float, s: Settings, out: Output, bar: ProgressBar) -> None:
    """Example 2: perturbation algorithm and four-layer scheme (Table 7.2, Figure 7.2)."""
    bar.message("Experiment 2: %s, perturbation algorithm and four-layer scheme" % problem.name)
    e_pert, e_bdf, taus = [], [], []
    history = None
    for n in s.step_numbers:
        bar.stage("Experiment 2/4: n = %d" % n)
        algorithm = PerturbationAlgorithm(problem, s.t_final, n, bound_B=b, tolerance=s.tolerance)
        scheme = FourLayerScheme(problem, s.t_final, n, bound_B=b, tolerance=s.tolerance)
        grid = algorithm.grid
        starting = exact_starting_values(problem, grid)       # u(t_1), u(t_2)
        pert = algorithm.run(*starting)
        bdf = scheme.run(*starting)
        taus.append(grid.tau)
        e_pert.append(max_error(problem, grid, pert.v, first_layer=2))
        e_bdf.append(max_error(problem, grid, bdf.u, first_layer=3))
        if n == s.history_steps:
            history = (grid.t, error_history(problem, grid, pert.v), error_history(problem, grid, bdf.u))
        bar.advance(2)
    rows = zip(s.step_numbers, taus, e_pert, observed_orders(e_pert), e_bdf, observed_orders(e_bdf))
    out.write("convergence_example2.csv",
              ["n", "tau", "err_pert", "ord_pert", "err_bdf", "ord_bdf"], rows)
    # Figure 7.2; the first three layers are omitted: there the errors vanish
    # (exact starting values), which cannot be shown on a logarithmic scale
    t, hp, hb = history
    out.write("error_history.csv", ["t", "err_pert", "err_bdf"],
              [(t[k], hp[k], hb[k]) for k in range(3, len(t))])


def experiment_3(problem: ModelProblem, b: float, s: Settings, out: Output, bar: ProgressBar,
                 taus_1: Sequence[float], sweeps_1: Sequence[float]) -> None:
    """Convergence of the iterations (3.4) on one layer (Figure 7.3, Table 7.3)."""
    bar.message("Experiment 3: convergence of the iterations (%s, layer t = %g)"
                % (problem.name, s.decay_layer))
    decay, stats = {}, []
    for n in s.step_numbers:
        bar.stage("Experiment 3/4: n = %d" % n)
        algorithm = PerturbationAlgorithm(problem, s.t_final, n, bound_B=b, tolerance=s.tolerance)
        grid = algorithm.grid
        tau = grid.tau
        k = int(round(s.decay_layer / tau))
        # grid function u0 of scheme (2.12) on the layers 0, ..., k - 1
        u0 = algorithm.run(*exact_starting_values(problem, grid), n_layers=k - 1).u0
        # equation (3.1) of scheme (2.12) on the layer t_k
        F = u0[k - 1] + tau * problem.f(grid.t[k])
        w = problem.direct_layer_solve(tau, grid.t[k], F)          # reference (direct method)
        solver = LayerIteration(problem, tau, b, sweeps=s.decay_sweeps)
        iterates: List[np.ndarray] = []
        solver.solve(grid.t[k], F, u0[k - 1], history=iterates)    # w^[0] = u0_{k-1}
        errors = [problem.norm(it - w) for it in iterates]
        # observed ratio: errors after the second and after the first iteration
        # (both are far above the level of rounding errors)
        q_observed = errors[2] / errors[1]
        stats.append((n, tau, solver.q, q_observed))
        if n in s.decay_steps:
            decay[n] = (errors, [errors[0] * solver.q ** j for j in range(s.decay_sweeps + 1)])
        bar.advance()
    header = ["s"]
    for n in s.decay_steps:
        header += ["err_n%d" % n, "bound_n%d" % n]
    rows = []
    for j in range(s.decay_sweeps + 1):
        row = [j]
        for n in s.decay_steps:
            row += [decay[n][0][j], decay[n][1][j]]
        rows.append(row)
    out.write("iteration_decay.csv", header, rows)
    assert all(abs(a - b_) < 1e-15 for a, b_ in zip(taus_1, [r[1] for r in stats]))
    out.write("iteration_statistics.csv", ["n", "tau", "q", "q_obs", "sweeps"],
              [(n, tau, q, q_obs, sw) for (n, tau, q, q_obs), sw in zip(stats, sweeps_1)])


def experiment_4(problem: ModelProblem, b: float, s: Settings, out: Output, bar: ProgressBar) -> None:
    """Example 1 with a prescribed number s of iterations (Table 7.4)."""
    bar.message("Experiment 4: %s, prescribed number of iterations" % problem.name)
    columns, header = [], ["n", "tau"]
    # sweeps = None: converged iterations (stopping criterion with tol)
    for sweeps in list(s.sweep_numbers) + [None]:
        errors = []
        for n in s.step_numbers:
            bar.stage("Experiment 4/4: s = %s, n = %d" % (sweeps if sweeps else "conv.", n))
            algorithm = PerturbationAlgorithm(problem, s.t_final, n, bound_B=b,
                                              tolerance=s.tolerance, sweeps=sweeps)
            solution = algorithm.run(*exact_starting_values(problem, algorithm.grid))
            errors.append(max_error(problem, algorithm.grid, solution.v, first_layer=2))
            bar.advance()
        columns += [errors, observed_orders(errors)]
        label = "s%d" % sweeps if sweeps else "conv"
        header += ["err_" + label, "ord_" + label]
    rows = [[n, s.t_final / n] + [column[i] for column in columns]
            for i, n in enumerate(s.step_numbers)]
    out.write("sweeps_study.csv", header, rows)


# ======================================================================
#  Main program
# ======================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Numerical experiments of Section 7 (CSV export).")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="output folder (default: the folder data next to this script)")
    parser.add_argument("--list-outputs", action="store_true",
                        help="print the names of the CSV files and exit")
    parser.add_argument("--no-progress", action="store_true", help="do not draw a progress bar")
    args = parser.parse_args(argv)

    if args.list_outputs:
        print("\n".join(OUTPUT_FILES))
        return 0

    s = Settings()
    started = time.time()
    problem_1, problem_2 = example_1(s.n_nodes), example_2(s.n_nodes)
    # the operators A_h, B_h(t) are the same in both examples
    b = problem_1.bound_B(s.t_final)
    print("Model problem:  N = %d,  h = 1/%d" % (s.n_nodes, s.n_nodes + 1))
    print("  nu  = lambda_min(A_h)     = %.6f" % problem_1.nu)
    print("  b   = max_t ||B_h(t)||    = %.6f" % b)

    n_runs = len(s.step_numbers)
    bar = ProgressBar(total=n_runs * (1 + 2 + 1 + len(s.sweep_numbers) + 1),
                      enabled=not args.no_progress)
    out = Output(args.data_dir, bar)
    out.write("constants.csv", ["nu", "b"], [(problem_1.nu, b)])
    print()

    result_1 = experiment_1(problem_1, b, s, out, bar)
    experiment_2(problem_2, b, s, out, bar)
    experiment_3(problem_2, b, s, out, bar, result_1["tau"], result_1["sweeps"])
    experiment_4(problem_1, b, s, out, bar)
    bar.finish()
    print("All experiments finished in %.1f s." % (time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
