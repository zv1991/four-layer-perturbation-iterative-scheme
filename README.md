# Perturbation algorithm and iterative realization of a four-layer semi-discrete scheme

Python implementation of the algorithms and of the numerical experiments of the article

> D. Gulua, J. Rogava, Z. Vashakidze,
> *Perturbation algorithm and iterative realization of a four-layer semi-discrete scheme
> for an evolutionary problem with a variable operator*,
> Proceedings of I. Vekua Institute of Applied Mathematics, Vol. 76, 2026.

* **Journal version** (Proceedings of I. Vekua Institute of Applied Mathematics, Vol. 76, 2026):
  <https://viam.science.tsu.ge/new/index.php?lang=eng&page=publishing&subpage=proceedings&vol=76&n=>
* **Preprint version** (arXiv): <https://arxiv.org/abs/TBA>

Repository: <https://github.com/zv1991/four-layer-perturbation-iterative-scheme> (MIT License, see [`LICENSE`](LICENSE)).

The code depends only on [NumPy](https://numpy.org). It is written for a **general**
evolutionary problem with a variable operator; the model problem of Section 7 of the
article is one particular case. On Windows, the script [`run_Python.bat`](run_Python.bat)
checks the requirements, tests the implementation and reproduces all numerical results of
the article in one step (see [Section 6](#6-installation-and-usage)).

**Numbering.** All equation, theorem, lemma, remark, table and figure numbers below (and in
the comments of the Python files) refer to the **arXiv preprint version**, in which they are
numbered within sections: equation (2.9) is the ninth equation of Section 2, Theorem 3.1 is
the first numbered statement of Section 3 (theorems, lemmas and remarks share one counter),
Table 7.1 is the first table of Section 7, etc. In the **journal version** they are numbered
consecutively through the article; the sections are the same in both versions. The
correspondence is given in [Section 9](#9-numbering-in-the-journal-and-preprint-versions).

---

## Contents

1. [The problem](#1-the-problem)
2. [The algorithms](#2-the-algorithms)
3. [Structure of the code](#3-structure-of-the-code)
4. [Description of the files](#4-description-of-the-files)
5. [The model problem and the experiments of Section 7](#5-the-model-problem-and-the-experiments-of-section-7)
6. [Installation and usage](#6-installation-and-usage)
7. [Using the code for another problem](#7-using-the-code-for-another-problem)
8. [Map from the article to the code](#8-map-from-the-article-to-the-code)
9. [Numbering in the journal and preprint versions](#9-numbering-in-the-journal-and-preprint-versions)
10. [How to cite](#10-how-to-cite)
11. [Authors](#11-authors)

---

## 1. The problem

In a Hilbert space $H$ with the inner product $(\cdot,\cdot)$ and the norm $\|\cdot\|$,
the Cauchy problem

```math
\frac{du(t)}{dt} + L(t)\,u(t) = f(t), \quad t \in (0,T], \qquad u(0) = u_0,
\qquad L(t) = A + B(t),
```

is considered (equations (2.1)–(2.2)), where

* $A$ is a self-adjoint positive definite (in general, unbounded) operator,
  $(Au,u) \ge \nu \|u\|^2$, $\nu > 0$ (inequality (2.3));
* $B(t)$ is a bounded operator depending on time, $b = \max_{0\le t\le T}\|B(t)\|$
  (definition (2.4)); it need not be self-adjoint, of constant sign, or commute with $A$.

On the grid $t_k = k\tau$, $k = 0,\dots,n$, $\tau = T/n$, the backward difference
quotients of a grid function $w_k$ are

```math
\delta w_k = \frac{w_k - w_{k-1}}{\tau}, \qquad
\delta^m w_k = \frac{1}{\tau^m}\sum_{i=0}^{m} (-1)^i \binom{m}{i} w_{k-i}.
```

## 2. The algorithms

### 2.1 The four-layer scheme (BDF3)

The purely implicit four-layer semi-discrete scheme of the third order of approximation
(the three-step backward differentiation formula), scheme (2.9):

```math
\frac{\tfrac{11}{6}u_k - 3u_{k-1} + \tfrac32 u_{k-2} - \tfrac13 u_{k-3}}{\tau}
+ \bigl(A + B(t_k)\bigr) u_k = f(t_k), \qquad k = 3,\dots,n .
```

### 2.2 The perturbation algorithm

Scheme (2.9) is embedded in the one-parameter family (2.10)

```math
\delta u_k + L_k u_k + \frac{\varepsilon}{2}\,\delta^2 u_k + \frac{\varepsilon^2}{3}\,\delta^3 u_k = f_k,
```

which coincides with (2.9) for $\varepsilon = \tau$. The expansion
$u_k = \sum_j \varepsilon^j u_k^{(j)}$ (2.11) gives three **two-layer (implicit Euler) schemes**
(2.12)–(2.14):

```math
\begin{aligned}
\frac{u_k^{(0)} - u_{k-1}^{(0)}}{\tau} + (A + B_k)\,u_k^{(0)} &= f_k, & u_0^{(0)} &= u_0,\ k = 1,\dots,n,\\
\frac{u_k^{(1)} - u_{k-1}^{(1)}}{\tau} + (A + B_k)\,u_k^{(1)} &= -\tfrac12\,\delta^2 u_k^{(0)}, & k &= 2,\dots,n,\\
\frac{u_k^{(2)} - u_{k-1}^{(2)}}{\tau} + (A + B_k)\,u_k^{(2)} &= -\tfrac12\,\delta^2 u_k^{(1)} - \tfrac13\,\delta^3 u_k^{(0)}, & k &= 3,\dots,n,
\end{aligned}
```

and the approximate solution (2.15)

```math
v_k = u_k^{(0)} + \tau\, u_k^{(1)} + \tau^2 u_k^{(2)} \approx u(t_k), \qquad k = 2,\dots,n .
```

**Starting vectors** (as in Rogava–Gulua, Georgian Math. J. 25 (2018), [doi:10.1515/gmj-2017-0061](https://doi.org/10.1515/gmj-2017-0061)): from given
approximations $v_1 \approx u(t_1)$, $v_2 \approx u(t_2)$ of accuracy $O(\tau^3)$,

```math
u_1^{(1)} = \frac{v_1 - u_1^{(0)}}{\tau}, \qquad
u_2^{(2)} = \frac{v_2 - u_2^{(0)} - \tau u_2^{(1)}}{\tau^2}.
```

$u_k^{(0)}$ is of the first order, $u_k^{(0)} + \tau u_k^{(1)}$ of the second order, and,
under the assumptions of Theorem 6.3 (in particular, the quantities $M_0, M_1, M_2$ of (5.3)
are bounded uniformly in $\tau$), $v_k$ is of the third order.

### 2.3 Iterative realization (Section 3)

Each layer of each scheme requires the solution of an equation of the form (3.1)

```math
(I + cA + cB_k)\,w = F_k ,
```

with $c = \tau$ for the two-layer schemes (with $F_k = u_{k-1}^{(j)} + \tau g_k^{(j)}$ and
$g^{(j)}$ of (3.2)) and $c = \tfrac{6}{11}\tau$ for the four-layer scheme (Remark 3.4, with
$F_k = \tfrac{1}{11}(18u_{k-1} - 9u_{k-2} + 2u_{k-3}) + cf_k$). The operator depends on $k$
and is not self-adjoint. The term with $B_k$ is **moved to the right-hand side** and the
equation is solved by the iterations (3.4)

```math
(I + cA)\,w^{[s+1]} = F_k - c\,B_k\,w^{[s]}, \qquad s = 0,1,\dots
```

so that only the time-independent operator $I + cA$ is inverted, on all layers and in all
iterations. With $S = (I + cA)^{-1}$ and $q = cb/(1 + c\nu)$ (3.5), **Theorem 3.1**: if
$q < 1$ (3.7), the iterations converge and

```math
\|w^{[s]} - w\| \le q^s \|w^{[0]} - w\|, \qquad
\|w^{[s]} - w\| \le \frac{q}{1-q}\,\|w^{[s]} - w^{[s-1]}\|.
```

**Stopping criterion** (Remark 3.3): stop when
$\|w^{[s]} - w^{[s-1]}\| \le (1-q)\,\mathrm{tol}/q$; then $\|w^{[s]} - w\| \le \mathrm{tol}$.
The initial approximation is the value on the preceding layer, $w^{[0]} = u_{k-1}^{(j)}$.
Alternatively, a prescribed number $s$ of iterations is performed on every layer (no bound
for $\|B(t)\|$ is needed then).

## 3. Structure of the code

```
four-layer-perturbation-iterative-scheme/
├── evolution_problem.py       problem interface; general finite-dimensional problem
├── model_problem.py           model problem (7.1) of Section 7, Examples 1 and 2
├── grid_functions.py          time grid, difference quotients, errors, orders
├── layer_iteration.py         iterations (3.4) of Section 3, Theorem 3.1, Remarks 3.3-3.4
├── perturbation_algorithm.py  class PerturbationAlgorithm: schemes (2.12)-(2.15)
├── four_layer_scheme.py       class FourLayerScheme: scheme (2.9), Remark 3.4
├── run_experiments.py         experiments of Section 7 -> data/*.csv  (main script)
├── diagnostics.py             optional: hypotheses, M_0, M_1, M_2, Lemmas 5.1 and 6.1
├── self_test.py               checks of the implementation on a general problem
├── check_requirements.py      checks/installs the packages of requirements.txt
├── requirements.txt           numpy>=1.20
├── run_Python.bat             Windows: requirements, self-test and experiments in one step
├── data/                      CSV files with the results of Tables 7.1-7.4, Figures 7.1-7.3
├── LICENSE                    MIT License
└── README.md                  this file
```

Dependencies between the modules (an arrow means "uses"):

```
run_experiments ──► perturbation_algorithm ──► layer_iteration ──► evolution_problem
       │                    │                                           ▲
       │                    └──► grid_functions                         │
       ├──────────► four_layer_scheme ──► layer_iteration               │
       └──────────► model_problem ──────────────────────────────────────┘
```

The algorithms (`layer_iteration`, `perturbation_algorithm`, `four_layer_scheme`) know
nothing about the model problem: they use only the interface `EvolutionProblem`.

## 4. Description of the files

### `evolution_problem.py` — the problem

* **`EvolutionProblem`** (abstract base class). The algorithms access a problem only
  through these methods, which are exactly what the iterative realization of Section 3
  needs:

  | method | meaning | used in |
  |---|---|---|
  | `inner(v, w)`, `norm(v)` | inner product and norm of $H$ | stopping criterion, errors |
  | `nu` | lower bound $\nu$ of $A$, (2.3) | ratio $q$ of (3.5) |
  | `resolvent(c)` | the operator $S = (I + cA)^{-1}$ as a function $g \mapsto Sg$ | iterations (3.4) |
  | `apply_B(t, v)` | $B(t)v$ (only applied to known vectors) | iterations (3.4) |
  | `bound_B(T)` | $b = \max_t \|B(t)\|$, (2.4) | ratio $q$ of (3.5) |
  | `f(t)`, `initial_value()` | data $f(t)$ and $u_0$ | schemes (2.12), (2.9) |
  | `apply_A(v)` | $Av$ (manufactured right-hand sides and checks only) | — |
  | `exact(t)`, `error(t, v)` | known solution (optional) | experiments |

  $A + B(t)$ is never inverted, and $B(t)$ is never inverted at all.

* **`MatrixEvolutionProblem`**: $H = \mathbb{R}^N$ with $(v,w) = \omega\sum_i v_iw_i$, a
  symmetric positive definite matrix $A$ and a matrix function $B(t)$. The spectral
  decomposition $A = Q\,\mathrm{diag}(\lambda)\,Q^T$ (computed by `numpy.linalg.eigh`, or
  supplied, e.g. the discrete sine transform) gives $\nu = \min\lambda$ and the resolvents
  $(I + cA)^{-1} = Q\,\mathrm{diag}\bigl(1/(1 + c\lambda)\bigr)Q^T$; each resolvent matrix is
  computed **once for every value of $c$** and cached. `bound_B(T)` evaluates the spectral
  norm of $B(t)$ at 401 equally spaced points of $[0,T]$ (as in Section 7).
  `direct_layer_solve(c, t, F)` solves (3.1) by Gaussian elimination; it serves only as a
  reference in the experiments.

* **`ExactSolution`**: optional known solution $u(t)$, $u'(t)$. If a problem is given an
  exact solution and no right-hand side, then $f(t) = u'(t) + (A + B(t))u(t)$ and
  $u_0 = u(0)$ (method of manufactured solutions).

### `model_problem.py` — the model problem of Section 7

`ModelProblem` is a subclass of `MatrixEvolutionProblem` for problem (7.1) after the space
discretization (see [Section 5](#5-the-model-problem-and-the-experiments-of-section-7)):
$A_h = h^{-2}\,\mathrm{tridiag}(-1,2,-1)$ with its explicitly known eigenpairs
$Q_{ij} = \sqrt{2h}\sin(\pi h ij)$, $\lambda_j = 4h^{-2}\sin^2(\pi hj/2)$, and $B_h(t)$
applied in its structured form (diagonal part plus dense kernel). The temporal factors
`ExponentialFactor`, `CosineFactor`, `FlatFactor` and the class `ModeSolution`
($u(x_i,t) = \sum_m g_m(t)\sin(m\pi x_i)$) define the manufactured solutions;
`example_1()` and `example_2()` create the two examples.

### `grid_functions.py` — grid functions

`TimeGrid` ($t_k = k\tau$), `backward_difference(w, k, m, tau)` ($\delta^m w_k$, using the
binomial formula above), `max_error`, `error_history`, `observed_orders`
($\log_2(E(\tau)/E(\tau/2))$) and `exact_starting_values`.

### `layer_iteration.py` — Section 3

`LayerIteration(problem, c, bound_B, tolerance=..., sweeps=None)` solves
$(I + cA + cB(t))w = F$ by the iterations (3.4):

1. `__init__` obtains $S = (I + cA)^{-1}$ from `problem.resolvent(c)` — the **only**
   inverted operator — computes $q = cb/(1 + c\nu)$ (3.5), checks condition (3.7) $q < 1$,
   and the threshold $(1-q)\,\mathrm{tol}/q$ of the stopping criterion (Remark 3.3).
2. `solve(t, F, initial)` iterates $w^{[s+1]} = S(F - cB(t)w^{[s]})$ from
   $w^{[0]} = $ `initial` until $\|w^{[s]} - w^{[s-1]}\|$ is below the threshold, or
   exactly `sweeps` times; the iterates can be recorded (`history`) for Figure 7.3.
3. `mean_sweeps` is the mean number of iterations (inversions of $I + cA$) per equation.

The same class serves the two-layer schemes ($c = \tau$) and the four-layer scheme
($c = 6\tau/11$, Remark 3.4).

### `perturbation_algorithm.py` — the perturbation algorithm

`PerturbationAlgorithm(problem, T, n, bound_B=..., tolerance=..., sweeps=None)`:

1. creates the grid and **one** `LayerIteration` with $c = \tau$, shared by the three
   schemes (they all invert the same operator $I + \tau A$);
2. `run(v1, v2)` computes the grid functions **layer by layer**, as stated in Section 2:
   on the layer $t_k$
   * scheme (2.12): $u_k^{(0)}$ with $g^{(0)}_k = f_k$;
   * at $k = 1$: the starting vector $u_1^{(1)} = (v_1 - u_1^{(0)})/\tau$;
   * scheme (2.13): $u_k^{(1)}$ with $g^{(1)}_k = -\tfrac12\delta^2u_k^{(0)}$;
   * at $k = 2$: the starting vector $u_2^{(2)} = (v_2 - u_2^{(0)} - \tau u_2^{(1)})/\tau^2$;
   * scheme (2.14): $u_k^{(2)}$ with $g^{(2)}_k = -\tfrac12\delta^2u_k^{(1)} - \tfrac13\delta^3u_k^{(0)}$;

   every layer equation is (3.1) with $F_k = u_{k-1}^{(j)} + \tau g_k^{(j)}$, solved by the
   iterations (3.4) with $w^{[0]} = u_{k-1}^{(j)}$;
3. forms $v_k = u_k^{(0)} + \tau u_k^{(1)} + \tau^2 u_k^{(2)}$ (2.15) and returns a
   `PerturbationSolution` with `u0`, `u1`, `u2`, `v`, `first_order()`, `second_order()`,
   `third_order()`, `mean_sweeps`, `q`.

`extrapolated_starting_values(problem, tau, b)` provides $v_1, v_2$ with error $O(\tau^3)$
for problems without a known solution: the implicit Euler scheme (2.12) with the steps
$h, h/2, h/4$ ($h = \tau/4$) on $[0, 2\tau]$, combined by Richardson extrapolation
$(8E_{h/4} - 6E_{h/2} + E_h)/3$. (The article assumes that $v_1, v_2$ are given; this is
one way to obtain them. In the model problem it gives the same errors as the exact
values.)

### `four_layer_scheme.py` — the four-layer scheme

`FourLayerScheme(problem, T, n, ...)` with $c = 6\tau/11$; `run(u1, u2)` computes, for
$k = 3,\dots,n$, $F_k = \tfrac{1}{11}(18u_{k-1} - 9u_{k-2} + 2u_{k-3}) + cf_k$ (scheme (2.9)
multiplied by $6\tau/11$) and solves $(I + cA + cB_k)u_k = F_k$ by the iterations of
Remark 3.4 with $w^{[0]} = u_{k-1}$. `four_layer_residual(problem, grid, w, k)` evaluates the
residual of (2.9) on any grid function.

### `run_experiments.py` — the experiments of Section 7

Runs four experiments and writes the CSV files read by the LaTeX figures and tables of the
article to the folder `data` next to the script (see
[Section 5](#5-the-model-problem-and-the-experiments-of-section-7)). Options:
`--list-outputs` (prints the names of the CSV files; used by `run_Python.bat`),
`--data-dir DIR` (default: the folder `data` next to the script), `--no-progress`. The CSV files are written with `\n` line endings, so
they are identical on all operating systems.

### `diagnostics.py` — quantities of the analysis (optional)

Not needed for the article's figures and tables. `python diagnostics.py` prints, for the
model problem: $\nu$, $b$, a numerical value of $\sup_k\max_{0\le x\le 1}|P_k(x)|$ for the
associated polynomials (6.2) of Lemma 6.1, the smallest $n$ with $\tau b \le 1/2$ (Lemmas 4.2, 4.4,
Theorem 5.2) and $c_0 b\tau \le 1/2$ (Theorems 6.2, 6.3), the quantities $M_0, M_1, M_2$ of (5.3)
for both examples, and a numerical check of Lemma 5.1 (the residual of (2.9) on $v$ equals
$\widetilde R_k(\tau)$ of (5.2)).

### `self_test.py` — checks of the implementation

Uses a **general** problem (random symmetric positive definite $A$, random
non-symmetric $B(t)$, manufactured solution) and checks: the resolvent; Theorem 3.1
(convergence of (3.4), error reduced by at least $q$ per iteration); Lemma 5.1; the orders
1, 2, 3 of $u^{(0)}$, $u^{(0)} + \tau u^{(1)}$, $v$ and the order 3 of scheme (2.9) for a
solution with vanishing derivatives at $t = 0$; that for a general smooth solution the
observed order of $v$ increases towards 3 only gradually (the behaviour seen in
Example 2); the extrapolated starting values; and the matrices of the model problem.
Exit code 0 if all checks pass.

### `check_requirements.py`, `requirements.txt`

`check_requirements.py` reads `requirements.txt` (pip format, e.g. `numpy>=1.20`) using
only the standard library, imports every package and compares its version. Exit codes:
0 = all present, 1 = missing/outdated, 2 = installation failed, 3 = Python too old or
unreadable file. `--install` installs the missing packages with
`python -m pip install --upgrade ...` (if necessary, `ensurepip` first, and a retry with
`--user` outside virtual environments); `--missing` prints only the missing
specifications.

## 5. The model problem and the experiments of Section 7

Problem (7.1) in $H = L_2(0,1)$:

```math
\frac{\partial u}{\partial t} - \frac{\partial^2u}{\partial x^2} + \beta(x,t)\,u
+ \gamma(t)\int_0^1 e^{x-y}u(y,t)\,dy = f(x,t), \qquad u(0,t) = u(1,t) = 0,
```

$\beta(x,t) = 5\sin(2\pi t)\cos(\pi x)$, $\gamma(t) = 3e^{-t}$, $T = 1$. Space
discretization on $x_i = ih$, $h = 1/(N+1)$, $N = 127$: central difference quotient and
rectangle rule; $H_h = \mathbb{R}^N$ with $(v,w)_h = h\sum_i v_iw_i$. The right-hand side
is formed on the grid, $f_h = U' + (A_h + B_h(t))U$, so that $U(t) = (u(x_i,t))_i$ is the
exact solution of the semi-discrete problem and all errors are errors of the time
discretization. Iterations: $\mathrm{tol} = 10^{-13}$; $b_h$ is the maximum of
$\|B_h(t)\|_h$ at 401 equally spaced points of $[0,T]$; since $b_h < \nu_h$ (the smallest
eigenvalue of $A_h$), the iterations converge for every $\tau$ (Remark 3.2).

* **Example 1**: $u = t^7e^{-2t}\sin\pi x + \tfrac12 t^7e^{-t}\sin 2\pi x$.
* **Example 2**: $u = e^{-t}\sin\pi x + \cos 3t\,\sin 2\pi x$.

Starting values: $v_1 = u(t_1)$, $v_2 = u(t_2)$ (perturbation algorithm) and
$u_1 = u(t_1)$, $u_2 = u(t_2)$ (four-layer scheme); $\tau = 1/n$,
$n = 10, 20, \dots, 1280$.

| experiment | CSV file(s) in `data` | columns | used by |
|---|---|---|---|
| 1. Example 1, perturbation algorithm | `convergence_example1.csv` | `n, tau, err0, ord0, err1, ord1, err2, ord2, sweeps` ($E^{(0)}, E^{(1)}, E^{(2)}$, orders, mean iterations) | Table 7.1, Figure 7.1 |
| 2. Example 2, perturbation algorithm and scheme (2.9) | `convergence_example2.csv`, `error_history.csv` | `n, tau, err_pert, ord_pert, err_bdf, ord_bdf`; `t, err_pert, err_bdf` ($n = 160$, layers $k \ge 3$) | Table 7.2, Figure 7.2 |
| 3. Iterations (3.4) on the layer $t_k = 1/2$ of Example 2 | `iteration_decay.csv`, `iteration_statistics.csv` | `s, err_n10, bound_n10, ...` ($\|w^{[s]}-w\|$ and $q^s\|w^{[0]}-w\|$); `n, tau, q, q_obs, sweeps` | Figure 7.3, Table 7.3 |
| 4. Example 1, prescribed number $s$ of iterations | `sweeps_study.csv` | `n, tau, err_s1, ord_s1, ..., err_s4, ord_s4, err_conv, ord_conv` | Table 7.4 |
| constants | `constants.csv` | `nu, b` | (not used by LaTeX) |

The whole run takes a few seconds. The folder `data` of the repository contains the CSV
files used for the tables and figures of the article; running the experiments again
overwrites them. Values at the level of the rounding errors (e.g. errors of order
$10^{-10}$–$10^{-13}$) may differ in the 6th–7th significant digit between computers
(floating-point arithmetic, NumPy/BLAS version); the digits shown in the tables of the
article are not affected.

## 6. Installation and usage

Requirements: Python 3.8 or newer and NumPy 1.20 or newer (`requirements.txt`). Download
the repository (button *Code → Download ZIP* on GitHub) or clone it:

```bash
git clone https://github.com/zv1991/four-layer-perturbation-iterative-scheme.git
cd four-layer-perturbation-iterative-scheme
```

### Windows: `run_Python.bat`

Double-click `run_Python.bat`, or type `run_Python.bat` in a Command Prompt opened in the
folder of the repository. The script always works in its own folder (wherever it is called
from), so it must stay in the same folder as the Python files. It performs seven steps,
shown with a progress bar:

1. **finds a Python 3 interpreter** — the Python launcher `py -3` first, then `python` from
   `PATH` — and checks that its version is 3.8 or newer (otherwise it explains where to get
   Python);
2. **checks the required packages** of `requirements.txt` with `check_requirements.py`; if
   some are missing or outdated, it lists them and asks whether to install them with pip
   (answer `Y` or `N`; if there is no answer within 60 s, nothing is installed), installs
   them, and checks them again in a new Python process;
3. **runs the self-test** `self_test.py` (the run stops if a check fails);
4. **deletes the CSV files of the previous run** from the folder `data` — only the files
   whose names are reported by `run_experiments.py --list-outputs`, nothing else;
5. **runs the main script** `run_experiments.py --data-dir data`, which writes the new CSV
   files with the data of Tables 7.1–7.4 and Figures 7.1–7.3;
6. **checks that all expected CSV files have been written**;
7. **removes the folder `__pycache__`**, if any (all scripts are run with `python -B`, so
   normally no cache files are created).

At the end it reports `SUCCESS` (with the start and finish times) or `FAILED` and waits for
a key. `run_Python.bat /nopause` does not wait for a key (useful when the script is called
from another script). The exit code is 0 on success and 1 on failure. The whole run takes
about 10 seconds.

### Any operating system

From the folder of the repository (on Linux and macOS, `python3` instead of `python`):

```bash
python -m pip install -r requirements.txt   # once
python check_requirements.py                # optional: check the packages
python self_test.py                         # optional checks
python run_experiments.py                   # main script: writes data/*.csv
python diagnostics.py                       # optional report
```

## 7. Using the code for another problem

Any problem of the form (2.1)–(2.4) can be solved by giving the matrices (or by subclassing
`EvolutionProblem`, e.g. with a sparse factorization in `resolvent()`):

```python
import math
import numpy as np

from evolution_problem import MatrixEvolutionProblem
from four_layer_scheme import FourLayerScheme
from perturbation_algorithm import PerturbationAlgorithm, extrapolated_starting_values

# u_t - u_xx + cos(t) * int_0^1 exp(-|x - y|) u(y, t) dy = 1,  u(0) = sin(pi x)
N, h = 63, 1.0 / 64
x = h * np.arange(1, N + 1)
A = (2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)) / h**2      # A_h
K = h * np.exp(-np.abs(x[:, None] - x[None, :]))                    # rectangle rule

problem = MatrixEvolutionProblem(
    A,
    B=lambda t: math.cos(t) * K,       # bounded, time dependent
    f=lambda t: np.ones(N),            # right-hand side f(t)
    u0=np.sin(math.pi * x),            # initial value
    weight=h,                          # (v, w) = h * sum_i v_i w_i
    name="my problem")

T, n = 1.0, 200
b = problem.bound_B(T)                                  # b of (2.4)
algorithm = PerturbationAlgorithm(problem, T, n, bound_B=b, tolerance=1e-12)
v1, v2 = extrapolated_starting_values(problem, algorithm.grid.tau, b)
solution = algorithm.run(v1, v2)

print("q =", solution.q, " mean iterations per solve =", solution.mean_sweeps)
print("||v_n|| =", problem.norm(solution.v[-1]))

# comparison with the four-layer scheme (2.9), realized as in Remark 3.4
bdf = FourLayerScheme(problem, T, n, bound_B=b, tolerance=1e-12).run(v1, v2)
print("||v_n - u_n|| =", problem.norm(solution.v[-1] - bdf.u[-1]))
```

To use a problem in another representation (e.g. finite elements with a mass matrix, or
a sparse $A$), subclass `EvolutionProblem` and implement `inner`, `nu`, `apply_A`,
`resolvent`, `apply_B`, `bound_B`, `f` and `initial_value`; the classes
`PerturbationAlgorithm`, `FourLayerScheme` and `LayerIteration` need no changes.

## 8. Map from the article to the code

| article | code |
|---|---|
| problem (2.1)–(2.4) | `EvolutionProblem`, `MatrixEvolutionProblem` |
| difference quotients $\delta^m$ (Section 2) | `grid_functions.backward_difference` |
| four-layer scheme (2.9) | `FourLayerScheme.run` |
| two-layer schemes (2.12)–(2.14), starting vectors | `PerturbationAlgorithm.run` |
| approximate solution (2.15) | `PerturbationSolution.v` / `third_order()` |
| layer equation (3.1), right-hand sides (3.2) | `PerturbationAlgorithm._two_layer_step` |
| iterations (3.4), $S$ and $q$ (3.5) | `LayerIteration.solve`, `LayerIteration.__init__` |
| Theorem 3.1, condition (3.7), estimates (3.9)–(3.10) | `LayerIteration` (check of $q<1$, stopping criterion); `self_test.py` |
| Remark 3.3 (initial guess, stopping criterion) | `LayerIteration.solve(..., initial=u_{k-1})` |
| Remark 3.4 (iterations for scheme (2.9)) | `FourLayerScheme` ($c = 6\tau/11$) |
| Lemma 5.1, residual (5.2) | `diagnostics.residual_identity_defect` |
| quantities $M_0, M_1, M_2$ (5.3) | `diagnostics.starting_quantities` |
| polynomials (6.2), Lemma 6.1 | `diagnostics.associated_polynomials_bound` |
| model problem (7.1), Examples 1, 2 | `model_problem.ModelProblem`, `example_1`, `example_2` |
| Tables 7.1–7.4, Figures 7.1–7.3 | `run_experiments.py` → `data/*.csv` |

## 9. Numbering in the journal and preprint versions

The sections (1–7) are the same in both versions. The code and this file use the numbers of
the preprint version.

**Equations.**

| preprint (arXiv) | journal (Proceedings) |
|---|---|
| (2.1)–(2.15) | (1)–(15) |
| (3.1)–(3.10) | (16)–(25) |
| (4.1)–(4.8) | (26)–(33) |
| (5.1)–(5.8) | (34)–(41) |
| (6.1)–(6.8) | (42)–(49) |
| (7.1) | (50) |

For example, scheme (2.9) is (9), the iterations (3.4) are (19), the ratio $q$ (3.5) is
(20), condition (3.7) is (22), the quantities $M_0, M_1, M_2$ (5.3) are (36), and the model
problem (7.1) is (50).

**Theorems, lemmas, remarks, tables and figures.**

| preprint (arXiv) | journal (Proceedings) |
|---|---|
| Theorem 3.1 | Theorem 1 |
| Remark 3.2, Remark 3.3, Remark 3.4 | Remark 1, Remark 2, Remark 3 |
| Lemma 4.1, Lemma 4.2 | Lemma 1, Lemma 2 |
| Remark 4.3 | Remark 4 |
| Lemma 4.4 | Lemma 3 |
| Lemma 5.1 | Lemma 4 |
| Theorem 5.2 | Theorem 2 |
| Lemma 6.1 | Lemma 5 |
| Theorem 6.2, Theorem 6.3 | Theorem 3, Theorem 4 |
| Tables 7.1, 7.2, 7.3, 7.4 | Tables 1, 2, 3, 4 |
| Figures 7.1, 7.2, 7.3 | Figures 1, 2, 3 |

## 10. How to cite

If you use this code, please cite the article:

> D. Gulua, J. Rogava, Z. Vashakidze, Perturbation algorithm and iterative realization of a
> four-layer semi-discrete scheme for an evolutionary problem with a variable operator,
> *Proc. I. Vekua Inst. Appl. Math.* **76** (2026).
> Journal version: <https://viam.science.tsu.ge/new/index.php?lang=eng&page=publishing&subpage=proceedings&vol=76&n=>;
> preprint: <https://arxiv.org/abs/TBA>.

```bibtex
@article{GuluaRogavaVashakidze2026,
    AUTHOR = {Gulua, David and Rogava, Jemal and Vashakidze, Zurab},
     TITLE = {Perturbation algorithm and iterative realization of a four-layer
              semi-discrete scheme for an evolutionary problem with a variable
              operator},
   JOURNAL = {Proc. I. Vekua Inst. Appl. Math.},
  FJOURNAL = {Proceedings of I. Vekua Institute of Applied Mathematics},
    VOLUME = {76},
      YEAR = {2026},
       URL = {https://viam.science.tsu.ge/new/index.php?lang=eng&page=publishing&subpage=proceedings&vol=76&n=},
}
```

The perturbation algorithm for the four-layer scheme with a constant operator was studied in

> J. Rogava, D. Gulua, The perturbation algorithm for the realization of a four-layer
> semi-discrete solution scheme of an abstract evolutionary problem, *Georgian Math. J.*
> **25** (2018), no. 1, 77–92. MR3767397.
> [doi:10.1515/gmj-2017-0061](https://doi.org/10.1515/gmj-2017-0061)

## 11. Authors

* **David Gulua** — Department of Computational Mathematics, Georgian Technical University,
  Tbilisi, Georgia.
* **Jemal Rogava** — Faculty of Exact and Natural Sciences and I. Vekua Institute of Applied
  Mathematics, Ivane Javakhishvili Tbilisi State University, Tbilisi, Georgia.
* **Zurab Vashakidze** (corresponding author, maintainer of the repository) — School of
  Science and Technology, The University of Georgia, Tbilisi, Georgia; I. Vekua Institute of
  Applied Mathematics, Ivane Javakhishvili Tbilisi State University, Tbilisi, Georgia.
  E-mail: z.vashakidze@ug.edu.ge, zurab.vashakidze@tsu.ge
