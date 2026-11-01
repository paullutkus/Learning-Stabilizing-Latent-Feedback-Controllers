# Learning Stabilizing Latent Feedback Controllers via Multi-Step Backward Losses

Code for the paper *Learning Stabilizing Latent Feedback Controllers via Multi-Step Backward Losses*

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # tested with Python 3.13
pip install -r requirements.txt
```

Everything runs CPU. On Linux, the worker
processes are pinned to cores with `taskset` (util-linux) when it is installed. The figures use
matplotlib's bundled Computer Modern font, so no LaTeX installation is needed.

## Quick start: re-draw the figures and tables

```bash
bash run_all.sh plot
```

This reads the cached results of the paper's runs in `results/paper/` and writes the figures to
`figures/paper/*.pdf` and the tables to `tables/paper/*.tex` (about a minute). The file names are
those of the paper's LaTeX source, and the output is identical to the figures and tables of the paper.

## Figures and tables

| Paper | Script (`experiments/`) | Output | Re-run cost |
|---|---|---|---|
| Fig. 1, motivating example | `motivating_example.py` | `figures/paper/arm_example.pdf` | 2 tasks, 10–20 min each |
| Figs. 2 and 3, non-normal example | `nonnormal.py` | `figures/paper/insufficiency_eta.pdf`, `insufficiency_T.pdf` | 1 multi-threaded process |
| Fig. 4, free-body diagram of the arm | — | a TikZ drawing in the paper's source | — |
| Fig. 5, Tables 1 and 4, linear latent models | `linear_models.py` | `figures/paper/arm_linear_models.pdf`, `tables/paper/linear_stability.tex`, `linear_conjugacy.tex` | 55 tasks, 52 CPU-hours (longest 3.6 h) |
| Fig. 6, Tables 5 and 6, MLP and linear models of the non-linear arm | `nonlinear_models.py` | `figures/paper/arm_nonlinear_models.pdf`, `tables/paper/nonlinear_conjugacy.tex`, `nonlinear_eps_tips.tex` | 30 tasks, ~90 CPU-hours for the 15 MLP tasks (longest 6.7 h) plus the shorter linear-model tasks |
| Fig. 7 and Table 2, several unstable modes | `multi_unstable.py` | `figures/paper/arm_multi_unstable.pdf`, `tables/paper/multi_unstable_mu.tex` | 80 tasks, 1.5 CPU-hours |
| Table 3, experimental protocols | — | written by hand; it summarizes the `CONFIGS` of the scripts | — |

Every experiment writes `results/paper/<experiment>/SUMMARY.md` with
the numbers quoted in the text (e.g. the subspace distances and stabilization counts of Sec. 6.2,
the relative eigenvalue errors of Sec. 6.3, and the gap of Δ between capturing and non-capturing runs
in the caption of Table 2). `motivating_example.py` prints the quantities quoted in Sec. 1.1 (initial
section angle and tip position, time at which the arm under the L_fwd controller reaches the ground).

## Re-running the experiments

```bash
bash run_all.sh run paper 16                                           # every experiment, then every figure and table
python -m experiments.linear_models --stage run --jobs 16 --cpus all   # one experiment
python -m experiments.linear_models --stage plot                       # its figure and tables
python -m experiments.linear_models --worker pop:bwd:orth:1            # one task (names: tasks() of each script)
```

Each script splits its sweep into independent tasks, each run as a single-threaded process, at most
`--jobs` at a time (default 8), pinned to the CPUs `--cpus` (e.g. `0-7`, or `all` -- by default one per
physical core). Logs go to `results/paper/<experiment>/logs/`. Re-running overwrites the cached
results in `results/paper/`. `nonnormal.py` runs in a single (multi-threaded) process.

The full set is about 150 CPU-hours. The paper's runs took about 10 hours of wall-clock time with 16
workers on a 16-thread laptop CPU (AMD Ryzen 7 7840U).

### Fast profile

```bash
bash run_all.sh run fast        # about 2 hours on 8 cores -> figures/fast/, tables/fast/
```

The `fast` profile (`CONFIGS["fast"]` in each script, `--profile fast` for a single script) runs every
experiment with reduced sweeps, for checking that the code runs or trying changes without the full
compute. Its figures and tables show the same qualitative behaviour but are not the paper's numbers.

| Script | Reduced to |
|---|---|
| `motivating_example.py` | not reduced (the paper's two models, 10–20 min each) |
| `nonnormal.py` | 41 instead of 201 values of η, T ≤ 12 instead of 16, coarser grid search |
| `linear_models.py` | 3 initializations instead of 8; L_fwd only at the λ_reg selected in the paper (no λ_reg sweep); finite samples only for L_bwd with R_orth, 2 data sets × 1 initialization; fewer L-BFGS steps |
| `nonlinear_models.py` | horizons T = 5, 10, 20, 40; 2 data sets; 8k + 3k instead of 30k + 10k MLP steps; 4 initializations of the linear models; data scale ε = 0.1 only at T = 10, 20 |
| `multi_unstable.py` | 1 data set, 8 runs per initialization strategy instead of 16 |

Re-drawing from the cached results reproduces the paper's figures and tables, but results may 
change slightly between CPU and the library versions. 
## What the experiments do

All hyperparameters are in `CONFIGS["paper"]` at the top of each script, and each script's docstring
describes its protocol. Appendix I of the paper (Table 3) summarizes them.

* **`motivating_example.py`** (Sec. 1.1): two tanh-MLP latent models of the non-linear arm, trained
  on 2000 trajectories of length T = 40: one with the backward loss L_bwd (prediction-reconstruction,
  prediction and SIGReg terms) and one with the decoder-free forward loss L_fwd (prediction and SIGReg
  terms). Both models are trained with the MLP protocol of `nonlinear_models.py`, with 8 random
  initializations and the multi-step forward term J_fwd^(T). Each model's latent LQR gain then closes
  the loop u = K J_E(0) x on the non-linear arm.
* **`nonnormal.py`** (Sec. 5.2): global minimizers of the population backward loss for the 2×2
  non-normal system, found by a grid over the encoder direction and latent pole followed by L-BFGS,
  and the exact sets of stabilizing gains of the latent model and of the true system: versus the
  non-normality η at T = 1 (Fig. 2), and versus the horizon T at η = 10 (Fig. 3).
* **`linear_models.py`** (Sec. 6.2): linear latent models (m = 4) of the linearized arm. The
  population and finite-sample losses L_bwd^(T) and L_fwd^(T) are minimized with five regularizers
  along a path of horizons T = 1 … 40, from 8 initializations, or from 5 data sets × 2 initializations
  for N = 5 … 1000 trajectories. The scripts compute latent H∞ and LQR gains and test them on the
  linearized and the non-linear arm.
* **`nonlinear_models.py`** (Sec. 6.3): tanh-MLP and linear latent models trained on non-linear-arm
  data (5 data sets of 2000 trajectories) along the horizons T = 1 … 40, at the data scale ε = 0.01
  and, for Tables 5 and 6, at ε = 0.03 and 0.1. The scripts compute equilibrium diagnostics of the
  linearized models and test the controllers u = K E(x) and u = K J_E(0) x on the non-linear arm.
* **`multi_unstable.py`** (Sec. 6.4): the arm with three unstable modes. The empirical L_bwd is trained
  from four initializations of L (16 runs × 5 data sets × T = 10, 20, 30, 40), including the
  growth-rate initialization L = ρ̂ I held for 500 Adam steps, with training traces for Fig. 7.

## Code

| `latentctl/` | contents |
|---|---|
| `systems.py` | the flexible arm (eq. arm_dynamics: one or three buckling modes, zero-order-hold linearization, non-linear step with the configuration-dependent inertia) and the 2×2 non-normal system |
| `data.py` | trajectory data sets of the data model (linearized or non-linear arm) |
| `losses.py` | population (`PopLoss`) and finite-sample (`EmpLoss`) losses L_bwd and L_fwd with the regularizers R_orth, R_vr and R_sr (base measure N(0, I) or the trajectory states) |
| `optim.py` | Adam followed by guarded L-BFGS (vmapped over initializations), the horizon-continuation protocol, initializations |
| `sweeps.py` | population and finite-sample horizon sweeps, scoring of the learned models |
| `linalg.py` | Algorithm 1, canonical variables, the subspace distance Δ, forward-conjugacy residuals |
| `control.py` | latent LQR and H∞ synthesis, closed-loop metrics, non-linear closed-loop simulation, the test initial states |
| `mlp.py` | tanh-MLP encoder, decoder and control-affine latent dynamics (the paper's a_z is `f` in the code), their losses |
| `style.py` | the common figure style |
| `io.py` | the command line, result caching, parallel workers |


