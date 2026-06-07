# ab_blackbox

Black-box optimisation of a web button's CTR via a simulated A/B-testing environment.
Compares DIRECT, Differential Evolution, and Bayesian Optimisation under two interchangeable
response models: an analytical oracle and a trained ML surrogate.

Course project, HSE University, 2nd year (БПАД244).  
Full report: [`course_project_report.pdf`](course_project_report.pdf)  
Source code: <https://github.com/StefanieTheSentinels/Course_project>

---

## Problem

A web call-to-action button is parameterised by twelve continuous variables (background
and text colour, size, font size, text quality, padding, vertical position, hour of day,
device). The click-through rate (CTR) as a function of these parameters is treated as an
**expensive noisy black-box**: every evaluation simulates a full A/B test with Bernoulli
outcomes on a finite user sample. The goal is

```
θ* ∈ argmax_{θ ∈ X ⊂ R^12} CTR(θ)
```

under a budget of B = 200 black-box calls, without access to gradients or any internal
state of the simulator.

---

## Pipeline

1. **Ground-truth formula** — `p_click(θ)` built from empirical UX research (WCAG 2.1,
   Baymard 2024, Chartbeat 2014, NN/g 2018, Itten 1961, Ou & Luo 2006, Infolinks, VWO).
2. **Synthetic dataset** — 120 000 random configurations with Bernoulli clicks drawn from
   the formula.
3. **ML surrogate** — logistic regression trained on the dataset; exposes the same interface
   as the oracle and can be hot-swapped into the black box.
4. **Optimisation** — DIRECT, DE, and BO query the black box (oracle or surrogate) within
   the call budget.
5. **Cross-validation** — configurations found via the surrogate are re-evaluated on the
   oracle to measure the extrapolation gap.
6. **A/B analysis** — two-proportion z-test, SRM check, bootstrap CI.

---

## Repository structure

```
ab_blackbox_project/
├── ab_blackbox/                  # core library
│   ├── generating_formula.py     # ground-truth p_click (oracle)
│   ├── model.py                  # ButtonModel, FullSyntheticModel, TrainedMLModel
│   ├── simulator.py              # BlackBox — noisy A/B simulator
│   ├── experiment.py             # run_ab_test
│   ├── analysis.py               # z-test, SRM check, bootstrap CI
│   ├── datasets.py               # synthetic dataset generator
│   └── training.py               # feature engineering + sklearn training
├── graphics/
│   └── surrogate_gap.png         # surrogate-vs-oracle gap figure
├── runs/
│   ├── run_1/ … run_5/           # per-seed outputs (CSV, GIF, logs)
├── tests/
│   └── test_ab_blackbox.py       # pytest suite
├── _run_all.sh                   # end-to-end pipeline (5 seeds)
├── aggregate_runs.py             # aggregates per-seed CSVs → summary tables
├── aggregated_crossval.csv       # cross-validation results (5 seeds)
├── aggregated_ml.csv             # surrogate-run results (5 seeds)
├── aggregated_oracle.csv         # oracle-run results (5 seeds)
├── generate_dataset.py           # CLI: generate synthetic dataset
├── optimize.py                   # CLI: run and compare optimisers
├── plot_gap.py                   # CLI: plot surrogate-vs-oracle gap figure
├── sensitivity_analysis.py       # CLI: sensitivity to formula weights
├── train_models.py               # CLI: train and save surrogate model
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt

# Full pipeline — 5 seeds, all steps, unit tests between each run:
chmod +x _run_all.sh
./_run_all.sh
```

Step by step:

```bash
# 1. Generate dataset
python3 generate_dataset.py --n 120000 --seed 42

# 2. Train surrogate
python3 train_models.py

# 3. Compare optimisers on oracle and surrogate
python3 optimize.py --seed 42 --plot-out runs/run_1/convergence.gif

# 4. Aggregate five-seed results
python3 aggregate_runs.py

# 5. Plot the surrogate-vs-oracle gap
python3 plot_gap.py

# 6. Sensitivity analysis
python3 sensitivity_analysis.py

# 7. Unit tests
pytest tests/ -v
```

---

## Results

### Oracle simulator (honest comparison)

Mean CTR ± std over 5 independent seeds, B = 200 calls, n_users = 10 000.

| Method                  | CTR mean | CTR std | Runtime  |
|-------------------------|----------|---------|----------|
| Bayesian Optimisation   | 0.335    | 0.011   | ~218 s   |
| DIRECT                  | 0.327    | 0.019   | ~0.03 s  |
| Differential Evolution  | 0.226    | 0.039   | ~0.03 s  |

BO and DIRECT are **statistically indistinguishable** on the honest simulator.
DE underperforms at this budget due to insufficient generations (popsize × maxiter < B).

### Surrogate extrapolation gap

When optimisers run on the ML surrogate, configurations are re-evaluated on the oracle
to reveal the true CTR. The gap quantifies surrogate over-optimism.

| Method                 | ML-claimed CTR | Oracle CTR | Gap     |
|------------------------|----------------|------------|---------|
| Bayesian Optimisation  | 0.637          | 0.165      | +0.472  |
| DIRECT                 | 0.278          | 0.191      | +0.087  |
| Differential Evolution | 0.220          | 0.141      | +0.079  |

BO overshoots the analytical oracle ceiling (~0.39) on the surrogate by a large margin
and its true CTR collapses by ~0.47. DIRECT and DE remain within ~0.09 of the truth.
The effect is stable in sign and order of magnitude across all five seeds.

**Root cause**: BO's acquisition function (Expected Improvement) actively follows the
surrogate gradient into out-of-distribution regions where the logistic regression
extrapolates linearly and predictions approach 1. DIRECT and DE operate in coordinate
space and do not exploit surrogate gradients, so they do not preferentially explore
extrapolation regions.

### Convergence

`optimize.py` produces an animated GIF (best-so-far CTR vs. calls) alongside a live
preview of the button found by each method at each step.

---

## Ground-truth formula

```
p(θ) = p_attr(θ) · Penalty(θ) · Vis(θ)
```

- **p_attr** = σ(β₀ + β_text·tq + β_time·(t−1) + β_ws·w(ws))  
  encodes text quality, time-of-day (Infolinks), and whitespace (VWO).
- **Penalty** = f_overflow · f_contrast · f_size · f_harmony · f_margin  
  each factor in (0, 1]: WCAG 2.1 contrast, Baymard/Apple/Material touch targets,
  Itten/Ou–Luo colour harmony, symmetric padding (Lidwell et al.).
- **Vis(scroll)**: linear ramp to peak at scroll = 0.15, then exponential decay
  with λ = 3.0 calibrated to Chartbeat's ~60% engagement drop per fold (NN/g).

The multiplicative structure ensures that a single severely violated constraint
collapses the entire predicted probability to near zero.

---

## API

```python
from ab_blackbox import BlackBox, FullSyntheticModel, run_ab_test, analyze, print_report

# Black-box simulator
box = BlackBox(model=FullSyntheticModel(), n_users=10_000, seed=42)
result = box({
    "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
    "btn_w": 200, "btn_h": 60, "font_size": 18,
    "text_quality": 0.95, "whitespace_ratio": 0.35,
    "scroll_to_button": 0.15, "hour": 13, "device": "desktop",
})
print(result.ctr)  # → ~0.30

# A/B test
exp = run_ab_test(box, params_A, params_B)
report = analyze(exp, alpha=0.05, run_bootstrap=True)
print_report(report)
# SRM Check: [OK] No SRM  (p=0.7821)
# A=0.1240  B=0.1530  delta=+0.0290  (+23.4% lift)
# z=4.812  p=0.0000  [SIG]
# DECISION: SHIP
```

---

## CLI reference

**`optimize.py`**

| Flag | Default | Description |
|------|---------|-------------|
| `--budget` | 200 | Black-box call budget |
| `--n-users` | 10000 | Users per CTR estimate |
| `--noise` | 0.01 | Logit-space Gaussian noise |
| `--device` | desktop | `mobile` or `desktop` |
| `--seed` | 42 | Random seed |
| `--plot-out` | convergence.gif | Convergence GIF path |

**`generate_dataset.py`**

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | 50000 | Dataset size |
| `--seed` | — | Random seed |
| `--noise` | 0.02 | Gaussian noise on p_click |
| `--out` | synthetic_dataset.csv | Output path |

**`train_models.py`**

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | synthetic_dataset.csv | Input CSV |
| `--cv` | 5 | Cross-validation folds |
| `--out` | best_model.pkl | Saved model path |

---

## Dependencies

```
numpy, scipy, pandas, scikit-learn, scikit-optimize, matplotlib, sympy, pytest
```

Pinned versions in `requirements.txt`.

---

## Reproducibility note

`_run_all.sh` draws seeds from `$RANDOM`, so results are not bit-reproducible across
executions. To reproduce exactly, replace `$RANDOM` with a fixed seed list.
The qualitative ordering of the three methods and the sign of the BO extrapolation gap
are stable across all observed runs.
