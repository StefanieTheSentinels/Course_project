"""
optimize.py
-----------
Sweep over button parameter combinations and find which maximizes CTR.

This is the "optimize the black box" part of the course project.
The black box is queried for each parameter combination — each call
is one simulated experiment. We track total call count to show the
cost of optimization.

Three approaches shown:
1. Grid search      — exhaustive, finds global optimum, expensive
2. Random search    — cheaper, finds near-optimal with fewer calls
3. Greedy 1-D sweep — vary one parameter at a time, very cheap, may miss interactions
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from itertools import product
from ab_blackbox import BlackBox
from ab_blackbox.model import LinearButtonModel

# ── Calibrated model from real data ──────────────────────────────────────────
# ctr_A=0.1470, ctr_B=0.1617 → w_color = 0.0147
# We also give size and text small positive weights to make the
# optimization non-trivial (otherwise color=1 trivially wins).
model = LinearButtonModel(
    base_click   = 0.1470,
    w_color      = 0.0147,   # from real data
    w_size       = 0.0080,   # assumed small effect
    w_text       = 0.0120,   # assumed moderate effect
    base_conv    = 0.1470,
    w_conv_color = 0.0147,
    w_conv_size  = 0.0080,
    w_conv_text  = 0.0120,
)

box = BlackBox(model=model, n_users=2000, noise_std=0.005, seed=42)

# Parameter grid: each parameter takes values in [0, 0.25, 0.5, 0.75, 1.0]
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
PARAM_NAMES = ["color", "size", "text"]


# ── 1. Grid search ────────────────────────────────────────────────────────────
print("=" * 60)
print("1. GRID SEARCH")
print("=" * 60)

box.reset_call_count()
results_grid = []

for color_val, size_val, text_val in product(GRID, GRID, GRID):
    params = {"color": color_val, "size": size_val, "text": text_val}
    sim = box(params)
    results_grid.append({
        "color": color_val,
        "size":  size_val,
        "text":  text_val,
        "ctr":   sim.ctr,
    })

df_grid = pd.DataFrame(results_grid).sort_values("ctr", ascending=False)
best_grid = df_grid.iloc[0]

print(f"Combinations evaluated: {box.call_count}")
print(f"Best config: color={best_grid['color']}  size={best_grid['size']}  text={best_grid['text']}")
print(f"Best CTR:    {best_grid['ctr']:.4f}")
print(f"\nTop 5:")
print(df_grid.head().to_string(index=False))


# ── 2. Random search ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. RANDOM SEARCH  (budget = 30 calls)")
print("=" * 60)

box.reset_call_count()
rng = np.random.default_rng(0)
results_random = []
BUDGET = 30

for _ in range(BUDGET):
    params = {p: float(rng.choice(GRID)) for p in PARAM_NAMES}
    sim = box(params)
    results_random.append({**params, "ctr": sim.ctr})

df_random = pd.DataFrame(results_random).sort_values("ctr", ascending=False)
best_random = df_random.iloc[0]

print(f"Combinations evaluated: {box.call_count}")
print(f"Best config: color={best_random['color']}  size={best_random['size']}  text={best_random['text']}")
print(f"Best CTR:    {best_random['ctr']:.4f}")


# ── 3. Greedy 1-D sweep ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. GREEDY 1-D SWEEP")
print("=" * 60)

box.reset_call_count()
current = {"color": 0.0, "size": 0.0, "text": 0.0}

for param in PARAM_NAMES:
    best_val  = current[param]
    best_ctr  = box(current).ctr
    for val in GRID[1:]:  # skip 0.0, already evaluated
        candidate = {**current, param: val}
        ctr = box(candidate).ctr
        if ctr > best_ctr:
            best_ctr = ctr
            best_val = val
    current[param] = best_val
    print(f"  After optimizing {param}: best={best_val}  CTR={best_ctr:.4f}")

print(f"\nCombinations evaluated: {box.call_count}")
print(f"Best config: {current}")
print(f"Best CTR:    {best_ctr:.4f}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
baseline_ctr = box({"color": 0.0, "size": 0.0, "text": 0.0}).ctr

print(f"{'Method':<20} {'Calls':>6}  {'Best CTR':>9}  {'Lift vs baseline':>17}")
print("-" * 58)
print(f"{'Baseline (A)':<20} {'1':>6}  {baseline_ctr:>9.4f}  {'—':>17}")
print(f"{'Grid search':<20} {125:>6}  {best_grid['ctr']:>9.4f}  "
      f"{(best_grid['ctr']-baseline_ctr)/baseline_ctr:>+16.1%}")
print(f"{'Random search':<20} {BUDGET:>6}  {best_random['ctr']:>9.4f}  "
      f"{(best_random['ctr']-baseline_ctr)/baseline_ctr:>+16.1%}")
print(f"{'Greedy sweep':<20} {box.call_count:>6}  {best_ctr:>9.4f}  "
      f"{(best_ctr-baseline_ctr)/baseline_ctr:>+16.1%}")

# ── Theoretical optimum (SymPy) ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("THEORETICAL OPTIMUM (SymPy, no sampling noise)")
print("=" * 60)
opt_params = {"color": 1.0, "size": 1.0, "text": 1.0}
theo = model.theoretical_effect({"color":0,"size":0,"text":0}, opt_params)
print(f"All params=1.0 → theoretical CTR = {theo['p_click_B']:.4f}")
print(f"Lift vs baseline: {(theo['p_click_B'] - theo['p_click_A'])/theo['p_click_A']:+.1%}")
print("\nSymPy formula shows why: p = 0.147 + 0.0147·color + 0.008·size + 0.012·text")
print("All three at max gives the highest possible p.")