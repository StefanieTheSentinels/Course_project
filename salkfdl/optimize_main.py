"""
optimize.py
-----------
Black-box optimization calibrated from real data.

The real dataset tells us:
  - baseline CTR (control group) = 0.1470
  - treatment CTR (test group)   = 0.1617
  - measured effect of one feature change = +0.0147

We know the experiment compared ONE change (control vs test).
That change is modeled as color=0 → color=1.

size and text are UNKNOWN from the data — we have no observations
for them. So we treat them as having the same unit effect as color
(conservative assumption: each parameter contributes equally).

The black box is then queried for combinations of all three parameters
to find the configuration that maximizes CTR.

This is honest: the simulator is grounded in real measurements,
and we are transparent about which effects are observed vs assumed.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from itertools import product
from ab_blackbox import BlackBox
from ab_blackbox.model import LinearButtonModel


# ── Step 1: load real data and extract observed effect ───────────────────────
print("=" * 60)
print("CALIBRATION FROM REAL DATA")
print("=" * 60)

df = pd.read_csv("ab_test_results_aggregated_views_clicks_2.csv")

ctr_control = (df[df["group"] == "control"]["clicks"] >= 1).mean()
ctr_test    = (df[df["group"] == "test"]["clicks"] >= 1).mean()
observed_effect = ctr_test - ctr_control

print(f"  Control CTR:       {ctr_control:.4f}  (baseline, color=0)")
print(f"  Test CTR:          {ctr_test:.4f}  (treatment, color=1)")
print(f"  Observed effect:   {observed_effect:+.4f}  (this is w_color)")
print(f"\n  Assumption: w_size and w_text are unknown from data.")
print(f"  We set them equal to w_color ({observed_effect:.4f}) — conservative.")
print(f"  This means each parameter contributes equally when fully activated.")


# ── Step 2: build calibrated model ───────────────────────────────────────────
model = LinearButtonModel(
    base_click   = ctr_control,       # anchored to real baseline
    w_color      = observed_effect,   # anchored to real measured effect
    w_size       = observed_effect,   # assumed: same unit effect as color
    w_text       = observed_effect,   # assumed: same unit effect as color
    base_conv    = ctr_control,
    w_conv_color = observed_effect,
    w_conv_size  = observed_effect,
    w_conv_text  = observed_effect,
)

print(f"\nCalibrated model formula:")
model.describe()

# Verify: params_A and params_B reproduce real CTRs exactly
params_A = {"color": 0.0, "size": 0.0, "text": 0.0}
params_B = {"color": 1.0, "size": 0.0, "text": 0.0}  # only color changes = real experiment
effects = model.theoretical_effect(params_A, params_B)
print(f"\nVerification (color only, size=text=0):")
print(f"  Predicted CTR A: {effects['p_click_A']:.4f}  (real: {ctr_control:.4f})")
print(f"  Predicted CTR B: {effects['p_click_B']:.4f}  (real: {ctr_test:.4f})")


# ── Step 3: build black box ───────────────────────────────────────────────────
box = BlackBox(model=model, n_users=5000, noise_std=0.003, seed=42)
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
PARAM_NAMES = ["color", "size", "text"]


# ── Step 4: grid search ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GRID SEARCH OVER ALL PARAMETER COMBINATIONS")
print("=" * 60)

box.reset_call_count()
records = []

for color_val, size_val, text_val in product(GRID, GRID, GRID):
    params = {"color": color_val, "size": size_val, "text": text_val}
    sim = box(params)
    records.append({
        "color": color_val,
        "size":  size_val,
        "text":  text_val,
        "ctr":   sim.ctr,
    })

df_results = pd.DataFrame(records).sort_values("ctr", ascending=False)
best = df_results.iloc[0]
baseline_ctr = df_results[
    (df_results["color"]==0) & (df_results["size"]==0) & (df_results["text"]==0)
]["ctr"].values[0]

real_experiment_ctr = df_results[
    (df_results["color"]==1) & (df_results["size"]==0) & (df_results["text"]==0)
]["ctr"].values[0]

print(f"\nTotal black-box calls: {box.call_count}")
print(f"\nBaseline  (control,  color=0 size=0 text=0): CTR={baseline_ctr:.4f}")
print(f"Real test (color=1,  size=0 text=0):          CTR={real_experiment_ctr:.4f}  "
      f"(lift={real_experiment_ctr-baseline_ctr:+.4f})")
print(f"Optimal   (color={best['color']} size={best['size']} text={best['text']}): "
      f"CTR={best['ctr']:.4f}  (lift={best['ctr']-baseline_ctr:+.4f})")

print(f"\nTop 10 configurations:")
print(df_results.head(10).to_string(index=False))


# ── Step 5: random search (cheaper) ──────────────────────────────────────────
print("\n" + "=" * 60)
print("RANDOM SEARCH  (budget = 30 calls)")
print("=" * 60)

box.reset_call_count()
rng = np.random.default_rng(1)
random_records = []

for _ in range(30):
    params = {p: float(rng.choice(GRID)) for p in PARAM_NAMES}
    sim = box(params)
    random_records.append({**params, "ctr": sim.ctr})

df_random = pd.DataFrame(random_records).sort_values("ctr", ascending=False)
best_random = df_random.iloc[0]
print(f"Calls used: {box.call_count}")
print(f"Best found: color={best_random['color']}  size={best_random['size']}  "
      f"text={best_random['text']}  CTR={best_random['ctr']:.4f}")


# ── Step 6: greedy 1-D sweep ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GREEDY 1-D SWEEP")
print("=" * 60)

box.reset_call_count()
current = {"color": 0.0, "size": 0.0, "text": 0.0}

for param in PARAM_NAMES:
    best_val = current[param]
    best_ctr_val = box(current).ctr
    for val in GRID[1:]:
        candidate = {**current, param: val}
        ctr_val = box(candidate).ctr
        if ctr_val > best_ctr_val:
            best_ctr_val = ctr_val
            best_val = val
    current[param] = best_val
    print(f"  Optimized {param}: best={best_val}  CTR={best_ctr_val:.4f}")

print(f"Calls used: {box.call_count}")
print(f"Best config: {current}  CTR={best_ctr_val:.4f}")


# ── Step 7: theoretical optimum via SymPy (zero calls) ───────────────────────
print("\n" + "=" * 60)
print("THEORETICAL OPTIMUM (SymPy — no black-box calls)")
print("=" * 60)

theo = model.theoretical_effect(
    {"color": 0.0, "size": 0.0, "text": 0.0},
    {"color": 1.0, "size": 1.0, "text": 1.0},
)
print(f"All params = 1.0  →  CTR = {theo['p_click_B']:.4f}")
print(f"Formula: p = {ctr_control:.4f} + {observed_effect:.4f}·color "
      f"+ {observed_effect:.4f}·size + {observed_effect:.4f}·text")
print(f"At max:  p = {ctr_control:.4f} + 3×{observed_effect:.4f} = {theo['p_click_B']:.4f}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Method':<25} {'Calls':>6}  {'Best CTR':>9}  {'Lift vs real test':>18}")
print("-" * 64)
print(f"{'Real experiment (A→B)':<25} {'—':>6}  {ctr_test:>9.4f}  {'(reference)':>18}")
print(f"{'Grid search':<25} {125:>6}  {best['ctr']:>9.4f}  "
      f"{(best['ctr']-ctr_test)/ctr_test:>+17.1%}")
print(f"{'Random search (30)':<25} {30:>6}  {best_random['ctr']:>9.4f}  "
      f"{(best_random['ctr']-ctr_test)/ctr_test:>+17.1%}")
print(f"{'Greedy sweep':<25} {box.call_count:>6}  {best_ctr_val:>9.4f}  "
      f"{(best_ctr_val-ctr_test)/ctr_test:>+17.1%}")
print(f"{'SymPy (theoretical)':<25} {'0':>6}  {theo['p_click_B']:>9.4f}  "
      f"{(theo['p_click_B']-ctr_test)/ctr_test:>+17.1%}")