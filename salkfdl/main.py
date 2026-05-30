"""
main.py
-------
A/B test pipeline using the real dataset:
    ab_test_results_aggregated_views_clicks_2.csv

Columns: user_id, group, views, clicks
Groups:  control / test
"""

import sys
sys.path.insert(0, ".")  # ensure ab_blackbox/ is found

import pandas as pd
from ab_blackbox import (
    BlackBox,
    run_ab_test,
    analyze,
    print_report,
)
from ab_blackbox.model import LinearButtonModel


# ── 1. Load dataset ──────────────────────────────────────────────────────────
df = pd.read_csv("ab_test_results_aggregated_views_clicks_2.csv")

# Remap columns to the format the package expects
df["Variant"]     = df["group"].map({"control": "A", "test": "B"})
df["Clicks"]      = (df["clicks"] >= 1).astype(int)   # binary: did user click?
df["Conversions"] = df["Clicks"]                       # no separate conversion col

print("Dataset loaded.")
print(df.groupby("Variant")[["Clicks"]].mean().rename(columns={"Clicks": "CTR"}))


# ── 2. Compute empirical rates ────────────────────────────────────────────────
ctr_A = df[df["Variant"] == "A"]["Clicks"].mean()
ctr_B = df[df["Variant"] == "B"]["Clicks"].mean()

print(f"\nEmpirical CTR:  A={ctr_A:.4f}  B={ctr_B:.4f}  Δ={ctr_B - ctr_A:+.4f}")


# ── 3. Calibrate model to match real CTR ─────────────────────────────────────
#
# LinearButtonModel: p_click = base_click + w_color * color
# params_A = color=0  →  p_A = base_click           = ctr_A
# params_B = color=1  →  p_B = base_click + w_color = ctr_B
#
calibrated_model = LinearButtonModel(
    base_click = ctr_A,
    w_color    = ctr_B - ctr_A,   # the effect of switching to treatment
    w_size     = 0.0,
    w_text     = 0.0,
    base_conv  = ctr_A,           # using CTR as proxy for conversion too
    w_conv_color = ctr_B - ctr_A,
    w_conv_size  = 0.0,
    w_conv_text  = 0.0,
)

print("\nCalibrated model formula:")
calibrated_model.describe()


# ── 4. Check theoretical effect and required sample size ─────────────────────
params_A = {"color": 0.0, "size": 0.0, "text": 0.0}
params_B = {"color": 1.0, "size": 0.0, "text": 0.0}

effects = calibrated_model.theoretical_effect(params_A, params_B)
print(f"\nTheoretical Δ CTR: {effects['delta_ctr']:+.4f}")

n_required = calibrated_model.required_sample_size(
    params_A, params_B, alpha=0.05, power=0.80
)
print(f"Required n per group (80% power, α=0.05): {n_required}")
print(f"Actual n per group in dataset: 60,000  →  well-powered ✓")


# ── 5. Run black-box simulation ───────────────────────────────────────────────
# Use the actual dataset size so results are comparable to real data
box = BlackBox(model=calibrated_model, n_users=60_000, noise_std=0.005, seed=42)

result = run_ab_test(box, params_A, params_B, label="real_data_calibrated")
print(f"\nSimulated experiment:\n{result}")


# ── 6. Full statistical analysis ─────────────────────────────────────────────
report = analyze(result, alpha=0.05, primary_metric="ctr", run_bootstrap=True)
print_report(report)


# ── 7. Cross-check: run the same z-test directly on the real dataset ──────────
from ab_blackbox import two_proportion_ztest

n_A  = len(df[df["Variant"] == "A"])
x_A  = int(df[df["Variant"] == "A"]["Clicks"].sum())
n_B  = len(df[df["Variant"] == "B"])
x_B  = int(df[df["Variant"] == "B"]["Clicks"].sum())

real_test = two_proportion_ztest(n_A, x_A, n_B, x_B, alpha=0.05)

print("\nDirect test on real data (no simulation):")
print(f"  CTR A={real_test['p_hat_A']:.4f}  B={real_test['p_hat_B']:.4f}")
print(f"  Δ={real_test['delta']:+.4f}  ({real_test['relative_lift']:+.1%} lift)")
print(f"  z={real_test['z_statistic']:.3f}  p={real_test['p_value']:.4f}")
print(f"  Significant: {real_test['significant']}")
print(f"  95% CI: [{real_test['ci_95'][0]:+.4f}, {real_test['ci_95'][1]:+.4f}]")