"""
demo.py
-------
End-to-end demonstration of the ab_blackbox package.

Covers:
1. Symbolic model inspection
2. Single black-box call
3. Full A/B test run + analysis report
4. Calibration from synthetic dataset
5. Power sweep (empirical power over repeated experiments)
"""

import sys
sys.path.insert(0, "/home/claude")

import numpy as np
from ab_blackbox import (
    get_model,
    BlackBox,
    run_ab_test,
    run_multi_experiment,
    analyze,
    print_report,
    generate_synthetic_dataset,
    calibrate_linear_model,
    print_dataset_summary,
)


# ============================================================
# 1. Inspect the symbolic model
# ============================================================
print("\n" + "="*60)
print("1. SYMBOLIC MODEL")
print("="*60)

model = get_model("linear")
print("\nLinear model expressions:")
model.describe()

params_A = {"color": 0.0, "size": 0.0, "text": 0.0}
params_B = {"color": 1.0, "size": 0.0, "text": 0.0}

print("\nTheoretical effect (population-level, before sampling):")
effects = model.theoretical_effect(params_A, params_B)
for k, v in effects.items():
    print(f"  {k}: {v:.4f}")

n_required = model.required_sample_size(params_A, params_B, alpha=0.05, power=0.80)
print(f"\nRequired sample size per group (80% power, α=0.05): {n_required}")


# ============================================================
# 2. Single black-box call
# ============================================================
print("\n" + "="*60)
print("2. SINGLE BLACK-BOX CALL")
print("="*60)

box = BlackBox(model=model, n_users=2000, noise_std=0.01, seed=42)

result_single = box(params_B)
print(f"\nBlack-box call with params_B = {params_B}")
print(result_single)
print(f"Total calls so far: {box.call_count}")


# ============================================================
# 3. Full A/B test + analysis
# ============================================================
print("\n" + "="*60)
print("3. A/B TEST + ANALYSIS")
print("="*60)

box.reset_call_count()
experiment = run_ab_test(box, params_A, params_B, label="black_button_test")
print(f"\nExperiment result:\n{experiment}")

report = analyze(experiment, alpha=0.05, primary_metric="ctr", run_bootstrap=True)
print_report(report)


# ============================================================
# 4. Calibration from synthetic dataset
# ============================================================
print("\n" + "="*60)
print("4. CALIBRATION FROM DATASET")
print("="*60)

df = generate_synthetic_dataset(n=1000, ctr_A=0.10, ctr_B=0.15, seed=0)
print_dataset_summary(df)

cal = calibrate_linear_model(df, params_A=params_A, params_B=params_B)
print(f"\nCalibrated parameters:")
for k, v in cal["params"].items():
    print(f"  {k}: {v:.4f}")
print(f"\nFit quality:")
for k, v in cal["fit_quality"].items():
    print(f"  {k}: {v:.4f}")

# Run a test with the calibrated model
cal_box    = BlackBox(model=cal["model"], n_users=1000, seed=7)
cal_result = run_ab_test(cal_box, params_A, params_B)
cal_report = analyze(cal_result, primary_metric="ctr")
print(f"\nCalibrated model A/B test decision: {cal_report['decision']['verdict']}")


# ============================================================
# 5. Empirical power sweep
# ============================================================
print("\n" + "="*60)
print("5. EMPIRICAL POWER SWEEP (n per group)")
print("="*60)

n_reps = 200
sample_sizes = [200, 500, 1000, 2000, 5000]

print(f"\n{'n':>6}  {'empirical power':>16}  {'avg delta':>10}")
print("-" * 38)

for n in sample_sizes:
    power_box = BlackBox(model=model, n_users=n, noise_std=0.01, seed=None)
    configs = [(params_A, params_B)] * n_reps
    results = run_multi_experiment(power_box, configs)
    reports = [analyze(r, primary_metric="ctr") for r in results]
    
    sig_count = sum(
        1 for rep in reports
        if rep["primary"]["significant"] and rep["primary"]["delta"] > 0
    )
    avg_delta = np.mean([rep["primary"]["delta"] for rep in reports])
    emp_power = sig_count / n_reps
    
    print(f"{n:>6}  {emp_power:>16.3f}  {avg_delta:>10.4f}")

print(f"\nAnalytical required n={n_required} → empirical power should be ≥0.80 there.")
print("\nDemo complete.")
