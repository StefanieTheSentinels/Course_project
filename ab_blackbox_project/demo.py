"""
End-to-end pipeline on a small synthetic dataset (~30-60s).

Sections:
    1. Ground-truth formula on sample configs
    2. Generate small dataset
    3. Train 4 models (3-fold CV)
    4. Logistic coefficient signs
    5. BlackBox with trained model
    6. A/B test
    7. Mini Bayesian Optimisation
    8. Compare against oracle (clean, noise-free comparison)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from skopt import gp_minimize
from skopt.space import Real

from ab_blackbox import (
    p_click,
    FullSyntheticModel,
    generate_full_synthetic_dataset,
    synthetic_dataset_summary,
    train_and_evaluate,
    print_coefficient_report,
    build_feature_vector,
    TrainedMLModel,
    BlackBox,
    run_ab_test,
    analyze,
    print_report,
)
from optimize import BOUNDS_CONT, vector_to_params, Objective


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:

    section("1. GROUND-TRUTH FORMULA")
    configs = {
        "baseline_bad": {
            "rgb_bg": (200, 200, 200), "rgb_text": (180, 180, 180),
            "btn_w": 40, "btn_h": 25, "font_size": 20,
            "text_quality": 0.2, "whitespace_ratio": 0.05,
            "scroll_to_button": 0.7, "hour": 3, "device": "mobile",
        },
        "neutral": {
            "rgb_bg": (255, 255, 255), "rgb_text": (100, 100, 100),
            "btn_w": 100, "btn_h": 50, "font_size": 16,
            "text_quality": 0.5, "whitespace_ratio": 0.2,
            "scroll_to_button": 0.3, "hour": 12, "device": "desktop",
        },
        "near_optimal": {
            "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
            "btn_w": 180, "btn_h": 60, "font_size": 20,
            "text_quality": 0.95, "whitespace_ratio": 0.35,
            "scroll_to_button": 0.05, "hour": 13, "device": "desktop",
        },
    }
    print(f"{'config':<20} {'p_click':>10}")
    print("-" * 32)
    for name, params in configs.items():
        print(f"{name:<20} {p_click(params):>10.4f}")

    section("2. GENERATE SMALL SYNTHETIC DATASET (n=5000)")
    df = generate_full_synthetic_dataset(n=5000, seed=42)
    print(f"Generated: {df.shape}")
    summary = synthetic_dataset_summary(df)
    for k, v in summary.items():
        print(f"  {k:<20}: {v:.4f}" if isinstance(v, float) else f"  {k:<20}: {v}")

    section("3. TRAIN ALL MODELS (3-fold CV)")
    tr = train_and_evaluate(df, cv_folds=3, verbose=True)
    print("\nComparison:")
    print(f"{'Model':<20} {'CV AUC':>10}")
    print("-" * 32)
    for name, info in tr["results"].items():
        marker = " <- BEST" if name == tr["best_name"] else ""
        print(f"{name:<20} {info['cv_auc_mean']:>10.4f}{marker}")

    section("4. LOGISTIC COEFFICIENTS (sign validation)")
    print_coefficient_report(tr["results"]["Logistic"]["fitted"])
    print("\nExpected positive: contrast_ratio, btn_area, text_quality, whitespace_ratio")
    print("Expected negative: font_to_btn_ratio, scroll_to_button, device_mobile")

    section("5. BUILD BLACKBOX WITH TRAINED MODEL")
    ml = TrainedMLModel(classifier=tr["best_model"],
                        feature_builder=build_feature_vector)
    box = BlackBox(model=ml, n_users=2000, noise_std=0.01, seed=7)
    print("Calling with baseline_bad:")
    print(f"  {box(configs['baseline_bad'])}")
    print("Calling with near_optimal:")
    print(f"  {box(configs['near_optimal'])}")

    section("6. A/B TEST: baseline_bad vs near_optimal")
    box.reset_call_count()
    exp = run_ab_test(box, configs["baseline_bad"], configs["near_optimal"],
                      label="demo_test")
    print(exp)
    report = analyze(exp, alpha=0.05, primary_metric="ctr", run_bootstrap=True)
    print_report(report)

    section("7. BAYESIAN OPTIMISATION (20 calls)")
    box.reset_call_count()
    obj = Objective(box, device="desktop")
    space = [Real(lo, hi) for lo, hi in BOUNDS_CONT]
    res_bo = gp_minimize(func=obj, dimensions=space,
                         n_calls=20, n_initial_points=8,
                         acq_func="EI", random_state=42)
    best_x      = np.array(res_bo.x)
    best_ctr_bo = -res_bo.fun
    best_params = vector_to_params(best_x, device="desktop")
    print(f"\nBest CTR found by BO: {best_ctr_bo:.4f} (in 20 calls)")
    print("Best config:")
    for k, v in best_params.items():
        if isinstance(v, tuple):
            print(f"  {k:<20}: {v}")
        elif isinstance(v, float):
            print(f"  {k:<20}: {v:.3f}")
        else:
            print(f"  {k:<20}: {v}")

    section("8. COMPARE AGAINST GROUND-TRUTH ORACLE")
    # Clean comparison: noise-free BlackBox on the trained model vs oracle.
    clean_box = BlackBox(model=ml, n_users=50_000, noise_std=0.0, seed=0)
    model_ctr_clean = clean_box(best_params).ctr
    oracle = FullSyntheticModel()
    p_true_at_bo = oracle.predict_proba(best_params)
    print(f"BO best CTR (noisy, 2000 users):      {best_ctr_bo:.4f}")
    print(f"Model CTR (clean, 50k users):         {model_ctr_clean:.4f}")
    print(f"True p_click at config (oracle):      {p_true_at_bo:.4f}")
    print(f"Model vs oracle gap (approx. error):  "
          f"{abs(model_ctr_clean - p_true_at_bo):+.4f}")
    print("\nThe clean model CTR isolates approximation error from sampling noise.")

    print("\nDemo complete. Full-scale runs:")
    print("  python generate_dataset.py --n 50000")
    print("  python train_models.py")
    print("  python optimize.py")


if __name__ == "__main__":
    main()
