"""
Train 4 models, validate logistic coefficient signs, pickle the best.

Usage:
    python train_models.py
    python train_models.py --data my_dataset.csv --cv 10
"""

import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from ab_blackbox import (
    train_and_evaluate,
    print_coefficient_report,
    logistic_coefficients,
    FEATURE_NAMES,
)


EXPECTED_SIGNS = {
    "contrast_ratio":     "+",
    "btn_area":           "+",
    "btn_w":              "+",
    "btn_h":              "?",
    "font_size":          "?",
    "font_to_btn_ratio":  "-",
    "text_quality":       "+",
    "whitespace_ratio":   "+",
    "scroll_to_button":   "-",
    "hour_sin":           "?",
    "hour_cos":           "?",
    "is_peak_hours":      "?",
    "is_night":           "?",
    "device_mobile":      "-",
}


def validate_signs(coefs: dict) -> dict:
    results = {}
    for feat in FEATURE_NAMES:
        if feat not in EXPECTED_SIGNS:
            continue
        if feat not in coefs:
            continue
        expected = EXPECTED_SIGNS[feat]
        observed = "+" if coefs[feat] >= 0 else "-"
        if expected == "?":
            status = "n/a"
        elif observed == expected:
            status = "OK"
        else:
            status = "MISMATCH"
        results[feat] = {
            "coef":     coefs[feat],
            "expected": expected,
            "observed": observed,
            "status":   status,
        }
    return results


def print_validation_table(validation: dict) -> int:
    print("\nSign validation (expected vs observed):")
    print(f"  {'feature':<22} {'coef':>10}  {'exp':>4}  {'obs':>4}  {'status':>10}")
    print("  " + "-" * 60)
    mismatches = 0
    for feat, info in validation.items():
        sign = "+" if info["coef"] >= 0 else "-"
        if info["status"] == "MISMATCH":
            mismatches += 1
        print(f"  {feat:<22} {sign}{abs(info['coef']):>9.4f}  "
              f"{info['expected']:>4}  {info['observed']:>4}  "
              f"{info['status']:>10}")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and pickle ML models.")
    parser.add_argument("--data",   type=str, default="synthetic_dataset.csv")
    parser.add_argument("--cv",     type=int, default=5)
    parser.add_argument("--out",    type=str, default="best_model.pkl")
    parser.add_argument("--report", type=str, default="training_report.txt")
    args = parser.parse_args()

    print("=" * 60)
    print("MODEL TRAINING + CROSS-VALIDATION")
    print("=" * 60)

    if not os.path.exists(args.data):
        print(f"Dataset not found: {args.data}")
        print("Generate it first with: python generate_dataset.py")
        sys.exit(1)

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")
    print(f"CTR in data: {df['click'].mean():.4f}")

    tr = train_and_evaluate(df, cv_folds=args.cv, verbose=True)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<20} {'AUC mean':>10} {'AUC std':>10} {'LogLoss':>10}")
    print("-" * 52)
    for name, info in tr["results"].items():
        marker = " <-- BEST" if name == tr["best_name"] else ""
        print(f"{name:<20} {info['cv_auc_mean']:>10.4f} "
              f"{info['cv_auc_std']:>10.4f} {info['cv_logloss_mean']:>10.4f}"
              f"{marker}")

    print("\n" + "=" * 60)
    print("LOGISTIC COEFFICIENT VALIDATION")
    print("=" * 60)
    logistic_pipe = tr["results"]["Logistic"]["fitted"]
    print_coefficient_report(logistic_pipe)

    coefs      = logistic_coefficients(logistic_pipe)
    validation = validate_signs(coefs)
    mismatches = print_validation_table(validation)

    if mismatches == 0:
        print("\n[OK] All checked signs match the generating formula.")
    else:
        print(f"\n[WARNING] {mismatches} sign mismatches.")

    with open(args.out, "wb") as f:
        pickle.dump({
            "pipeline":      tr["best_model"],
            "feature_names": FEATURE_NAMES,
            "model_name":    tr["best_name"],
        }, f)
    print(f"\nSaved best model ({tr['best_name']}) to {args.out}")

    with open(args.report, "w") as f:
        f.write("Model training report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.data}, n={len(df)}, CTR={df['click'].mean():.4f}\n")
        f.write(f"CV folds: {args.cv}\n\n")
        f.write(f"{'Model':<20} {'AUC':>10} {'LogLoss':>10}\n")
        f.write("-" * 42 + "\n")
        for name, info in tr["results"].items():
            f.write(f"{name:<20} {info['cv_auc_mean']:>10.4f} "
                    f"{info['cv_logloss_mean']:>10.4f}\n")
        f.write(f"\nBest: {tr['best_name']}\n\n")
        f.write("Logistic coefficients:\n")
        for feat, c in sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True):
            f.write(f"  {feat:<22} {c:+.4f}\n")
        f.write(f"\nSign mismatches: {mismatches}\n")
    print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
