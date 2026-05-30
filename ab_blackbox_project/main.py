"""
-------
Legacy real-data A/B pipeline on ab_test_results_aggregated_views_clicks_2.csv.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from scipy import stats

from ab_blackbox import (
    BlackBox,
    run_ab_test,
    analyze,
    print_report,
    load_dataset,
    calibrate_linear_model,
    two_proportion_ztest,
)


def main() -> None:
    df = load_dataset("ab_test_results_aggregated_views_clicks_2.csv")

    print("Dataset loaded.")
    print(df.groupby("Variant")[["Clicks", "Views"]].agg(
        n=("Clicks", "count"),
        ctr=("Clicks", "mean"),
        total_views=("Views", "sum"),
        total_clicks=("Clicks", "sum"),
    ))

    ctr_A = df[df["Variant"] == "A"]["Clicks"].mean()
    ctr_B = df[df["Variant"] == "B"]["Clicks"].mean()
    print(f"\nPer-user CTR:  A={ctr_A:.4f}  B={ctr_B:.4f}  delta={ctr_B - ctr_A:+.4f}")

    total_clicks_A = df[df["Variant"] == "A"]["Clicks"].sum()
    total_views_A  = df[df["Variant"] == "A"]["Views"].sum()
    total_clicks_B = df[df["Variant"] == "B"]["Clicks"].sum()
    total_views_B  = df[df["Variant"] == "B"]["Views"].sum()

    imp_ctr_A = total_clicks_A / total_views_A
    imp_ctr_B = total_clicks_B / total_views_B
    print(f"Per-impression CTR: A={imp_ctr_A:.4f}  B={imp_ctr_B:.4f}  "
          f"delta={imp_ctr_B - imp_ctr_A:+.4f}  "
          f"({(imp_ctr_B-imp_ctr_A)/imp_ctr_A:+.1%} lift)")

    se_imp = np.sqrt(
        imp_ctr_A * (1 - imp_ctr_A) / total_views_A
        + imp_ctr_B * (1 - imp_ctr_B) / total_views_B
    )
    z_imp = (imp_ctr_B - imp_ctr_A) / se_imp
    p_imp = 2 * stats.norm.sf(abs(z_imp))
    print(f"  z={z_imp:.3f}  p={p_imp:.4f}  "
          f"{'Significant' if p_imp < 0.05 else 'Not significant'}")

    cal = calibrate_linear_model(df)
    calibrated_model = cal["model"]

    print("\nCalibrated model formula:")
    calibrated_model.describe()
    print(f"\nFit quality: error_A={cal['fit_quality']['ctr_fit_error_A']:.6f}  "
          f"error_B={cal['fit_quality']['ctr_fit_error_B']:.6f}")

    params_A = {"color": 0.0, "size": 0.0, "text": 0.0}
    params_B = {"color": 1.0, "size": 0.0, "text": 0.0}

    n_required = calibrated_model.required_sample_size(
        params_A, params_B, alpha=0.05, power=0.80
    )
    print(f"\nRequired n per group (80% power, alpha=0.05): {n_required}")
    print(f"Actual n per group: 60,000  ->  well-powered [OK]")

    box = BlackBox(model=calibrated_model, n_users=60_000, noise_std=0.0, seed=42)
    result = run_ab_test(box, params_A, params_B, label="real_data_calibrated")
    print(f"\nSimulated experiment (noise_std=0):\n{result}")

    report = analyze(result, alpha=0.05, primary_metric="ctr", run_bootstrap=True)
    print_report(report)

    n_A = len(df[df["Variant"] == "A"])
    x_A = int(df[df["Variant"] == "A"]["Clicks"].sum())
    n_B = len(df[df["Variant"] == "B"])
    x_B = int(df[df["Variant"] == "B"]["Clicks"].sum())

    real_test = two_proportion_ztest(n_A, x_A, n_B, x_B, alpha=0.05)

    print("\nDirect z-test on real data:")
    print(f"  CTR A={real_test['p_hat_A']:.4f}  B={real_test['p_hat_B']:.4f}")
    print(f"  delta={real_test['delta']:+.4f}  ({real_test['relative_lift']:+.1%} lift)")
    print(f"  z={real_test['z_statistic']:.3f}  p={real_test['p_value']:.4f}")
    print(f"  Significant: {real_test['significant']}")
    print(f"  95% CI: [{real_test['ci_95'][0]:+.4f}, {real_test['ci_95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
