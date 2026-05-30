"""
Vary the assumed formula weights (beta_size, beta_text, lambda) and check
whether optimisation conclusions are robust.

Anchored weights held fixed: contrast (WCAG), time (Infolinks),
whitespace (VWO). Device size-ratio kept identical to the main formula
(mobile = 2x desktop) by overriding beta_size_mobile/desktop together.

Usage:
    python sensitivity_analysis.py
    python sensitivity_analysis.py --n-dataset 5000 --bo-calls 20
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skopt import gp_minimize
from skopt.space import Real

from ab_blackbox import (
    BlackBox,
    TrainedMLModel,
    build_feature_vector,
    build_feature_matrix,
    generate_full_synthetic_dataset,
    p_click,
    DEFAULT_WEIGHTS,
)
from optimize import BOUNDS_CONT, vector_to_params, Objective

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_weights(beta_size: float, beta_text: float, lam: float) -> dict:
    """
    Build a weights override. Preserves the main formula's device asymmetry
    (mobile size weight = 2x desktop) by setting both consistently.
    """
    return {
        **DEFAULT_WEIGHTS,
        "beta_size_desktop": beta_size,
        "beta_size_mobile":  beta_size * 2.0,
        "beta_text":         beta_text,
        "lam":               lam,
    }


def train_quick_model(df: pd.DataFrame):
    X = build_feature_matrix(df)
    y = df["click"].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


def run_bo_small(box: BlackBox, device: str,
                 n_calls: int = 30, seed: int = 42) -> dict:
    obj = Objective(box, device=device)
    space = [Real(lo, hi) for lo, hi in BOUNDS_CONT]
    res = gp_minimize(
        func=obj, dimensions=space,
        n_calls=n_calls, n_initial_points=10,
        acq_func="EI", random_state=seed,
    )
    return {
        "best_ctr":    float(-res.fun),
        "best_params": vector_to_params(np.array(res.x), device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity analysis on weights.")
    parser.add_argument("--n-dataset", type=int, default=10_000)
    parser.add_argument("--bo-calls",  type=int, default=30)
    parser.add_argument("--n-users",   type=int, default=2_000)
    parser.add_argument("--device",    type=str, default="desktop",
                        choices=["mobile", "desktop"])
    parser.add_argument("--out",  type=str, default="sensitivity_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)

    beta_size_grid = [0.3, 0.6, 0.9, 1.2, 1.5]
    beta_text_grid = [0.3, 0.6, 0.9]
    lam_grid       = [1.0, 2.0, 3.0]

    combos = list(product(beta_size_grid, beta_text_grid, lam_grid))
    print(f"Grid points: {len(combos)} "
          f"({len(beta_size_grid)} x {len(beta_text_grid)} x {len(lam_grid)})")
    print(f"Dataset size per point: {args.n_dataset}")
    print(f"BO budget per point:    {args.bo_calls}")
    print(f"Optimising for device:  {args.device}\n")

    rows = []
    for idx, (bs, bt, lam) in enumerate(combos, 1):
        print(f"[{idx}/{len(combos)}] beta_size={bs}, beta_text={bt}, lambda={lam}")

        weights = make_weights(bs, bt, lam)
        df = generate_full_synthetic_dataset(
            n=args.n_dataset, seed=args.seed + idx,
            extra_noise=0.02, weights=weights,
        )
        pipe = train_quick_model(df)

        ml  = TrainedMLModel(classifier=pipe, feature_builder=build_feature_vector)
        box = BlackBox(model=ml, n_users=args.n_users, noise_std=0.01,
                       seed=args.seed + idx)

        out = run_bo_small(box, device=args.device,
                           n_calls=args.bo_calls, seed=args.seed + idx)
        bp  = out["best_params"]
        print(f"  best CTR: {out['best_ctr']:.4f}, "
              f"btn=({bp['btn_w']:.0f}x{bp['btn_h']:.0f}), "
              f"text_quality={bp['text_quality']:.2f}, "
              f"scroll={bp['scroll_to_button']:.2f}")

        rows.append({
            "beta_size":             bs,
            "beta_text":             bt,
            "lambda":                lam,
            "best_ctr":              out["best_ctr"],
            "best_btn_w":            bp["btn_w"],
            "best_btn_h":            bp["btn_h"],
            "best_font_size":        bp["font_size"],
            "best_text_quality":     bp["text_quality"],
            "best_whitespace_ratio": bp["whitespace_ratio"],
            "best_scroll":           bp["scroll_to_button"],
            "best_hour":             bp["hour"],
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.out, index=False)
    print(f"\nSaved sensitivity results to {args.out}")

    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)
    print("Stable optimum across the grid => robust to assumed weights.\n")
    print(f"  Best CTR range:         "
          f"[{df_out['best_ctr'].min():.4f}, {df_out['best_ctr'].max():.4f}]")
    print(f"  Best btn_w range:       "
          f"[{df_out['best_btn_w'].min():.0f}, {df_out['best_btn_w'].max():.0f}]")
    print(f"  Best btn_h range:       "
          f"[{df_out['best_btn_h'].min():.0f}, {df_out['best_btn_h'].max():.0f}]")
    print(f"  Best text_quality mean: "
          f"{df_out['best_text_quality'].mean():.2f} +/- "
          f"{df_out['best_text_quality'].std():.2f}")
    print(f"  Best scroll mean:       "
          f"{df_out['best_scroll'].mean():.2f} +/- "
          f"{df_out['best_scroll'].std():.2f}")
    print(f"  Best whitespace mean:   "
          f"{df_out['best_whitespace_ratio'].mean():.2f} +/- "
          f"{df_out['best_whitespace_ratio'].std():.2f}")


if __name__ == "__main__":
    main()
