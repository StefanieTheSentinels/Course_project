"""
datasets.py
-----------
Loads ab_test_results_aggregated_views_clicks_2.csv and calibrates
LinearButtonModel to match empirical per-user CTR.

Schema: user_id, group (control/test), views, clicks.
No conversion column — analysis uses CTR only.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path


# Loader — matches the real CSV schema
def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load the A/B test dataset from a CSV file.
    Accepts the real schema (user_id, group, views, clicks) and returns
    a normalised DataFrame with columns:
        User_ID, Variant, Views, Clicks

    The 'Clicks' column is binarised (>=1 = clicked).
    No 'Conversions' column is created since the dataset has none.
    """
    df = pd.read_csv(path)

    # Normalise column names (case-insensitive)
    df.columns = [c.lower() for c in df.columns]

    required = {"user_id", "group", "views", "clicks"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    out = pd.DataFrame()
    out["User_ID"] = df["user_id"]
    out["Variant"] = df["group"].map({"control": "A", "test": "B"})
    out["Views"]   = df["views"].astype(float)
    out["Clicks"]  = (df["clicks"] >= 1).astype(int)   # binary per-user

    if out["Variant"].isna().any():
        raise ValueError("'group' column contains unexpected values.")

    return out


def dataset_summary(df: pd.DataFrame) -> Dict:
    
    """
    Compute empirical per-user CTR and per-impression CTR per variant.

    df : DataFrame as returned by load_dataset
         Must have columns: Variant, Clicks, Views

    Returns dict with per-user CTR and per-impression CTR for A and B
    """
    summary = {}
    for variant in ["A", "B"]:
        g = df[df["Variant"] == variant]
        n           = len(g)
        ctr_user    = float(g["Clicks"].mean())
        ctr_impress = float(g["Clicks"].sum() / g["Views"].sum())
        summary[variant] = {
            "n":               n,
            "ctr_per_user":    ctr_user,
            "ctr_per_impression": ctr_impress,
            "n_clicks":        int(g["Clicks"].sum()),
            "total_views":     float(g["Views"].sum()),
        }
    return summary


def print_dataset_summary(df: pd.DataFrame) -> None:
    # Print formatted dataset summary.
    s = dataset_summary(df)
    print("=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    for v in ["A", "B"]:
        g = s[v]
        print(f"\nVariant {v}:")
        print(f"  n={g['n']}")
        print(f"  Per-user CTR (clicked>=1): {g['ctr_per_user']:.4f}")
        print(f"  Per-impression CTR (clicks/views): {g['ctr_per_impression']:.4f}")
        print(f"  Total views: {g['total_views']:.0f}")


# Calibration
def calibrate_linear_model(
    df: pd.DataFrame,
    params_A: Optional[Dict[str, float]] = None,
    params_B: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Calibrate a LinearButtonModel to match empirical per-user CTR.
    Uses the per-user binary CTR (Clicks column, binarised).
    """
    from .model import LinearButtonModel

    if params_A is None:
        params_A = {"color": 0.0, "size": 0.0, "text": 0.0}
    if params_B is None:
        params_B = {"color": 1.0, "size": 0.0, "text": 0.0}

    s = dataset_summary(df)

    emp_ctr_A = s["A"]["ctr_per_user"]
    emp_ctr_B = s["B"]["ctr_per_user"]

    base_click       = emp_ctr_A
    delta_color_click = emp_ctr_B - emp_ctr_A

    # Conversion: use CONDITIONAL rate if Conversions column present
    has_conversions = "Conversions" in df.columns
    if has_conversions:
        clickers_A    = df[(df["Variant"] == "A") & (df["Clicks"] >= 1)]
        clickers_B    = df[(df["Variant"] == "B") & (df["Clicks"] >= 1)]
        base_conv     = float(clickers_A["Conversions"].mean()) if len(clickers_A) > 0 else 0.0
        delta_color_conv = (
            float(clickers_B["Conversions"].mean()) - base_conv
            if len(clickers_B) > 0 else 0.0
        )
    else:
        base_conv        = 0.0
        delta_color_conv = 0.0

    calibrated_model = LinearButtonModel(
        base_click=base_click,
        w_color=delta_color_click,
        w_size=0.0,
        w_text=0.0,
        base_conv=base_conv,
        w_conv_color=delta_color_conv,
        w_conv_size=0.0,
        w_conv_text=0.0,
    )

    pred_A = calibrated_model._eval(calibrated_model.click_expr, params_A)
    pred_B = calibrated_model._eval(calibrated_model.click_expr, params_B)

    fit_quality = {
        "empirical_ctr_A": emp_ctr_A,
        "empirical_ctr_B": emp_ctr_B,
        "predicted_ctr_A": pred_A,
        "predicted_ctr_B": pred_B,
        "ctr_fit_error_A": abs(pred_A - emp_ctr_A),
        "ctr_fit_error_B": abs(pred_B - emp_ctr_B),
    }

    return {
        "model":       calibrated_model,
        "fit_quality": fit_quality,
        "params": {
            "base_click":   base_click,
            "w_color":      delta_color_click,
            "base_conv":    base_conv,
            "w_conv_color": delta_color_conv,
        },
    }


# Synthetic dataset generator
def generate_synthetic_dataset(
    n: int = 1000,
    ctr_A: float = 0.10,
    ctr_B: float = 0.15,
    conv_given_click_A: float = 0.30,
    conv_given_click_B: float = 0.35,
    mean_views: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    
    # Generate a synthetic dataset matching the real CSV schema.
    rng = np.random.default_rng(seed)
    records = []

    for i, (variant, ctr, conv_rate) in enumerate([
        ("A", ctr_A, conv_given_click_A),
        ("B", ctr_B, conv_given_click_B),
    ]):
        clicks      = rng.binomial(1, ctr, size=n)
        # Conversion is CONDITIONAL on click
        conversions = clicks * rng.binomial(1, conv_rate, size=n)
        views       = rng.poisson(mean_views, size=n)
        views       = np.maximum(views, 1)   # at least 1 view

        for j in range(n):
            records.append({
                "User_ID":     i * n + j + 1,
                "Variant":     variant,
                "Views":       int(views[j]),
                "Clicks":      int(clicks[j]),
                "Conversions": int(conversions[j]),
            })

    return pd.DataFrame(records)
