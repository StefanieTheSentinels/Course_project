"""
datasets.py
-----------
Loader and calibration utilities for the Kaggle mock A/B dataset.

Dataset schema (expected CSV columns)
--------------------------------------
User_ID     : int
Variant     : str, "A" or "B"
Clicks      : int
Conversions : int (0 or 1)

Calibration
-----------
We fit the LinearButtonModel parameters so that the simulator's
output distribution matches the empirical CTR and conversion rates
from the dataset. This validates that the black box is realistic.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load the Kaggle mock A/B dataset from a CSV file.

    Parameters
    ----------
    path : path to the CSV file

    Returns
    -------
    pd.DataFrame with columns: User_ID, Variant, Clicks, Conversions
    """
    df = pd.read_csv(path)

    required = {"User_ID", "Variant", "Clicks", "Conversions"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Normalize variant labels
    df["Variant"] = df["Variant"].str.strip().str.upper()
    if not set(df["Variant"].unique()).issubset({"A", "B"}):
        raise ValueError("Variant column must contain only 'A' and 'B'.")

    return df


def dataset_summary(df: pd.DataFrame) -> Dict:
    """
    Compute empirical CTR and conversion rates per variant.

    Returns
    -------
    dict with n, ctr, conversion_rate for A and B
    """
    summary = {}
    for variant in ["A", "B"]:
        g = df[df["Variant"] == variant]
        n = len(g)
        # Clicks column: treat as binary (>=1 = clicked)
        clicked = (g["Clicks"] >= 1).astype(int)
        ctr = float(clicked.mean())
        conv_rate = float(g["Conversions"].mean())
        summary[variant] = {
            "n":               n,
            "ctr":             ctr,
            "conversion_rate": conv_rate,
            "n_clicks":        int(clicked.sum()),
            "n_conversions":   int(g["Conversions"].sum()),
        }
    return summary


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print formatted dataset summary."""
    s = dataset_summary(df)
    print(f"{'='*50}")
    print("Dataset Summary")
    print(f"{'='*50}")
    for v in ["A", "B"]:
        g = s[v]
        print(f"\nVariant {v}:")
        print(f"  n={g['n']}, CTR={g['ctr']:.4f}, Conversion={g['conversion_rate']:.4f}")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate_linear_model(
    df: pd.DataFrame,
    params_A: Optional[Dict[str, float]] = None,
    params_B: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Calibrate a LinearButtonModel to match empirical CTR from the dataset.

    Strategy
    --------
    The empirical CTR for variant A gives us the base_click rate.
    The empirical CTR for variant B, minus A, gives us an effect size.
    We then back out w_color assuming color=1 in variant B, color=0 in A.

    This is a simple closed-form calibration. More complex fitting
    (e.g., MLE over all parameters) is possible but not needed here.

    Parameters
    ----------
    params_A : parameter dict for variant A (default: all zeros)
    params_B : parameter dict for variant B (default: color=1)

    Returns
    -------
    dict with calibrated model parameters and fit quality
    """
    from .model import LinearButtonModel

    if params_A is None:
        params_A = {"color": 0.0, "size": 0.0, "text": 0.0}
    if params_B is None:
        params_B = {"color": 1.0, "size": 0.0, "text": 0.0}

    s = dataset_summary(df)

    emp_ctr_A    = s["A"]["ctr"]
    emp_ctr_B    = s["B"]["ctr"]
    emp_conv_A   = s["A"]["conversion_rate"]
    emp_conv_B   = s["B"]["conversion_rate"]

    # base_click = p_A (since all params are 0)
    base_click = emp_ctr_A

    # delta_ctr = w_color * (color_B - color_A) = w_color * 1
    delta_color_click = emp_ctr_B - emp_ctr_A

    # Similarly for conversion
    base_conv        = emp_conv_A
    delta_color_conv = emp_conv_B - emp_conv_A

    calibrated_model = LinearButtonModel(
        base_click=base_click,
        w_color=delta_color_click,
        w_size=0.0,   # dataset has no size signal
        w_text=0.0,
        base_conv=base_conv,
        w_conv_color=delta_color_conv,
        w_conv_size=0.0,
        w_conv_text=0.0,
    )

    # Check: predicted vs empirical
    pred_A = calibrated_model._eval(calibrated_model.click_expr, params_A)
    pred_B = calibrated_model._eval(calibrated_model.click_expr, params_B)

    fit_quality = {
        "empirical_ctr_A":    emp_ctr_A,
        "empirical_ctr_B":    emp_ctr_B,
        "predicted_ctr_A":    pred_A,
        "predicted_ctr_B":    pred_B,
        "ctr_fit_error_A":    abs(pred_A - emp_ctr_A),
        "ctr_fit_error_B":    abs(pred_B - emp_ctr_B),
    }

    return {
        "model":       calibrated_model,
        "fit_quality": fit_quality,
        "params": {
            "base_click":      base_click,
            "w_color":         delta_color_click,
            "base_conv":       base_conv,
            "w_conv_color":    delta_color_conv,
        }
    }


# ---------------------------------------------------------------------------
# Synthetic dataset generator (for testing without the Kaggle file)
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(
    n: int = 1000,
    ctr_A: float = 0.10,
    ctr_B: float = 0.15,
    conv_given_click_A: float = 0.30,
    conv_given_click_B: float = 0.35,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset matching the Kaggle schema.

    Useful for testing the package without downloading external data.

    Parameters
    ----------
    n    : users per variant
    ctr_A, ctr_B : click-through rates
    conv_given_click_A/B : conversion rate conditional on click

    Returns
    -------
    pd.DataFrame with columns: User_ID, Variant, Clicks, Conversions
    """
    rng = np.random.default_rng(seed)

    records = []
    for i, (variant, ctr, conv_rate) in enumerate([
        ("A", ctr_A, conv_given_click_A),
        ("B", ctr_B, conv_given_click_B),
    ]):
        clicks      = rng.binomial(1, ctr, size=n)
        conversions = clicks * rng.binomial(1, conv_rate, size=n)
        for j in range(n):
            records.append({
                "User_ID":     i * n + j + 1,
                "Variant":     variant,
                "Clicks":      int(clicks[j]),
                "Conversions": int(conversions[j]),
            })

    return pd.DataFrame(records)
