"""
Real CSV loaders + multi-parameter synthetic dataset generator.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable
from pathlib import Path

from .generating_formula import p_click as ground_truth_p_click


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    required = {"user_id", "group", "views", "clicks"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    out = pd.DataFrame()
    out["User_ID"] = df["user_id"]
    out["Variant"] = df["group"].map({"control": "A", "test": "B"})
    out["Views"]   = df["views"].astype(float)
    out["Clicks"]  = (df["clicks"] >= 1).astype(int)

    if out["Variant"].isna().any():
        raise ValueError("'group' column contains unexpected values.")

    return out


def dataset_summary(df: pd.DataFrame) -> Dict:
    summary = {}
    for variant in ["A", "B"]:
        g = df[df["Variant"] == variant]
        summary[variant] = {
            "n":                  len(g),
            "ctr_per_user":       float(g["Clicks"].mean()),
            "ctr_per_impression": float(g["Clicks"].sum() / g["Views"].sum()),
            "n_clicks":           int(g["Clicks"].sum()),
            "total_views":        float(g["Views"].sum()),
        }
    return summary


def print_dataset_summary(df: pd.DataFrame) -> None:
    s = dataset_summary(df)
    print("=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    for v in ["A", "B"]:
        g = s[v]
        print(f"\nVariant {v}:")
        print(f"  n={g['n']}")
        print(f"  Per-user CTR: {g['ctr_per_user']:.4f}")
        print(f"  Per-impression CTR: {g['ctr_per_impression']:.4f}")


def calibrate_linear_model(df: pd.DataFrame,
                           params_A: Optional[Dict[str, float]] = None,
                           params_B: Optional[Dict[str, float]] = None) -> Dict:
    from .model import LinearButtonModel

    if params_A is None:
        params_A = {"color": 0.0, "size": 0.0, "text": 0.0}
    if params_B is None:
        params_B = {"color": 1.0, "size": 0.0, "text": 0.0}

    s = dataset_summary(df)
    emp_ctr_A = s["A"]["ctr_per_user"]
    emp_ctr_B = s["B"]["ctr_per_user"]

    base_click        = emp_ctr_A
    delta_color_click = emp_ctr_B - emp_ctr_A

    calibrated = LinearButtonModel(
        base_click=base_click,
        w_color=delta_color_click,
        w_size=0.0, w_text=0.0,
        base_conv=0.0,
        w_conv_color=0.0, w_conv_size=0.0, w_conv_text=0.0,
    )

    pred_A = calibrated._eval(calibrated.click_expr, params_A)
    pred_B = calibrated._eval(calibrated.click_expr, params_B)

    return {
        "model": calibrated,
        "fit_quality": {
            "empirical_ctr_A": emp_ctr_A,
            "empirical_ctr_B": emp_ctr_B,
            "predicted_ctr_A": pred_A,
            "predicted_ctr_B": pred_B,
            "ctr_fit_error_A": abs(pred_A - emp_ctr_A),
            "ctr_fit_error_B": abs(pred_B - emp_ctr_B),
        },
        "params": {
            "base_click":   base_click,
            "w_color":      delta_color_click,
            "base_conv":    0.0,
            "w_conv_color": 0.0,
        },
    }


def generate_synthetic_dataset(n: int = 1000,
                               ctr_A: float = 0.10,
                               ctr_B: float = 0.15,
                               conv_given_click_A: float = 0.30,
                               conv_given_click_B: float = 0.35,
                               mean_views: float = 5.0,
                               seed: int = 42) -> pd.DataFrame:
    """Legacy A/B generator (matches real CSV schema)."""
    rng = np.random.default_rng(seed)
    records = []
    for i, (variant, ctr, conv_rate) in enumerate([
        ("A", ctr_A, conv_given_click_A),
        ("B", ctr_B, conv_given_click_B),
    ]):
        clicks      = rng.binomial(1, ctr, size=n)
        conversions = clicks * rng.binomial(1, conv_rate, size=n)
        views       = np.maximum(rng.poisson(mean_views, size=n), 1)
        for j in range(n):
            records.append({
                "User_ID":     i * n + j + 1,
                "Variant":     variant,
                "Views":       int(views[j]),
                "Clicks":      int(clicks[j]),
                "Conversions": int(conversions[j]),
            })
    return pd.DataFrame(records)


def _sample_params(rng: np.random.Generator) -> Dict:
    device = "mobile" if rng.random() < 0.6 else "desktop"
    return {
        "rgb_bg":           (int(rng.integers(0, 256)),
                             int(rng.integers(0, 256)),
                             int(rng.integers(0, 256))),
        "rgb_text":         (int(rng.integers(0, 256)),
                             int(rng.integers(0, 256)),
                             int(rng.integers(0, 256))),
        "btn_w":            int(rng.integers(20, 301)),
        "btn_h":            int(rng.integers(20, 121)),
        "font_size":        int(rng.integers(8, 49)),
        "text_quality":     float(rng.uniform(0.0, 1.0)),
        "whitespace_ratio": float(rng.uniform(0.0, 0.5)),
        "scroll_to_button": float(rng.uniform(0.0, 1.0)),
        "hour":             int(rng.integers(0, 24)),
        "device":           device,
    }


def _params_to_row(params: Dict, click: int, p: float, user_id: int) -> Dict:
    r_bg,  g_bg,  b_bg  = params["rgb_bg"]
    r_txt, g_txt, b_txt = params["rgb_text"]
    return {
        "user_id":          user_id,
        "bg_r":             r_bg,  "bg_g":   g_bg,  "bg_b":   b_bg,
        "text_r":           r_txt, "text_g": g_txt, "text_b": b_txt,
        "btn_w":            params["btn_w"],
        "btn_h":            params["btn_h"],
        "font_size":        params["font_size"],
        "text_quality":     params["text_quality"],
        "whitespace_ratio": params["whitespace_ratio"],
        "scroll_to_button": params["scroll_to_button"],
        "hour":             params["hour"],
        "device":           params["device"],
        "p_true":           p,
        "click":            click,
    }


def generate_full_synthetic_dataset(n: int = 50_000,
                                    seed: int = 42,
                                    extra_noise: float = 0.02,
                                    p_click_fn: Optional[Callable] = None,
                                    weights: Optional[Dict[str, float]] = None
                                    ) -> pd.DataFrame:
    """
    Generate synthetic dataset by sampling random button configs.

    p_click_fn : custom click probability function (defaults to ground truth).
                 Must accept a params dict and return p in [0,1].
    weights    : if p_click_fn is None, passed to default p_click.
    """
    rng = np.random.default_rng(seed)
    if p_click_fn is None:
        p_fn = lambda params: ground_truth_p_click(params, weights=weights)
    else:
        p_fn = p_click_fn

    rows = []
    for i in range(n):
        params = _sample_params(rng)
        p = p_fn(params)
        if extra_noise > 0:
            p_noisy = float(np.clip(p + rng.normal(0, extra_noise), 0.0, 1.0))
        else:
            p_noisy = p
        click = int(rng.binomial(1, p_noisy))
        rows.append(_params_to_row(params, click, p, i + 1))

    return pd.DataFrame(rows)


def synthetic_dataset_summary(df: pd.DataFrame) -> Dict:
    return {
        "n":            len(df),
        "ctr":          float(df["click"].mean()),
        "n_clicks":     int(df["click"].sum()),
        "mean_p_true":  float(df["p_true"].mean()),
        "mobile_share": float((df["device"] == "mobile").mean()),
        "ctr_mobile":   float(df[df["device"] == "mobile"]["click"].mean()),
        "ctr_desktop":  float(df[df["device"] == "desktop"]["click"].mean()),
    }
