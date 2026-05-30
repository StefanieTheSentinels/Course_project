"""
multi-parameter synthetic dataset generator.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable

from .generating_formula import p_click as ground_truth_p_click



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
