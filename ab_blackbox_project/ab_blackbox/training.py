"""
Train 4 sklearn models on synthetic dataset with 5-fold CV.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from .generating_formula import contrast_ratio, relative_luminance


FEATURE_NAMES: List[str] = [
    "contrast_ratio",
    "btn_area",
    "btn_w",
    "btn_h",
    "font_size",
    "font_to_btn_ratio",
    "text_quality",
    "whitespace_ratio",
    "scroll_to_button",
    "hour_sin",
    "hour_cos",
    "is_peak_hours",
    "is_night",
    "device_mobile",
]


def build_feature_vector(params: Dict) -> np.ndarray:
    cr = contrast_ratio(params["rgb_bg"], params["rgb_text"])
    btn_w = float(params["btn_w"])
    btn_h = float(params["btn_h"])
    font  = float(params["font_size"])
    hour  = int(params["hour"])
    hour_rad = 2 * np.pi * hour / 24.0

    return np.array([
        cr,
        btn_w * btn_h,
        btn_w,
        btn_h,
        font,
        font / btn_h if btn_h > 0 else 0.0,
        float(params["text_quality"]),
        float(params["whitespace_ratio"]),
        float(params["scroll_to_button"]),
        float(np.sin(hour_rad)),
        float(np.cos(hour_rad)),
        1.0 if 11 <= hour <= 14 else 0.0,
        1.0 if (hour >= 23 or hour <= 5) else 0.0,
        1.0 if params["device"] == "mobile" else 0.0,
    ], dtype=float)


def _channel_vec(c: np.ndarray) -> np.ndarray:
    c_norm = c / 255.0
    return np.where(
        c_norm <= 0.03928,
        c_norm / 12.92,
        ((c_norm + 0.055) / 1.055) ** 2.4,
    )


def _luminance_vec(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.2126 * _channel_vec(r) + 0.7152 * _channel_vec(g) + 0.0722 * _channel_vec(b)


def _contrast_ratio_vec(r1, g1, b1, r2, g2, b2) -> np.ndarray:
    l1 = _luminance_vec(r1, g1, b1)
    l2 = _luminance_vec(r2, g2, b2)
    light = np.maximum(l1, l2)
    dark  = np.minimum(l1, l2)
    return (light + 0.05) / (dark + 0.05)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Vectorized feature matrix construction."""
    bg_r = df["bg_r"].values.astype(float)
    bg_g = df["bg_g"].values.astype(float)
    bg_b = df["bg_b"].values.astype(float)
    t_r  = df["text_r"].values.astype(float)
    t_g  = df["text_g"].values.astype(float)
    t_b  = df["text_b"].values.astype(float)

    cr = _contrast_ratio_vec(bg_r, bg_g, bg_b, t_r, t_g, t_b)

    btn_w = df["btn_w"].values.astype(float)
    btn_h = df["btn_h"].values.astype(float)
    font  = df["font_size"].values.astype(float)
    hour  = df["hour"].values.astype(int)

    hour_rad = 2 * np.pi * hour / 24.0
    font_to_btn = np.where(btn_h > 0, font / np.maximum(btn_h, 1e-9), 0.0)

    is_peak  = ((hour >= 11) & (hour <= 14)).astype(float)
    is_night = ((hour >= 23) | (hour <= 5)).astype(float)
    is_mobile = (df["device"].values == "mobile").astype(float)

    return np.column_stack([
        cr,
        btn_w * btn_h,
        btn_w,
        btn_h,
        font,
        font_to_btn,
        df["text_quality"].values.astype(float),
        df["whitespace_ratio"].values.astype(float),
        df["scroll_to_button"].values.astype(float),
        np.sin(hour_rad),
        np.cos(hour_rad),
        is_peak,
        is_night,
        is_mobile,
    ])


def get_models() -> Dict[str, Pipeline]:
    return {
        "Logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(penalty=None, max_iter=1000,
                                          random_state=42)),
        ]),
        "Logistic_L2": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(penalty="l2", C=1.0, max_iter=1000,
                                          random_state=42)),
        ]),
    }


def train_and_evaluate(df: pd.DataFrame,
                       cv_folds: int = 5,
                       verbose: bool = True) -> Dict:
    X = build_feature_matrix(df)
    y = df["click"].values

    if verbose:
        print(f"Feature matrix: {X.shape}, positive share: {y.mean():.4f}")

    models  = get_models()
    results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, pipe in models.items():
        if verbose:
            print(f"\nTraining {name}...")
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring=["roc_auc", "neg_log_loss"],
            n_jobs=-1, return_train_score=False,
        )
        auc_mean     = float(scores["test_roc_auc"].mean())
        auc_std      = float(scores["test_roc_auc"].std())
        logloss_mean = float(-scores["test_neg_log_loss"].mean())

        pipe.fit(X, y)

        results[name] = {
            "cv_auc_mean":     auc_mean,
            "cv_auc_std":      auc_std,
            "cv_logloss_mean": logloss_mean,
            "fitted":          pipe,
        }
        if verbose:
            print(f"  AUC = {auc_mean:.4f} +/- {auc_std:.4f}   "
                  f"LogLoss = {logloss_mean:.4f}")

    best_name  = max(results, key=lambda n: results[n]["cv_auc_mean"])
    best_model = results[best_name]["fitted"]

    if verbose:
        print(f"\nBest model: {best_name} "
              f"(AUC = {results[best_name]['cv_auc_mean']:.4f})")

    return {
        "X": X, "y": y,
        "results": results,
        "best_name":  best_name,
        "best_model": best_model,
    }


def logistic_coefficients(fitted_pipeline: Pipeline) -> Dict[str, float]:
    clf = fitted_pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise TypeError(f"{type(clf).__name__} has no coef_ attribute.")
    return dict(zip(FEATURE_NAMES, clf.coef_[0]))


def print_coefficient_report(fitted_pipeline: Pipeline) -> None:
    coefs = logistic_coefficients(fitted_pipeline)
    print("\nLogistic coefficients (sorted by |coef|):")
    print(f"  {'feature':<20} {'coef':>10}")
    print("  " + "-" * 32)
    for name, c in sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True):
        sign = "+" if c >= 0 else "-"
        print(f"  {name:<20} {sign}{abs(c):>9.4f}")
