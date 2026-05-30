"""
Response models for BlackBox.

Classes:
    ButtonModel         : abstract base (symbolic)
    LinearButtonModel   : linear (used with real CSV)
    LogisticButtonModel : sigmoid linear
    FullSyntheticModel  : wraps ground-truth formula (oracle)
    TrainedMLModel      : wraps a trained sklearn classifier
"""

from __future__ import annotations
import sympy as sp
import numpy as np
from typing import Dict, Any, Optional

from .generating_formula import p_click as ground_truth_p_click


color = sp.Symbol("color", real=True)
size  = sp.Symbol("size",  real=True)
text  = sp.Symbol("text",  real=True)


class ButtonModel:
    """Abstract symbolic model (legacy)."""

    def __init__(self, click_expr: sp.Expr, conversion_expr: sp.Expr):
        self.click_expr      = click_expr
        self.conversion_expr = conversion_expr

    def describe(self) -> str:
        lines = [
            "Click probability:",
            str(sp.pretty(self.click_expr)),
            "",
            "Conversion probability (given click):",
            str(sp.pretty(self.conversion_expr)),
        ]
        result = "\n".join(lines)
        print(result)
        return result

    def theoretical_effect(self, params_A, params_B) -> Dict[str, float]:
        p_click_A = self._eval(self.click_expr, params_A)
        p_click_B = self._eval(self.click_expr, params_B)
        p_conv_A  = self._eval(self.conversion_expr, params_A)
        p_conv_B  = self._eval(self.conversion_expr, params_B)
        return {
            "p_click_A":        p_click_A,
            "p_click_B":        p_click_B,
            "delta_ctr":        p_click_B - p_click_A,
            "p_conversion_A":   p_click_A * p_conv_A,
            "p_conversion_B":   p_click_B * p_conv_B,
            "delta_conversion": p_click_B * p_conv_B - p_click_A * p_conv_A,
        }

    def required_sample_size(self, params_A, params_B,
                             alpha: float = 0.05, power: float = 0.80) -> int:
        import math
        from scipy.stats import norm

        effects = self.theoretical_effect(params_A, params_B)
        p1    = effects["p_click_A"]
        p2    = effects["p_click_B"]
        delta = abs(effects["delta_ctr"])

        if delta == 0:
            raise ValueError("Theoretical effect is zero.")

        z_a = norm.ppf(1 - alpha / 2)
        z_b = norm.ppf(power)
        n = ((z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))) / delta ** 2
        return math.ceil(n)

    def _eval(self, expr: sp.Expr, params: Dict[str, float]) -> float:
        sym_map = {color: "color", size: "size", text: "text"}
        subs = {sym: params.get(name, 0.0) for sym, name in sym_map.items()}
        val = float(expr.subs(subs))
        return max(0.0, min(1.0, val))


class LinearButtonModel(ButtonModel):

    def __init__(self,
                 base_click: float = 0.10,
                 w_color: float = 0.05,
                 w_size:  float = 0.02,
                 w_text:  float = 0.03,
                 base_conv: float = 0.30,
                 w_conv_color: float = 0.04,
                 w_conv_size:  float = 0.01,
                 w_conv_text:  float = 0.05):
        click_expr = (
            sp.Float(base_click)
            + sp.Float(w_color) * color
            + sp.Float(w_size)  * size
            + sp.Float(w_text)  * text
        )
        conv_expr = (
            sp.Float(base_conv)
            + sp.Float(w_conv_color) * color
            + sp.Float(w_conv_size)  * size
            + sp.Float(w_conv_text)  * text
        )
        super().__init__(click_expr=click_expr, conversion_expr=conv_expr)


class LogisticButtonModel(ButtonModel):

    def __init__(self,
                 intercept_click: float = -2.2,
                 b_color: float = 0.5,
                 b_size:  float = 0.2,
                 b_text:  float = 0.3,
                 intercept_conv: float = -0.85,
                 b_conv_color: float = 0.4,
                 b_conv_size:  float = 0.1,
                 b_conv_text:  float = 0.5):
        def sigmoid(x):
            return sp.Integer(1) / (sp.Integer(1) + sp.exp(-x))

        click_expr = sigmoid(
            sp.Float(intercept_click)
            + sp.Float(b_color) * color
            + sp.Float(b_size)  * size
            + sp.Float(b_text)  * text
        )
        conv_expr = sigmoid(
            sp.Float(intercept_conv)
            + sp.Float(b_conv_color) * color
            + sp.Float(b_conv_size)  * size
            + sp.Float(b_conv_text)  * text
        )
        super().__init__(click_expr=click_expr, conversion_expr=conv_expr)


class FullSyntheticModel:
    """Oracle wrapping the ground-truth formula directly."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.click_expr      = None
        self.conversion_expr = None
        self.weights         = weights

    def predict_proba(self, params: Dict) -> float:
        return ground_truth_p_click(params, weights=self.weights)

    def _eval(self, _expr, params: Dict) -> float:
        return self.predict_proba(params)

    def describe(self):
        print("FullSyntheticModel: wraps generating_formula.p_click")


class TrainedMLModel:
    """Wraps a trained sklearn classifier (must have predict_proba)."""

    def __init__(self,
                 classifier: Any,
                 feature_builder,
                 feature_names: Optional[list] = None):
        self.classifier      = classifier
        self.feature_builder = feature_builder
        self.feature_names   = feature_names or []
        self.click_expr      = None
        self.conversion_expr = None

    def _classes(self):
        if hasattr(self.classifier, "classes_"):
            return self.classifier.classes_
        if hasattr(self.classifier, "named_steps"):
            return self.classifier.named_steps["clf"].classes_
        return None

    def predict_proba(self, params: Dict) -> float:
        X = self.feature_builder(params).reshape(1, -1)
        proba = self.classifier.predict_proba(X)
        if proba.shape[1] == 1:
            classes = self._classes()
            if classes is not None and classes[0] == 1:
                return 1.0
            return 0.0
        return float(np.clip(proba[0, 1], 0.0, 1.0))

    def predict_proba_batch(self, params_list: list) -> np.ndarray:
        X = np.vstack([self.feature_builder(p) for p in params_list])
        proba = self.classifier.predict_proba(X)
        if proba.shape[1] == 1:
            classes = self._classes()
            fill = 1.0 if (classes is not None and classes[0] == 1) else 0.0
            return np.full(len(params_list), fill)
        return np.clip(proba[:, 1], 0.0, 1.0)

    def _eval(self, _expr, params: Dict) -> float:
        return self.predict_proba(params)

    def describe(self):
        name = type(self.classifier).__name__
        print(f"TrainedMLModel wrapping: {name}")


_MODELS: Dict[str, type] = {
    "linear":   LinearButtonModel,
    "logistic": LogisticButtonModel,
}


def get_model(name: str = "linear", **kwargs) -> ButtonModel:
    if name not in _MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_MODELS)}")
    return _MODELS[name](**kwargs)
