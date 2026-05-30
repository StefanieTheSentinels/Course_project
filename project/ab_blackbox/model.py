"""
model.py
--------
Symbolic response models built with SymPy.

LinearButtonModel  : p = base + w_color*color + w_size*size + w_text*text
LogisticButtonModel: p = sigmoid(a + b*color + ...)
get_model          : factory
"""

from __future__ import annotations
import sympy as sp
from dataclasses import dataclass
from typing import Dict


color = sp.Symbol("color", real=True)
size  = sp.Symbol("size",  real=True)
text  = sp.Symbol("text",  real=True)


@dataclass
class ButtonModel:
    # Abstract button response model.
    click_expr: sp.Expr
    conversion_expr: sp.Expr

    def describe(self) -> str:
        # Print symbolic click and conversion expressions.
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

    def theoretical_effect(
        self,
        params_A: Dict[str, float],
        params_B: Dict[str, float],
    ) -> Dict[str, float]:
        # Population-level treatment effect (no sampling noise).
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

    def required_sample_size(
        self,
        params_A: Dict[str, float],
        params_B: Dict[str, float],
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """
        Minimum per-group n for a two-proportion z-test.
            n = (z_a/2 + z_b)^2 * (p1*(1-p1) + p2*(1-p2)) / delta^2
        """
        import math
        from scipy.stats import norm

        effects = self.theoretical_effect(params_A, params_B)
        p1    = effects["p_click_A"]
        p2    = effects["p_click_B"]
        delta = abs(effects["delta_ctr"])

        if delta == 0:
            raise ValueError("Theoretical effect is zero — variants are identical.")

        z_a = norm.ppf(1 - alpha / 2)
        z_b = norm.ppf(power)
        n   = ((z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))) / delta ** 2
        return math.ceil(n)

    def _eval(self, expr: sp.Expr, params: Dict[str, float]) -> float:
        # Evaluate a sympy expression given a parameter dict, clamped to [0,1].
        sym_map = {color: "color", size: "size", text: "text"}
        subs = {sym: params.get(name, 0.0) for sym, name in sym_map.items()}
        val = float(expr.subs(subs))
        return max(0.0, min(1.0, val))


class LinearButtonModel(ButtonModel):
    # p_click = base_click + w_color*color + w_size*size + w_text*text

    def __init__(
        self,
        base_click: float = 0.10,
        w_color: float = 0.05,
        w_size:  float = 0.02,
        w_text:  float = 0.03,
        base_conv: float = 0.30,
        w_conv_color: float = 0.04,
        w_conv_size:  float = 0.01,
        w_conv_text:  float = 0.05,
    ):
        click_expr = (
            sp.Float(base_click)
            + sp.Float(w_color) * color
            + sp.Float(w_size)  * size
            + sp.Float(w_text)  * text
        )
        conversion_expr = (
            sp.Float(base_conv)
            + sp.Float(w_conv_color) * color
            + sp.Float(w_conv_size)  * size
            + sp.Float(w_conv_text)  * text
        )
        super().__init__(click_expr=click_expr, conversion_expr=conversion_expr)


class LogisticButtonModel(ButtonModel):
    # p_click = sigmoid(a + b_color*color + b_size*size + b_text*text)

    def __init__(
        self,
        intercept_click: float = -2.2,
        b_color: float = 0.5,
        b_size:  float = 0.2,
        b_text:  float = 0.3,
        intercept_conv: float = -0.85,
        b_conv_color: float = 0.4,
        b_conv_size:  float = 0.1,
        b_conv_text:  float = 0.5,
    ):
        def sigmoid(x: sp.Expr) -> sp.Expr:
            return sp.Integer(1) / (sp.Integer(1) + sp.exp(-x))

        click_expr = sigmoid(
            sp.Float(intercept_click)
            + sp.Float(b_color) * color
            + sp.Float(b_size)  * size
            + sp.Float(b_text)  * text
        )
        conversion_expr = sigmoid(
            sp.Float(intercept_conv)
            + sp.Float(b_conv_color) * color
            + sp.Float(b_conv_size)  * size
            + sp.Float(b_conv_text)  * text
        )
        super().__init__(click_expr=click_expr, conversion_expr=conversion_expr)


_MODELS: Dict[str, type] = {
    "linear":   LinearButtonModel,
    "logistic": LogisticButtonModel,
}


def get_model(name: str = "linear", **kwargs) -> ButtonModel:
    # Return a model instance by name. Options: 'linear' | 'logistic'.
    if name not in _MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_MODELS)}")
    return _MODELS[name](**kwargs)
