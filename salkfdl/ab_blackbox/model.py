"""
model.py
--------
Symbolic response models using SymPy.

Each model defines click_prob and conversion_prob as symbolic expressions
over button parameters. The simulator evaluates these numerically.

Exported
--------
ButtonModel          : base class
LinearButtonModel    : p = base + w_color*color + w_size*size + w_text*text
LogisticButtonModel  : p = sigmoid(a + b_color*color + b_size*size + b_text*text)
get_model            : factory function
"""

from __future__ import annotations
import sympy as sp
from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Symbolic variables (shared across models)
# ---------------------------------------------------------------------------
color = sp.Symbol("color", real=True)   # in [0, 1], 0=default, 1=black
size  = sp.Symbol("size",  real=True)   # in [0, 1], 0=small, 1=large
text  = sp.Symbol("text",  real=True)   # in [0, 1], 0=generic, 1=specific

PARAMS = (color, size, text)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
@dataclass
class ButtonModel:
    """
    Abstract button response model.

    Attributes
    ----------
    click_expr      : sympy expression for click probability
    conversion_expr : sympy expression for conversion probability (given click)
    param_symbols   : tuple of free sympy symbols that are input parameters
    """
    click_expr: sp.Expr
    conversion_expr: sp.Expr
    param_symbols: tuple = field(default_factory=lambda: PARAMS)

    def describe(self) -> None:
        """Print the symbolic expressions."""
        print("Click probability:")
        sp.pprint(self.click_expr)
        print("\nConversion probability (given click):")
        sp.pprint(self.conversion_expr)

    def theoretical_effect(
        self,
        params_A: Dict[str, float],
        params_B: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute the theoretical (population-level) treatment effect analytically.

        Returns delta_ctr and delta_conversion before any sampling noise.
        """
        sym_map = {color: "color", size: "size", text: "text"}

        def evaluate(expr, params):
            subs = {sym: params.get(name, 0.0) for sym, name in sym_map.items()}
            return float(expr.subs(subs))

        p_click_A = evaluate(self.click_expr, params_A)
        p_click_B = evaluate(self.click_expr, params_B)
        p_conv_A  = evaluate(self.conversion_expr, params_A)
        p_conv_B  = evaluate(self.conversion_expr, params_B)

        return {
            "p_click_A":       p_click_A,
            "p_click_B":       p_click_B,
            "delta_ctr":       p_click_B - p_click_A,
            "p_conversion_A":  p_click_A * p_conv_A,
            "p_conversion_B":  p_click_B * p_conv_B,
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
        Analytically derive the minimum per-group n for a two-proportion z-test
        using the symbolic effect size.

        Formula
        -------
        n = (z_alpha/2 + z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / delta^2

        This uses SymPy to first compute p_A and p_B symbolically, then
        evaluates numerically.
        """
        import math
        from scipy.stats import norm

        effects = self.theoretical_effect(params_A, params_B)
        p1 = effects["p_click_A"]
        p2 = effects["p_click_B"]
        delta = abs(effects["delta_ctr"])

        if delta == 0:
            raise ValueError("Theoretical effect is zero — variants are identical.")

        z_a = norm.ppf(1 - alpha / 2)
        z_b = norm.ppf(power)
        n = ((z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))) / delta ** 2
        return math.ceil(n)

    def _eval(self, expr: sp.Expr, params: Dict[str, float]) -> float:
        """Evaluate a sympy expression given a parameter dict."""
        sym_map = {color: "color", size: "size", text: "text"}
        subs = {sym: params.get(name, 0.0) for sym, name in sym_map.items()}
        val = float(expr.subs(subs))
        return max(0.0, min(1.0, val))  # clamp to [0, 1]


# ---------------------------------------------------------------------------
# Linear model
# ---------------------------------------------------------------------------
class LinearButtonModel(ButtonModel):
    """
    Click probability is a linear function of button parameters.

        p_click = base_click + w_color*color + w_size*size + w_text*text

    Naturally clipped to [0,1] at evaluation time.
    """

    def __init__(
        self,
        base_click: float = 0.10,
        w_color: float = 0.05,
        w_size: float  = 0.02,
        w_text: float  = 0.03,
        base_conv: float = 0.30,
        w_conv_color: float = 0.04,
        w_conv_size: float  = 0.01,
        w_conv_text: float  = 0.05,
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


# ---------------------------------------------------------------------------
# Logistic model
# ---------------------------------------------------------------------------
class LogisticButtonModel(ButtonModel):
    """
    Click probability is a logistic (sigmoid) function — naturally in (0,1).

        p_click = sigmoid(a + b_color*color + b_size*size + b_text*text)
    """

    def __init__(
        self,
        intercept_click: float = -2.2,   # sigmoid(-2.2) ≈ 0.10
        b_color: float = 0.5,
        b_size:  float = 0.2,
        b_text:  float = 0.3,
        intercept_conv: float = -0.85,   # sigmoid(-0.85) ≈ 0.30
        b_conv_color: float = 0.4,
        b_conv_size:  float = 0.1,
        b_conv_text:  float = 0.5,
    ):
        def sigmoid(x):
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_MODELS = {
    "linear":   LinearButtonModel,
    "logistic": LogisticButtonModel,
}

def get_model(name: str = "linear", **kwargs) -> ButtonModel:
    """
    Factory function.

    Parameters
    ----------
    name : "linear" | "logistic"
    **kwargs : passed to the model constructor
    """
    if name not in _MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_MODELS)}")
    return _MODELS[name](**kwargs)
