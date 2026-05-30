"""
BlackBox simulator. Works with both symbolic and ML models via duck typing.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Optional, Any

from .model import ButtonModel, LinearButtonModel


@dataclass
class SimulationResult:
    params:          Dict[str, Any]
    n_users:         int
    n_clicks:        int
    n_conversions:   int
    ctr:             float
    conversion_rate: float
    clicks:          np.ndarray
    conversions:     np.ndarray

    def __repr__(self) -> str:
        return (
            f"SimulationResult(n={self.n_users}, "
            f"ctr={self.ctr:.4f}, conversion_rate={self.conversion_rate:.4f})"
        )

    def to_dict(self) -> Dict:
        return {
            "params":          self.params,
            "n_users":         self.n_users,
            "n_clicks":        self.n_clicks,
            "n_conversions":   self.n_conversions,
            "ctr":             self.ctr,
            "conversion_rate": self.conversion_rate,
        }


class BlackBox:

    def __init__(self,
                 model: Optional[Any] = None,
                 n_users: int = 1000,
                 noise_std: float = 0.0,
                 seed: Optional[int] = None):
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if n_users <= 0:
            raise ValueError("n_users must be > 0")
        self.model       = model if model is not None else LinearButtonModel()
        self.n_users     = n_users
        self.noise_std   = noise_std
        self.rng         = np.random.default_rng(seed)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def _get_p_click(self, params: Dict) -> float:
        if hasattr(self.model, "click_expr") and self.model.click_expr is not None:
            return float(self.model._eval(self.model.click_expr, params))
        return float(self.model._eval(None, params))

    def _get_p_conv(self, params: Dict) -> float:
        if hasattr(self.model, "conversion_expr") and self.model.conversion_expr is not None:
            return float(self.model._eval(self.model.conversion_expr, params))
        return 0.0

    def __call__(self,
                 params: Dict,
                 n_users: Optional[int] = None) -> SimulationResult:
        self._call_count += 1
        n = n_users if n_users is not None else self.n_users

        p_click = self._get_p_click(params)
        p_conv  = self._get_p_conv(params)

        if self.noise_std > 0:
            logit = np.log(p_click / (1 - p_click + 1e-12))
            logit += self.rng.normal(0, self.noise_std * 4)  # *4 ≈ компенсация масштаба
            p_click = 1.0 / (1.0 + np.exp(-logit))

        clicks = self.rng.binomial(1, p_click, size=n)

        if p_conv > 0:
            conversions = clicks * self.rng.binomial(1, p_conv, size=n)
        else:
            conversions = np.zeros_like(clicks)

        return SimulationResult(
            params=params,
            n_users=n,
            n_clicks=int(clicks.sum()),
            n_conversions=int(conversions.sum()),
            ctr=float(clicks.mean()),
            conversion_rate=float(conversions.mean()),
            clicks=clicks,
            conversions=conversions,
        )

    def reset_call_count(self) -> None:
        self._call_count = 0
