"""
simulator.py
------------
BlackBox wraps the symbolic model into a single callable experiment:
params in → sample n_users Bernoulli trials → aggregate metrics out.
Conversions require a prior click.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Optional

from .model import ButtonModel, LinearButtonModel


# Result dataclass
@dataclass
class SimulationResult:
    """Holds output of one black-box call."""

    params:          Dict[str, float]
    n_users:         int
    n_clicks:        int
    n_conversions:   int
    ctr:             float
    conversion_rate: float
    clicks:          np.ndarray
    conversions:     np.ndarray

    def __repr__(self) -> str:
        return (
            f"SimulationResult("
            f"n={self.n_users}, "
            f"ctr={self.ctr:.4f}, "
            f"conversion_rate={self.conversion_rate:.4f})"
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


# Black box
class BlackBox:
    """
    Black-box simulator.

    Parameters:
    model     : ButtonModel – symbolic response model (default: LinearButtonModel)
    n_users   : int         – users simulated per call
    noise_std : float       – Gaussian noise on p_click per arm; use 0 for calibration
    seed      : int | None  – random seed
    """

    def __init__(
        self,
        model: Optional[ButtonModel] = None,
        n_users: int = 1000,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.model     = model if model is not None else LinearButtonModel()
        self.n_users   = n_users
        self.noise_std = noise_std
        self.rng       = np.random.default_rng(seed)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        # Number of times the black box has been called.
        return self._call_count

    def __call__(
        self,
        params: Dict[str, float],
        n_users: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run one experiment call.

        params  : dict with keys "color", "size", "text" (all in [0,1])
        n_users : override instance default if provided
        """
        self._call_count += 1
        n = n_users if n_users is not None else self.n_users

        # Evaluate symbolic model
        p_click = self.model._eval(self.model.click_expr, params)
        p_conv  = self.model._eval(self.model.conversion_expr, params)

        # Optional between-experiment noise on population CTR
        if self.noise_std > 0:
            p_click = float(np.clip(
                p_click + self.rng.normal(0, self.noise_std), 0.0, 1.0
            ))

        # Sample user behaviour
        clicks = self.rng.binomial(1, p_click, size=n)

        # Conversions: conditional on click — p_conv = P(conversion | click)
        conversions = clicks * self.rng.binomial(1, p_conv, size=n)

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
