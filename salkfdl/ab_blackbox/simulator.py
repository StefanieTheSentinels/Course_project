"""
simulator.py
------------
Black-box simulator.

A single call to `black_box()` is the "expensive experiment":
  - takes button configuration (dict of parameter values)
  - evaluates the symbolic response model numerically
  - samples n_users Bernoulli trials for clicks and conversions
  - returns aggregate metrics

This is intentionally opaque from the outside: callers see only
inputs (button params) and outputs (metrics). The symbolic model
inside is the mechanism, but the black-box interface hides it.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional
from .model import ButtonModel, LinearButtonModel


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
class SimulationResult:
    """Holds output of one black-box call."""

    def __init__(
        self,
        params: Dict[str, float],
        clicks: np.ndarray,
        conversions: np.ndarray,
        n_users: int,
    ):
        self.params      = params
        self.n_users     = n_users
        self.n_clicks    = int(clicks.sum())
        self.n_conversions = int(conversions.sum())
        self.ctr         = float(clicks.mean())
        self.conversion_rate = float(conversions.mean())
        # Raw arrays kept for analysis (e.g., bootstrap, t-test)
        self._clicks      = clicks
        self._conversions = conversions

    def __repr__(self) -> str:
        return (
            f"SimulationResult("
            f"n={self.n_users}, "
            f"ctr={self.ctr:.4f}, "
            f"conversion_rate={self.conversion_rate:.4f})"
        )

    def to_dict(self) -> Dict:
        return {
            "params":           self.params,
            "n_users":          self.n_users,
            "n_clicks":         self.n_clicks,
            "n_conversions":    self.n_conversions,
            "ctr":              self.ctr,
            "conversion_rate":  self.conversion_rate,
        }


# ---------------------------------------------------------------------------
# Black box
# ---------------------------------------------------------------------------
class BlackBox:
    """
    The black-box simulator.

    Parameters
    ----------
    model : ButtonModel
        Symbolic response model. Default: LinearButtonModel().
    n_users : int
        Number of users simulated per call.
    noise_std : float
        Gaussian noise added to p_click before sampling (models
        unobserved confounders). Set to 0 for a deterministic response surface.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model: Optional[ButtonModel] = None,
        n_users: int = 1000,
        noise_std: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.model     = model if model is not None else LinearButtonModel()
        self.n_users   = n_users
        self.noise_std = noise_std
        self.rng       = np.random.default_rng(seed)
        self._call_count = 0  # track how many times the box was queried

    @property
    def call_count(self) -> int:
        """Number of times the black box has been called."""
        return self._call_count

    def __call__(
        self,
        params: Dict[str, float],
        n_users: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run one experiment call.

        Parameters
        ----------
        params : dict with keys "color", "size", "text" (all in [0,1])
                 Missing keys default to 0.0 (baseline).
        n_users : override instance default if provided.

        Returns
        -------
        SimulationResult
        """
        self._call_count += 1
        n = n_users if n_users is not None else self.n_users

        # --- evaluate symbolic model ---
        p_click = self.model._eval(self.model.click_expr, params)
        p_conv  = self.model._eval(self.model.conversion_expr, params)

        # --- add noise to simulate unobserved variation ---
        if self.noise_std > 0:
            p_click = float(np.clip(
                p_click + self.rng.normal(0, self.noise_std), 0.0, 1.0
            ))

        # --- sample user behavior ---
        clicks      = self.rng.binomial(1, p_click, size=n)
        # Conversion only possible if user clicked
        conversions = clicks * self.rng.binomial(1, p_conv, size=n)

        return SimulationResult(
            params=params,
            clicks=clicks,
            conversions=conversions,
            n_users=n,
        )

    def reset_call_count(self) -> None:
        self._call_count = 0
