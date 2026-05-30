"""
experiment.py
-------------
A/B experiment runner.

`run_ab_test()` is the main entry point:
  - accepts configs for variant A (control) and variant B (treatment)
  - calls the black box once per variant
  - returns an ExperimentResult with raw data + metadata

Design choice: the experiment itself does NOT do statistics.
That is the job of analysis.py. This separation mirrors real pipelines
where data collection and analysis are distinct phases.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

from .simulator import BlackBox, SimulationResult


# ---------------------------------------------------------------------------
# Experiment result
# ---------------------------------------------------------------------------
@dataclass
class ExperimentResult:
    """
    Container for one A/B experiment run.

    Attributes
    ----------
    control   : SimulationResult for variant A
    treatment : SimulationResult for variant B
    metadata  : dict with experiment-level info (duration proxy, call count, etc.)
    """
    control:   SimulationResult
    treatment: SimulationResult
    metadata:  Dict = field(default_factory=dict)

    def summary(self) -> Dict:
        """Return a flat dict of key metrics for both variants."""
        return {
            "n_control":           self.control.n_users,
            "n_treatment":         self.treatment.n_users,
            "ctr_A":               self.control.ctr,
            "ctr_B":               self.treatment.ctr,
            "delta_ctr":           self.treatment.ctr - self.control.ctr,
            "conversion_A":        self.control.conversion_rate,
            "conversion_B":        self.treatment.conversion_rate,
            "delta_conversion":    self.treatment.conversion_rate - self.control.conversion_rate,
        }

    def srm_ratio(self) -> float:
        """
        Sample Ratio Mismatch diagnostic.
        Returns actual traffic split. Ideal = 0.5 for equal allocation.
        """
        total = self.control.n_users + self.treatment.n_users
        return self.control.n_users / total if total > 0 else float("nan")

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"ExperimentResult(\n"
            f"  CTR:        A={s['ctr_A']:.4f}  B={s['ctr_B']:.4f}  Δ={s['delta_ctr']:+.4f}\n"
            f"  Conversion: A={s['conversion_A']:.4f}  B={s['conversion_B']:.4f}"
            f"  Δ={s['delta_conversion']:+.4f}\n"
            f"  n:          A={s['n_control']}  B={s['n_treatment']}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_ab_test(
    black_box: BlackBox,
    params_A: Dict[str, float],
    params_B: Dict[str, float],
    n_users_per_variant: Optional[int] = None,
    label: str = "experiment",
) -> ExperimentResult:
    """
    Run a single A/B experiment.

    Parameters
    ----------
    black_box : BlackBox instance
    params_A  : button parameters for the control variant
    params_B  : button parameters for the treatment variant
    n_users_per_variant : override black_box.n_users if provided
    label     : identifier stored in metadata

    Returns
    -------
    ExperimentResult
    """
    calls_before = black_box.call_count

    result_A = black_box(params_A, n_users=n_users_per_variant)
    result_B = black_box(params_B, n_users=n_users_per_variant)

    return ExperimentResult(
        control=result_A,
        treatment=result_B,
        metadata={
            "label":       label,
            "params_A":    params_A,
            "params_B":    params_B,
            "calls_used":  black_box.call_count - calls_before,
        },
    )


def run_multi_experiment(
    black_box: BlackBox,
    configs: list[tuple[Dict, Dict]],
    n_users_per_variant: Optional[int] = None,
) -> list[ExperimentResult]:
    """
    Run multiple A/B experiments sequentially.

    Parameters
    ----------
    configs : list of (params_A, params_B) tuples
              Each tuple is one experiment.

    Returns
    -------
    list of ExperimentResult, one per config

    Use case
    --------
    Simulating repeated experiments to estimate power empirically,
    or sweeping over parameter combinations.
    """
    results = []
    for i, (params_A, params_B) in enumerate(configs):
        result = run_ab_test(
            black_box=black_box,
            params_A=params_A,
            params_B=params_B,
            n_users_per_variant=n_users_per_variant,
            label=f"experiment_{i}",
        )
        results.append(result)
    return results
