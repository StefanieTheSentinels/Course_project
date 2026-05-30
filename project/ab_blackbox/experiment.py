"""
experiment.py
-------------
Thin runner: calls the black box once per variant, returns ExperimentResult.
All statistics live in analysis.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .simulator import BlackBox, SimulationResult


# Experiment result
@dataclass
class ExperimentResult:
    """
    Container for one A/B experiment run.

    control   : SimulationResult for variant A
    treatment : SimulationResult for variant B
    metadata  : experiment-level info (label, params, call count)
    """
    control:   SimulationResult
    treatment: SimulationResult
    metadata:  Dict = field(default_factory=dict)

    def summary(self) -> Dict:
        # Return a flat dict of key metrics for both variants.
        return {
            "n_control":        self.control.n_users,
            "n_treatment":      self.treatment.n_users,
            "ctr_A":            self.control.ctr,
            "ctr_B":            self.treatment.ctr,
            "delta_ctr":        self.treatment.ctr - self.control.ctr,
            "conversion_A":     self.control.conversion_rate,
            "conversion_B":     self.treatment.conversion_rate,
            "delta_conversion": self.treatment.conversion_rate - self.control.conversion_rate,
        }

    def srm_ratio(self) -> float:
        # Actual traffic split. Ideal = 0.5 for equal allocation.
        total = self.control.n_users + self.treatment.n_users
        return self.control.n_users / total if total > 0 else float("nan")

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"ExperimentResult(\n"
            f"  CTR:        A={s['ctr_A']:.4f}  B={s['ctr_B']:.4f}  "
            f"delta={s['delta_ctr']:+.4f}\n"
            f"  Conversion: A={s['conversion_A']:.4f}  B={s['conversion_B']:.4f}  "
            f"delta={s['delta_conversion']:+.4f}\n"
            f"  n:          A={s['n_control']}  B={s['n_treatment']}\n"
            f")"
        )


# Runner
def run_ab_test(
    black_box: BlackBox,
    params_A: Dict[str, float],
    params_B: Dict[str, float],
    n_users_per_variant: Optional[int] = None,
    label: str = "experiment",
) -> ExperimentResult:
    """
    Run a single A/B experiment (two black-box calls).

    black_box           : BlackBox instance
    params_A            : button parameters for control
    params_B            : button parameters for treatment
    n_users_per_variant : override black_box.n_users if provided
    label               : stored in metadata
    """
    calls_before = black_box.call_count
    result_A = black_box(params_A, n_users=n_users_per_variant)
    result_B = black_box(params_B, n_users=n_users_per_variant)

    return ExperimentResult(
        control=result_A,
        treatment=result_B,
        metadata={
            "label":      label,
            "params_A":   params_A,
            "params_B":   params_B,
            "calls_used": black_box.call_count - calls_before,
        },
    )


def run_multi_experiment(
    black_box: BlackBox,
    configs: List[Tuple[Dict, Dict]],
    n_users_per_variant: Optional[int] = None,
) -> List[ExperimentResult]:
    
    # Run multiple A/B experiments sequentially.

    return [
        run_ab_test(
            black_box=black_box,
            params_A=pa,
            params_B=pb,
            n_users_per_variant=n_users_per_variant,
            label=f"experiment_{i}",
        )
        for i, (pa, pb) in enumerate(configs)
    ]
