"""
ab_blackbox
-----------
Black-box A/B test simulator.

    >>> from ab_blackbox import BlackBox, run_ab_test, analyze, print_report, get_model
    >>> box    = BlackBox(model=get_model("linear"), n_users=2000, seed=42)
    >>> result = run_ab_test(box, {"color": 0.0}, {"color": 1.0})
    >>> print_report(analyze(result, run_bootstrap=True))
"""

from .model      import get_model, LinearButtonModel, LogisticButtonModel, ButtonModel
from .simulator  import BlackBox, SimulationResult
from .experiment import run_ab_test, run_multi_experiment, ExperimentResult
from .analysis   import (
    analyze,
    print_report,
    check_srm,
    two_proportion_ztest,
    welch_ttest,
    bootstrap_ci,
)
from .datasets   import (
    load_dataset,
    dataset_summary,
    print_dataset_summary,
    calibrate_linear_model,
    generate_synthetic_dataset,
)

__all__ = [
    "get_model", "LinearButtonModel", "LogisticButtonModel", "ButtonModel",
    "BlackBox", "SimulationResult",
    "run_ab_test", "run_multi_experiment", "ExperimentResult",
    "analyze", "print_report", "check_srm",
    "two_proportion_ztest", "welch_ttest", "bootstrap_ci",
    "load_dataset", "dataset_summary", "print_dataset_summary",
    "calibrate_linear_model", "generate_synthetic_dataset",
]
