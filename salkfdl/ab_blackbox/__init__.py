"""
ab_blackbox
-----------
Plug-and-play Python package for black-box A/B test simulation.

Quick start
-----------
>>> from ab_blackbox import BlackBox, run_ab_test, analyze, print_report
>>> from ab_blackbox import get_model

>>> model = get_model("linear")
>>> box   = BlackBox(model=model, n_users=2000, seed=42)

>>> params_A = {"color": 0.0, "size": 0.0, "text": 0.0}  # default button
>>> params_B = {"color": 1.0, "size": 0.0, "text": 0.0}  # black button

>>> result   = run_ab_test(box, params_A, params_B)
>>> report   = analyze(result, run_bootstrap=True)
>>> print_report(report)
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
    bonferroni_correction,
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
    "two_proportion_ztest", "welch_ttest", "bootstrap_ci", "bonferroni_correction",
    "load_dataset", "dataset_summary", "print_dataset_summary",
    "calibrate_linear_model", "generate_synthetic_dataset",
]
