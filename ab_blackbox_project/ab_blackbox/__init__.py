"""
ab_blackbox
-----------
Black-box A/B test simulator + ML response modelling + optimisation.
"""

from .model import (
    ButtonModel,
    LinearButtonModel,
    get_model,
    TrainedMLModel,
    FullSyntheticModel,
)

from .simulator import BlackBox, SimulationResult

from .experiment import (
    run_ab_test,
    run_multi_experiment,
    ExperimentResult,
)

from .analysis import (
    analyze,
    print_report,
    check_srm,
    two_proportion_ztest,
    bootstrap_ci,
)

from .datasets import (
    generate_full_synthetic_dataset,
    synthetic_dataset_summary,
)

from .generating_formula import (
    p_click,
    contrast_ratio,
    relative_luminance,
    f_contrast,
    f_size,
    f_time,
    f_whitespace,
    f_colour_harmony,
    f_position_decay,
    f_margin_balance,
    DEFAULT_WEIGHTS,
)

from .training import (
    FEATURE_NAMES,
    build_feature_vector,
    build_feature_matrix,
    get_models,
    train_and_evaluate,
    logistic_coefficients,
    print_coefficient_report,
)


__all__ = [
    # model
    "ButtonModel", "LinearButtonModel", "get_model",
    "TrainedMLModel", "FullSyntheticModel",
    # simulator
    "BlackBox", "SimulationResult",
    # experiment
    "run_ab_test", "run_multi_experiment", "ExperimentResult",
    # analysis
    "analyze", "print_report", "check_srm",
    "two_proportion_ztest", "bootstrap_ci",
    # datasets
    "calibrate_linear_model",
    "generate_full_synthetic_dataset", "synthetic_dataset_summary",
    # generating_formula
    "p_click",
    "contrast_ratio", "relative_luminance",
    "f_contrast", "f_size", "f_time", "f_whitespace", "f_scroll_decay",
    "total_penalty",
    "f_colour_harmony", "f_position_decay", "f_margin_balance",
    "DEFAULT_WEIGHTS",
    # training
    "FEATURE_NAMES",
    "build_feature_vector", "build_feature_matrix",
    "get_models", "train_and_evaluate",
    "logistic_coefficients", "print_coefficient_report",
]