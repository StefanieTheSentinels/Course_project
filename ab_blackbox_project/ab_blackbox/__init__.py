"""
ab_blackbox
-----------
Black-box A/B test simulator + ML response modelling + optimisation.
"""

from .model import (
    ButtonModel,
    LinearButtonModel,
    LogisticButtonModel,
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
    welch_ttest,
    bootstrap_ci,
)

from .datasets import (
    load_dataset,
    dataset_summary,
    print_dataset_summary,
    calibrate_linear_model,
    generate_synthetic_dataset,
    generate_full_synthetic_dataset,
    synthetic_dataset_summary,
)

from .generating_formula import (
    p_click,
    sample_click,
    contrast_ratio,
    relative_luminance,
    f_contrast,
    f_size,
    f_time,
    f_whitespace,
    f_scroll_decay,
    total_penalty,
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
    "ButtonModel", "LinearButtonModel", "LogisticButtonModel", "get_model",
    "TrainedMLModel", "FullSyntheticModel",
    "BlackBox", "SimulationResult",
    "run_ab_test", "run_multi_experiment", "ExperimentResult",
    "analyze", "print_report", "check_srm",
    "two_proportion_ztest", "welch_ttest", "bootstrap_ci",
    "load_dataset", "dataset_summary", "print_dataset_summary",
    "calibrate_linear_model",
    "generate_synthetic_dataset",
    "generate_full_synthetic_dataset", "synthetic_dataset_summary",
    "p_click", "sample_click",
    "contrast_ratio", "relative_luminance",
    "f_contrast", "f_size", "f_time", "f_whitespace", "f_scroll_decay",
    "total_penalty", "DEFAULT_WEIGHTS",
    "FEATURE_NAMES",
    "build_feature_vector", "build_feature_matrix",
    "get_models", "train_and_evaluate",
    "logistic_coefficients", "print_coefficient_report", "f_colour_harmony", "f_position_decay", "f_margin_balance",
]
