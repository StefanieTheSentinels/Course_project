"""
analysis.py
-----------
Statistical analysis of A/B experiment results.

Covers
------
1. SRM check          — chi-square test for sample ratio mismatch
2. Two-proportion z-test  — primary metric (CTR, conversion)
3. Welch t-test           — alternative when variance differs
4. Bootstrap CI           — non-parametric confidence interval
5. Multiple testing correction — Bonferroni for secondary metrics
6. Decision rule          — ship / reject / inconclusive

All functions accept an ExperimentResult and return structured dicts
so results can be logged, printed, or fed into further analysis.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Dict, List, Literal, Tuple

from .experiment import ExperimentResult


# ---------------------------------------------------------------------------
# 1. SRM check
# ---------------------------------------------------------------------------
def check_srm(
    result: ExperimentResult,
    expected_split: float = 0.5,
    alpha: float = 0.01,  # strict: SRM is a data quality issue
) -> Dict:
    """
    Chi-square test for sample ratio mismatch.

    H0: observed traffic split equals expected_split.

    A significant result means the randomization or logging is broken —
    stop analysis and investigate before interpreting any metrics.

    Parameters
    ----------
    expected_split : fraction of traffic expected in control (default 0.5)
    alpha          : significance level (0.01 recommended — SRM is critical)

    Returns
    -------
    dict with keys: srm_detected, p_value, observed_split, expected_split
    """
    n_A = result.control.n_users
    n_B = result.treatment.n_users
    N   = n_A + n_B

    expected_A = N * expected_split
    expected_B = N * (1 - expected_split)

    chi2, p_value = stats.chisquare(
        f_obs=[n_A, n_B],
        f_exp=[expected_A, expected_B],
    )

    return {
        "srm_detected":   p_value < alpha,
        "chi2_statistic": float(chi2),
        "p_value":        float(p_value),
        "observed_split": n_A / N,
        "expected_split": expected_split,
        "warning": (
            "SRM detected — results are unreliable. Investigate assignment/logging."
            if p_value < alpha else None
        ),
    }


# ---------------------------------------------------------------------------
# 2. Two-proportion z-test
# ---------------------------------------------------------------------------
def two_proportion_ztest(
    n_A: int,
    x_A: int,
    n_B: int,
    x_B: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> Dict:
    """
    Two-proportion z-test for H0: p_B = p_A.

    Parameters
    ----------
    n_A, n_B : sample sizes
    x_A, x_B : number of successes (clicks or conversions)
    alpha     : significance level
    alternative : direction of H1

    Returns
    -------
    dict with: p_hat_A, p_hat_B, delta, z_stat, p_value, significant, ci_95
    """
    p_A = x_A / n_A
    p_B = x_B / n_B
    delta = p_B - p_A

    # pooled proportion under H0
    p_pool = (x_A + x_B) / (n_A + n_B)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_A + 1 / n_B))

    if se == 0:
        return {"error": "Standard error is zero — check input data."}

    z = delta / se

    if alternative == "two-sided":
        p_value = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        p_value = stats.norm.sf(z)
    else:
        p_value = stats.norm.cdf(z)

    # 95% CI on delta using unpooled SE
    se_unpooled = np.sqrt(p_A * (1 - p_A) / n_A + p_B * (1 - p_B) / n_B)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (delta - z_crit * se_unpooled, delta + z_crit * se_unpooled)

    return {
        "p_hat_A":     float(p_A),
        "p_hat_B":     float(p_B),
        "delta":       float(delta),
        "relative_lift": float(delta / p_A) if p_A > 0 else float("nan"),
        "z_statistic": float(z),
        "p_value":     float(p_value),
        "significant": bool(p_value < alpha),
        "ci_95":       (float(ci[0]), float(ci[1])),
        "alpha":       alpha,
        "alternative": alternative,
    }


# ---------------------------------------------------------------------------
# 3. Welch t-test (per-user metric)
# ---------------------------------------------------------------------------
def welch_ttest(
    sample_A: np.ndarray,
    sample_B: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Welch t-test on two independent samples.

    More appropriate than z-test when:
    - metric is continuous (e.g., session length, revenue per user)
    - variances differ between groups

    Parameters
    ----------
    sample_A, sample_B : 1-D arrays of per-user metric values

    Returns
    -------
    dict with: mean_A, mean_B, delta, t_stat, p_value, significant, ci_95
    """
    t_stat, p_value = stats.ttest_ind(sample_A, sample_B, equal_var=False)

    mean_A = float(sample_A.mean())
    mean_B = float(sample_B.mean())
    delta  = mean_B - mean_A

    # CI via scipy
    ci = stats.t.interval(
        confidence=1 - alpha,
        df=len(sample_A) + len(sample_B) - 2,
        loc=delta,
        scale=stats.sem(np.concatenate([sample_B - mean_B, sample_A - mean_A])),
    )

    return {
        "mean_A":      mean_A,
        "mean_B":      mean_B,
        "delta":       delta,
        "relative_lift": delta / mean_A if mean_A != 0 else float("nan"),
        "t_statistic": float(t_stat),
        "p_value":     float(p_value),
        "significant": bool(p_value < alpha),
        "ci_95":       (float(ci[0]), float(ci[1])),
        "alpha":       alpha,
    }


# ---------------------------------------------------------------------------
# 4. Bootstrap confidence interval
# ---------------------------------------------------------------------------
def bootstrap_ci(
    sample_A: np.ndarray,
    sample_B: np.ndarray,
    statistic=np.mean,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """
    Non-parametric bootstrap CI for the difference in a statistic.

    delta_boot[i] = statistic(resample_B) - statistic(resample_A)

    Useful as a robustness check alongside the z-test.
    """
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        resample_A = rng.choice(sample_A, size=len(sample_A), replace=True)
        resample_B = rng.choice(sample_B, size=len(sample_B), replace=True)
        deltas[i] = statistic(resample_B) - statistic(resample_A)

    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    point_estimate = float(statistic(sample_B) - statistic(sample_A))

    return {
        "point_estimate": point_estimate,
        "ci_lower":       lo,
        "ci_upper":       hi,
        "ci_excludes_zero": not (lo <= 0 <= hi),
        "n_bootstrap":    n_bootstrap,
        "alpha":          alpha,
    }


# ---------------------------------------------------------------------------
# 5. Multiple testing correction (Bonferroni)
# ---------------------------------------------------------------------------
def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict:
    """
    Bonferroni correction for multiple metrics.

    Adjusted threshold = alpha / m where m = number of tests.

    Conservative but appropriate for guardrail metrics where
    we want to avoid false positives.

    Returns
    -------
    dict with adjusted_alpha and per-test significance flags
    """
    m = len(p_values)
    adjusted_alpha = alpha / m
    return {
        "n_tests":        m,
        "original_alpha": alpha,
        "adjusted_alpha": adjusted_alpha,
        "significant":    [p < adjusted_alpha for p in p_values],
        "p_values":       p_values,
    }


# ---------------------------------------------------------------------------
# 6. Full analysis pipeline
# ---------------------------------------------------------------------------
def analyze(
    result: ExperimentResult,
    alpha: float = 0.05,
    primary_metric: Literal["ctr", "conversion"] = "ctr",
    run_bootstrap: bool = False,
    n_bootstrap: int = 2000,
) -> Dict:
    """
    Full analysis pipeline for one ExperimentResult.

    Steps
    -----
    1. SRM check          — halt flag if mismatch detected
    2. Primary metric     — two-proportion z-test on chosen metric
    3. Secondary metric   — the other metric, Bonferroni corrected
    4. Bootstrap CI       — optional robustness check
    5. Decision rule      — ship / reject / inconclusive

    Returns
    -------
    Nested dict with all results. Top-level keys:
    srm, primary, secondary, bootstrap (if requested), decision
    """
    A = result.control
    B = result.treatment

    # 1. SRM
    srm = check_srm(result, alpha=0.01)

    # 2. Primary metric
    if primary_metric == "ctr":
        primary = two_proportion_ztest(
            A.n_users, A.n_clicks,
            B.n_users, B.n_clicks,
            alpha=alpha,
        )
        secondary = two_proportion_ztest(
            A.n_users, A.n_conversions,
            B.n_users, B.n_conversions,
            alpha=alpha,
        )
        primary_label   = "CTR"
        secondary_label = "Conversion"
    else:
        primary = two_proportion_ztest(
            A.n_users, A.n_conversions,
            B.n_users, B.n_conversions,
            alpha=alpha,
        )
        secondary = two_proportion_ztest(
            A.n_users, A.n_clicks,
            B.n_users, B.n_clicks,
            alpha=alpha,
        )
        primary_label   = "Conversion"
        secondary_label = "CTR"

    # 3. Bonferroni on both metrics together
    multi = bonferroni_correction(
        [primary["p_value"], secondary["p_value"]], alpha=alpha
    )

    # 4. Bootstrap (optional)
    boot = None
    if run_bootstrap:
        if primary_metric == "ctr":
            boot = bootstrap_ci(A._clicks, B._clicks, n_bootstrap=n_bootstrap)
        else:
            boot = bootstrap_ci(A._conversions, B._conversions, n_bootstrap=n_bootstrap)

    # 5. Decision rule
    decision = _decision_rule(
        srm_detected=srm["srm_detected"],
        primary_significant=primary["significant"],
        primary_positive=primary.get("delta", 0) > 0,
        secondary_significant=secondary["significant"],
        secondary_positive=secondary.get("delta", 0) >= 0,
    )

    return {
        "srm":             srm,
        "primary":         {**primary, "metric": primary_label},
        "secondary":       {**secondary, "metric": secondary_label},
        "multiple_testing": multi,
        "bootstrap":       boot,
        "decision":        decision,
    }


def _decision_rule(
    srm_detected: bool,
    primary_significant: bool,
    primary_positive: bool,
    secondary_significant: bool,
    secondary_positive: bool,
) -> Dict:
    """
    Decision framework:

    1. SRM detected           → INVALID (do not interpret)
    2. Primary significant + positive + secondary not harmful
                              → SHIP
    3. Primary significant + negative → REJECT
    4. Primary not significant        → INCONCLUSIVE
    """
    if srm_detected:
        verdict = "INVALID"
        reason  = "SRM detected. Investigate randomization and logging before re-running."
    elif primary_significant and primary_positive and secondary_positive:
        verdict = "SHIP"
        reason  = "Primary metric improved significantly with no secondary regression."
    elif primary_significant and not primary_positive:
        verdict = "REJECT"
        reason  = "Primary metric significantly worse in treatment."
    elif primary_significant and not secondary_positive:
        verdict = "REVIEW"
        reason  = "Primary improved but secondary metric regressed — review trade-off."
    else:
        verdict = "INCONCLUSIVE"
        reason  = "No significant effect detected. Consider increasing sample size or experiment duration."

    return {"verdict": verdict, "reason": reason}


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def print_report(analysis_result: Dict) -> None:
    """Print a human-readable analysis report."""
    sep = "-" * 60

    print(sep)
    print("A/B TEST ANALYSIS REPORT")
    print(sep)

    # SRM
    srm = analysis_result["srm"]
    srm_flag = "⚠ SRM DETECTED" if srm["srm_detected"] else "✓ No SRM"
    print(f"\nSRM Check: {srm_flag}  (p={srm['p_value']:.4f})")
    if srm["warning"]:
        print(f"  {srm['warning']}")

    # Primary
    p = analysis_result["primary"]
    sig = "✓ SIGNIFICANT" if p["significant"] else "✗ not significant"
    print(f"\nPrimary metric ({p['metric']}):")
    print(f"  A={p['p_hat_A']:.4f}  B={p['p_hat_B']:.4f}  "
          f"Δ={p['delta']:+.4f}  ({p['relative_lift']:+.1%} lift)")
    print(f"  z={p['z_statistic']:.3f}  p={p['p_value']:.4f}  {sig}")
    print(f"  95% CI on Δ: [{p['ci_95'][0]:+.4f}, {p['ci_95'][1]:+.4f}]")

    # Secondary
    s = analysis_result["secondary"]
    sig2 = "✓ SIGNIFICANT" if s["significant"] else "✗ not significant"
    print(f"\nSecondary metric ({s['metric']}):")
    print(f"  A={s['p_hat_A']:.4f}  B={s['p_hat_B']:.4f}  "
          f"Δ={s['delta']:+.4f}")
    print(f"  p={s['p_value']:.4f}  {sig2}")

    # Multiple testing
    mt = analysis_result["multiple_testing"]
    print(f"\nMultiple testing (Bonferroni): adjusted α = {mt['adjusted_alpha']:.4f}")

    # Bootstrap
    if analysis_result.get("bootstrap"):
        b = analysis_result["bootstrap"]
        excl = "excludes zero ✓" if b["ci_excludes_zero"] else "includes zero"
        print(f"\nBootstrap CI on Δ: [{b['ci_lower']:+.4f}, {b['ci_upper']:+.4f}]  ({excl})")

    # Decision
    d = analysis_result["decision"]
    print(f"\n{'='*60}")
    print(f"DECISION: {d['verdict']}")
    print(f"Reason:   {d['reason']}")
    print(sep)
