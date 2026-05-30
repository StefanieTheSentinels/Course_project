"""
analysis.py
-----------
Statistics for ExperimentResult: SRM check, z-test, Welch t-test,
bootstrap CI, and a five-way decision rule (SHIP/REJECT/REVIEW/INCONCLUSIVE/INVALID).
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Dict, List, Literal, Optional, Tuple

from .experiment import ExperimentResult


# 1. SRM check
def check_srm(
    result: ExperimentResult,
    expected_split: float = 0.5,
    alpha: float = 0.01,
) -> Dict:
    """
    Chi-square test for sample ratio mismatch.

    H0: observed traffic split equals expected_split.
    Use alpha=0.01 — SRM is a data quality issue requiring high confidence.
    """
    n_A = result.control.n_users
    n_B = result.treatment.n_users
    N   = n_A + n_B

    chi2, p_value = stats.chisquare(
        f_obs=[n_A, n_B],
        f_exp=[N * expected_split, N * (1 - expected_split)],
    )

    return {
        "srm_detected":   bool(p_value < alpha),
        "chi2_statistic": float(chi2),
        "p_value":        float(p_value),
        "observed_split": n_A / N,
        "expected_split": expected_split,
        "warning": (
            "SRM detected — results are unreliable. Investigate assignment/logging."
            if p_value < alpha else None
        ),
    }


# 2. Two-proportion z-test
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

    Test statistic uses pooled SE under H0.
    Confidence interval uses unpooled SE (standard practice).
    """
    p_A   = x_A / n_A
    p_B   = x_B / n_B
    delta = p_B - p_A

    p_pool = (x_A + x_B) / (n_A + n_B)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_A + 1 / n_B))

    if se_pool == 0:
        return {"error": "Standard error is zero — check input data."}

    z = delta / se_pool

    if alternative == "two-sided":
        p_value = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        p_value = stats.norm.sf(z)
    else:
        p_value = stats.norm.cdf(z)

    # CI uses unpooled SE
    se_unpooled = np.sqrt(p_A * (1 - p_A) / n_A + p_B * (1 - p_B) / n_B)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (delta - z_crit * se_unpooled, delta + z_crit * se_unpooled)

    return {
        "p_hat_A":       float(p_A),
        "p_hat_B":       float(p_B),
        "delta":         float(delta),
        "relative_lift": float(delta / p_A) if p_A > 0 else float("nan"),
        "z_statistic":   float(z),
        "p_value":       float(p_value),
        "significant":   bool(p_value < alpha),
        "ci_95":         (float(ci[0]), float(ci[1])),
        "alpha":         alpha,
        "alternative":   alternative,
    }


# 3. Welch t-test (fixed)
def welch_ttest(
    sample_A: np.ndarray,
    sample_B: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Welch t-test on two independent samples.

    Uses Welch–Satterthwaite df and correct SE = sqrt(var_A/n_A + var_B/n_B).
    Appropriate for continuous per-user metrics (e.g., revenue, session length).
    """
    n_A, n_B   = len(sample_A), len(sample_B)
    mean_A     = float(sample_A.mean())
    mean_B     = float(sample_B.mean())
    var_A      = float(sample_A.var(ddof=1))
    var_B      = float(sample_B.var(ddof=1))
    delta      = mean_B - mean_A

    se = np.sqrt(var_A / n_A + var_B / n_B)

    # Welch–Satterthwaite degrees of freedom
    df = (var_A / n_A + var_B / n_B) ** 2 / (
        (var_A / n_A) ** 2 / (n_A - 1) + (var_B / n_B) ** 2 / (n_B - 1)
    )

    t_stat = delta / se if se > 0 else float("nan")

    if alternative := "two-sided":
        p_value = 2 * stats.t.sf(abs(t_stat), df=df)

    z_crit = stats.t.ppf(1 - alpha / 2, df=df)
    ci = (delta - z_crit * se, delta + z_crit * se)

    return {
        "mean_A":        mean_A,
        "mean_B":        mean_B,
        "delta":         delta,
        "relative_lift": delta / mean_A if mean_A != 0 else float("nan"),
        "t_statistic":   float(t_stat),
        "df":            float(df),
        "p_value":       float(p_value),
        "significant":   bool(p_value < alpha),
        "ci_95":         (float(ci[0]), float(ci[1])),
        "alpha":         alpha,
    }


# 4. Bootstrap CI (vectorised)
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
    Vectorised: draws all resamples in two matrix operations.
    """
    rng = np.random.default_rng(seed)
    n_A, n_B = len(sample_A), len(sample_B)

    # Draw all resamples at once: shape (n_bootstrap, n)
    idx_A = rng.integers(0, n_A, size=(n_bootstrap, n_A))
    idx_B = rng.integers(0, n_B, size=(n_bootstrap, n_B))

    deltas = (
        statistic(sample_B[idx_B], axis=1)
        - statistic(sample_A[idx_A], axis=1)
    )

    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    return {
        "point_estimate":   float(statistic(sample_B) - statistic(sample_A)),
        "ci_lower":         lo,
        "ci_upper":         hi,
        "ci_excludes_zero": not (lo <= 0 <= hi),
        "n_bootstrap":      n_bootstrap,
        "alpha":            alpha,
    }


# 5. Full analysis pipeline (primary metric only — no duplicate secondary)
def analyze(
    result: ExperimentResult,
    alpha: float = 0.05,
    primary_metric: Literal["ctr", "conversion"] = "ctr",
    run_bootstrap: bool = False,
    n_bootstrap: int = 2000,
) -> Dict:
    """
    1. SRM check
    2. Primary metric z-test
    3. Bootstrap CI (optional)
    4. Decision rule
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
        primary_label = "CTR"
        boot_sample_A = A.clicks
        boot_sample_B = B.clicks
    else:
        primary = two_proportion_ztest(
            A.n_users, A.n_conversions,
            B.n_users, B.n_conversions,
            alpha=alpha,
        )
        primary_label = "Conversion"
        boot_sample_A = A.conversions
        boot_sample_B = B.conversions

    # 3. Bootstrap
    boot = None
    if run_bootstrap:
        boot = bootstrap_ci(boot_sample_A, boot_sample_B, n_bootstrap=n_bootstrap)

    # 4. Decision rule
    decision = _decision_rule(
        srm_detected=srm["srm_detected"],
        primary_significant=primary["significant"],
        primary_positive=primary.get("delta", 0) > 0,
    )

    return {
        "srm":      srm,
        "primary":  {**primary, "metric": primary_label},
        "bootstrap": boot,
        "decision": decision,
    }


def _decision_rule(
    srm_detected: bool,
    primary_significant: bool,
    primary_positive: bool,
) -> Dict:
    
    if srm_detected:
        verdict = "INVALID"
        reason  = "SRM detected. Investigate randomization and logging before re-running."
    elif primary_significant and primary_positive:
        verdict = "SHIP"
        reason  = "Primary metric improved significantly."
    elif primary_significant and not primary_positive:
        verdict = "REJECT"
        reason  = "Primary metric significantly worse in treatment."
    else:
        verdict = "INCONCLUSIVE"
        reason  = "No significant effect detected. Consider increasing sample size."

    return {"verdict": verdict, "reason": reason}


# Pretty printer (ASCII only)
def print_report(analysis_result: Dict) -> None:
    """Print a human-readable analysis report."""
    sep = "-" * 60
    print(sep)
    print("A/B TEST ANALYSIS REPORT")
    print(sep)

    srm = analysis_result["srm"]
    srm_flag = "[!] SRM DETECTED" if srm["srm_detected"] else "[OK] No SRM"
    print(f"\nSRM Check: {srm_flag}  (p={srm['p_value']:.4f})")
    if srm["warning"]:
        print(f"  {srm['warning']}")

    p   = analysis_result["primary"]
    sig = "[SIG]" if p["significant"] else "[ns] "
    print(f"\nPrimary metric ({p['metric']}):")
    print(f"  A={p['p_hat_A']:.4f}  B={p['p_hat_B']:.4f}  "
          f"delta={p['delta']:+.4f}  ({p['relative_lift']:+.1%} lift)")
    print(f"  z={p['z_statistic']:.3f}  p={p['p_value']:.4f}  {sig}")
    print(f"  95% CI on delta: [{p['ci_95'][0]:+.4f}, {p['ci_95'][1]:+.4f}]")

    if analysis_result.get("bootstrap"):
        b    = analysis_result["bootstrap"]
        excl = "excludes zero [OK]" if b["ci_excludes_zero"] else "includes zero"
        print(f"\nBootstrap CI on delta: [{b['ci_lower']:+.4f}, {b['ci_upper']:+.4f}]  ({excl})")

    d = analysis_result["decision"]
    print(f"\n{'='*60}")
    print(f"DECISION: {d['verdict']}")
    print(f"Reason:   {d['reason']}")
    print(sep)
