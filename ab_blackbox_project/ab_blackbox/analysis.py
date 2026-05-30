"""
Statistical analysis: SRM, z-test, Welch t-test, bootstrap CI, decision rule.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Dict, Literal

from .experiment import ExperimentResult


def check_srm(result: ExperimentResult,
              expected_split: float = 0.5,
              alpha: float = 0.01) -> Dict:
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
        "observed_split": n_A / N if N > 0 else float("nan"),
        "expected_split": expected_split,
        "warning": (
            "SRM detected — results are unreliable."
            if p_value < alpha else None
        ),
    }


def two_proportion_ztest(n_A: int, x_A: int,
                         n_B: int, x_B: int,
                         alpha: float = 0.05,
                         alternative: Literal["two-sided", "greater", "less"] = "two-sided"
                         ) -> Dict:
    p_A   = x_A / n_A
    p_B   = x_B / n_B
    delta = p_B - p_A

    p_pool  = (x_A + x_B) / (n_A + n_B)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_A + 1 / n_B))

    if se_pool == 0:
        return {"error": "Standard error is zero."}

    z = delta / se_pool

    if alternative == "two-sided":
        p_value = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        p_value = stats.norm.sf(z)
    else:
        p_value = stats.norm.cdf(z)

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




def bootstrap_ci(sample_A: np.ndarray,
                 sample_B: np.ndarray,
                 statistic=np.mean,
                 n_bootstrap: int = 2000,
                 alpha: float = 0.05,
                 seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    n_A, n_B = len(sample_A), len(sample_B)

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


def analyze(result: ExperimentResult,
            alpha: float = 0.05,
            primary_metric: Literal["ctr", "conversion"] = "ctr",
            run_bootstrap: bool = False,
            n_bootstrap: int = 2000) -> Dict:
    A = result.control
    B = result.treatment

    srm = check_srm(result, alpha=0.01)

    if primary_metric == "ctr":
        primary = two_proportion_ztest(A.n_users, A.n_clicks,
                                       B.n_users, B.n_clicks, alpha=alpha)
        primary_label = "CTR"
        boot_sample_A = A.clicks
        boot_sample_B = B.clicks
    else:
        primary = two_proportion_ztest(A.n_users, A.n_conversions,
                                       B.n_users, B.n_conversions, alpha=alpha)
        primary_label = "Conversion"
        boot_sample_A = A.conversions
        boot_sample_B = B.conversions

    boot = None
    if run_bootstrap:
        boot = bootstrap_ci(boot_sample_A, boot_sample_B, n_bootstrap=n_bootstrap)

    decision = _decision_rule(
        srm_detected=srm["srm_detected"],
        primary_significant=primary.get("significant", False),
        primary_positive=primary.get("delta", 0) > 0,
    )

    return {
        "srm":       srm,
        "primary":   {**primary, "metric": primary_label},
        "bootstrap": boot,
        "decision":  decision,
    }


def _decision_rule(srm_detected: bool,
                   primary_significant: bool,
                   primary_positive: bool) -> Dict:
    if srm_detected:
        return {"verdict": "INVALID",
                "reason":  "SRM detected. Investigate randomization and logging."}
    if primary_significant and primary_positive:
        return {"verdict": "SHIP",
                "reason":  "Primary metric improved significantly."}
    if primary_significant and not primary_positive:
        return {"verdict": "REJECT",
                "reason":  "Primary metric significantly worse in treatment."}
    return {"verdict": "INCONCLUSIVE",
            "reason":  "No significant effect detected."}


def print_report(analysis_result: Dict) -> None:
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
    if "error" in p:
        print(f"\nPrimary metric error: {p['error']}")
        return
    sig = "[SIG]" if p["significant"] else "[ns] "
    print(f"\nPrimary metric ({p['metric']}):")
    print(f"  A={p['p_hat_A']:.4f}  B={p['p_hat_B']:.4f}  "
          f"delta={p['delta']:+.4f}  ({p['relative_lift']:+.1%} lift)")
    print(f"  z={p['z_statistic']:.3f}  p={p['p_value']:.4f}  {sig}")
    print(f"  95% CI on delta: [{p['ci_95'][0]:+.4f}, {p['ci_95'][1]:+.4f}]")

    if analysis_result.get("bootstrap"):
        b    = analysis_result["bootstrap"]
        excl = "excludes zero [OK]" if b["ci_excludes_zero"] else "includes zero"
        print(f"\nBootstrap CI: [{b['ci_lower']:+.4f}, {b['ci_upper']:+.4f}]  ({excl})")

    d = analysis_result["decision"]
    print(f"\n{'='*60}")
    print(f"DECISION: {d['verdict']}")
    print(f"Reason:   {d['reason']}")
    print(sep)
