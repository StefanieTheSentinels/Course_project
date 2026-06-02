"""
Plot the surrogate-truth gap: what each optimizer CLAIMED (ML surrogate)
vs what was actually TRUE (oracle evaluation of the found configuration).

Reads cross-validation CSVs produced by optimize.py and aggregates across runs.

Run:
    python plot_gap.py
    python plot_gap.py --pattern "optimization_results_crossval_*.csv"
    python plot_gap.py --out custom/path.png

Output (default):
    graphics/surrogate_gap.png
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Mapping from full method label (as written in CSVs) to short display label.
METHOD_DISPLAY = {
    "DIRECT [ML]":                     "DIRECT",
    "Differential Evolution [ML]":     "DE",
    "Bayesian Optimisation (GP) [ML]": "BO (GP)",
}


def load_crossval(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No cross-validation CSVs matched pattern: {pattern}\n"
            "Did you run optimize.py with --target both?"
        )
    print(f"Aggregating {len(files)} crossval files: {files}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean + std per method across runs."""
    agg = df.groupby("method").agg(
        ml_mean  =("ml_ctr",   "mean"),
        ml_std   =("ml_ctr",   "std"),
        true_mean=("true_ctr", "mean"),
        true_std =("true_ctr", "std"),
        gap_mean =("gap",      "mean"),
        gap_std  =("gap",      "std"),
    )
    # std is NaN for n=1 — replace with 0 so error bars don't break
    return agg.fillna(0.0)


def plot(agg: pd.DataFrame, out_path: str, ceiling: float = 0.39) -> None:
    # Order methods consistently
    order   = [m for m in METHOD_DISPLAY if m in agg.index]
    methods = [METHOD_DISPLAY[m] for m in order]
    ml_m    = agg.loc[order, "ml_mean"].values
    ml_s    = agg.loc[order, "ml_std"].values
    tr_m    = agg.loc[order, "true_mean"].values
    tr_s    = agg.loc[order, "true_std"].values

    plt.rcParams.update({
        "font.family": "serif",
        "font.size":   11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    x         = np.arange(len(methods))
    bar_w     = 0.35
    color_clm = "#c0392b"   # red — what optimiser reported
    color_tru = "#27ae60"   # green — what is really true

    ax.bar(x - bar_w/2, ml_m, bar_w, yerr=ml_s, capsize=4,
           color=color_clm, label="ML claim (what optimiser reported)",
           edgecolor="white", linewidth=0.8)
    ax.bar(x + bar_w/2, tr_m, bar_w, yerr=tr_s, capsize=4,
           color=color_tru, label="Oracle truth (real CTR of that config)",
           edgecolor="white", linewidth=0.8)

    # Annotate gap above each pair
    for i, (clm, tru) in enumerate(zip(ml_m, tr_m)):
        gap = clm - tru
        y   = max(clm, tru) + 0.04
        ax.annotate(
            f"gap = {gap:+.2f}",
            xy=(x[i], y), ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color="#c0392b" if gap > 0.2 else "#555555",
        )
        ax.plot([x[i] - bar_w/2, x[i] + bar_w/2],
                [clm, tru], color="#888888", linestyle=":", linewidth=1.2)

    # Analytical ceiling
    ax.axhline(ceiling, color="#444444", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(methods) - 0.5, ceiling + 0.01,
            f"analytical ceiling (≈{ceiling:.2f})",
            fontsize=9, color="#444444", ha="right", va="bottom", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("CTR")
    ax.set_ylim(0, max(ml_m.max() + ml_s.max(), 0.8) + 0.1)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_title(
        "Surrogate vs Oracle: what optimisers report and what is actually true\n"
        "(configurations found on ML surrogate, re-evaluated on the ground-truth formula)",
        fontsize=11, pad=12,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot surrogate-truth gap from crossval CSVs.")
    p.add_argument("--pattern", type=str,
                   default="runs/run_*/optimization_results_crossval.csv",
                   help="Glob pattern for cross-validation CSV files.")
    p.add_argument("--out", type=str, default="graphics/surrogate_gap.png",
                   help="Output PNG path.")
    p.add_argument("--ceiling", type=float, default=0.39,
                   help="Analytical CTR ceiling to mark on the plot.")
    return p.parse_args()


def main():
    args = parse_args()
    df   = load_crossval(args.pattern)
    agg  = aggregate(df)

    # Console summary
    print("\nAggregated across runs:")
    print(agg.round(4))

    plot(agg, args.out, ceiling=args.ceiling)


if __name__ == "__main__":
    main()