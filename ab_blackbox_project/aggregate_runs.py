"""
Aggregate statistics across multiple optimisation runs.

Reads from runs/run_*/:
    optimization_results_oracle.csv   -> aggregated_oracle.csv
    optimization_results_ml.csv       -> aggregated_ml.csv
    optimization_results_crossval.csv -> aggregated_crossval.csv
"""

import glob
import pandas as pd


def aggregate_main_block(pattern: str, label: str, out_csv: str) -> None:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[{label}] no files matched {pattern}\n")
        return

    print(f"[{label}] aggregating {len(files)} files: {files}")
    dfs = []
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f)
        df["run"] = i
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    agg = all_df.groupby("method").agg(
        ctr_mean =("best_ctr",    "mean"),
        ctr_std  =("best_ctr",    "std"),
        ctr_min  =("best_ctr",    "min"),
        ctr_max  =("best_ctr",    "max"),
        time_mean=("runtime_sec", "mean"),
        n_runs   =("best_ctr",    "count"),
    ).round(4).sort_values("ctr_mean", ascending=False)

    print(f"\n=== AGGREGATED CTR [{label}] across {len(files)} runs ===")
    print(f"{'Method':<35} {'CTR mean':>10} {'CTR std':>10} "
          f"{'CTR range':>16} {'Time (s)':>10} {'N':>4}")
    print("-" * 90)
    for method, row in agg.iterrows():
        ctr_range = f"[{row['ctr_min']:.3f}, {row['ctr_max']:.3f}]"
        print(f"{method:<35} {row['ctr_mean']:>10.4f} {row['ctr_std']:>10.4f} "
              f"{ctr_range:>16} {row['time_mean']:>10.2f} {int(row['n_runs']):>4}")
    agg.to_csv(out_csv)
    print(f"Saved {out_csv}\n")


def aggregate_crossval(pattern: str, out_csv: str) -> None:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[CROSSVAL] no files matched {pattern}")
        return

    print(f"[CROSSVAL] aggregating {len(files)} files: {files}")
    cv_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cv_agg = cv_df.groupby("method").agg(
        ml_mean  =("ml_ctr",   "mean"),
        true_mean=("true_ctr", "mean"),
        gap_mean =("gap",      "mean"),
        gap_std  =("gap",      "std"),
    ).round(4).sort_values("gap_mean", ascending=False)

    print(f"\n=== CROSS-VALIDATION GAP across {len(files)} runs ===")
    print(f"{'Method':<35} {'ML mean':>10} {'True mean':>10} "
          f"{'Gap mean':>10} {'Gap std':>10}")
    print("-" * 85)
    for method, row in cv_agg.iterrows():
        print(f"{method:<35} {row['ml_mean']:>10.4f} {row['true_mean']:>10.4f} "
              f"{row['gap_mean']:>+10.4f} {row['gap_std']:>10.4f}")
    cv_agg.to_csv(out_csv)
    print(f"Saved {out_csv}\n")


def main():
    aggregate_main_block("runs/run_*/optimization_results_oracle.csv",
                         "ORACLE", "aggregated_oracle.csv")
    aggregate_main_block("runs/run_*/optimization_results_ml.csv",
                         "ML",     "aggregated_ml.csv")
    aggregate_crossval  ("runs/run_*/optimization_results_crossval.csv",
                                   "aggregated_crossval.csv")


if __name__ == "__main__":
    main()