"""
Aggregate statistics across multiple optimisation runs.

Reads optimization_results_*.csv and optimization_results_*_crossval.csv,
prints mean +/- std per method.
"""

import glob
import pandas as pd


def main():
    # --- main results ---
    files = sorted(glob.glob("optimization_results_[0-9]*.csv"))
    files = [f for f in files if "_crossval" not in f]

    if not files:
        print("No optimization_results_*.csv files found.")
        return

    print(f"Aggregating {len(files)} runs: {files}\n")

    dfs = []
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f)
        df["run"] = i
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    print("=" * 90)
    print(f"AGGREGATED CTR ACROSS {len(files)} RUNS")
    print("=" * 90)

    agg = all_df.groupby("method").agg(
        ctr_mean =("best_ctr", "mean"),
        ctr_std  =("best_ctr", "std"),
        ctr_min  =("best_ctr", "min"),
        ctr_max  =("best_ctr", "max"),
        time_mean=("runtime_sec", "mean"),
        n_runs   =("best_ctr", "count"),
    ).round(4).sort_values("ctr_mean", ascending=False)

    print(f"\n{'Method':<35} {'CTR mean':>10} {'CTR std':>10} "
          f"{'CTR range':>16} {'Time (s)':>10} {'N':>4}")
    print("-" * 90)
    for method, row in agg.iterrows():
        ctr_range = f"[{row['ctr_min']:.3f}, {row['ctr_max']:.3f}]"
        print(f"{method:<35} {row['ctr_mean']:>10.4f} {row['ctr_std']:>10.4f} "
              f"{ctr_range:>16} {row['time_mean']:>10.2f} {int(row['n_runs']):>4}")

    agg.to_csv("aggregated_results.csv")
    print(f"\nSaved to aggregated_results.csv")

    # --- cross-validation gaps ---
    cv_files = sorted(glob.glob("optimization_results_*_crossval.csv"))
    if cv_files:
        print("\n" + "=" * 90)
        print(f"CROSS-VALIDATION GAP ACROSS {len(cv_files)} RUNS")
        print("=" * 90)

        cv_df = pd.concat([pd.read_csv(f) for f in cv_files], ignore_index=True)
        cv_agg = cv_df.groupby("method").agg(
            ml_mean  =("ml_ctr", "mean"),
            true_mean=("true_ctr", "mean"),
            gap_mean =("gap", "mean"),
            gap_std  =("gap", "std"),
        ).round(4).sort_values("gap_mean", ascending=False)

        print(f"\n{'Method':<35} {'ML mean':>10} {'True mean':>10} "
              f"{'Gap mean':>10} {'Gap std':>10}")
        print("-" * 85)
        for method, row in cv_agg.iterrows():
            print(f"{method:<35} {row['ml_mean']:>10.4f} {row['true_mean']:>10.4f} "
                  f"{row['gap_mean']:>+10.4f} {row['gap_std']:>10.4f}")

        cv_agg.to_csv("aggregated_crossval.csv")
        print(f"\nSaved to aggregated_crossval.csv")


if __name__ == "__main__":
    main()