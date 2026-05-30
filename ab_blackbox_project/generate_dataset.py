"""
CLI wrapper around generate_full_synthetic_dataset.

Usage:
    python generate_dataset.py
    python generate_dataset.py --n 100000 --seed 7 --noise 0.03
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ab_blackbox import (
    generate_full_synthetic_dataset,
    synthetic_dataset_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic dataset.")
    parser.add_argument("--n",     type=int,   default=50_000)
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--noise", type=float, default=0.02,
                        help="Gaussian noise std on p_click (clipped to [0,1]).")
    parser.add_argument("--out",   type=str,   default="synthetic_dataset.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("SYNTHETIC DATASET GENERATION")
    print("=" * 60)
    print(f"  n          : {args.n}")
    print(f"  seed       : {args.seed}")
    print(f"  extra noise: {args.noise}")
    print(f"  output     : {args.out}\n")

    df = generate_full_synthetic_dataset(
        n=args.n, seed=args.seed, extra_noise=args.noise
    )

    print(f"Generated {len(df)} rows.")

    summary = synthetic_dataset_summary(df)
    print("\nDataset summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")

    df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    print("\nSanity checks:")
    ctr = float(df["click"].mean())
    print(f"  Overall CTR ({ctr:.4f}) should be in [0.02, 0.25]: "
          f"{'OK' if 0.02 <= ctr <= 0.25 else 'WARNING'}")
    diff_p = abs(summary['mean_p_true'] - ctr)
    print(f"  |mean_p_true - CTR| = {diff_p:.4f}, should be < 0.02: "
          f"{'OK' if diff_p <= 0.02 else 'WARNING'}")
    print(f"  CTR_desktop ({summary['ctr_desktop']:.4f}) >= "
          f"CTR_mobile ({summary['ctr_mobile']:.4f}): "
          f"{'OK' if summary['ctr_desktop'] >= summary['ctr_mobile'] else 'WARNING'}")


if __name__ == "__main__":
    main()
