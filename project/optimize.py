"""
optimize.py
-----------
Compares three black-box search strategies (grid, random, greedy sweep).
Results are averaged over N_RUNS=20 seeds to reduce noise.

Usage:
    python optimize.py --mode demo   # synthetic model
    python optimize.py --mode real   # calibrated from real data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from itertools import product
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ab_blackbox import BlackBox
from ab_blackbox.model import LinearButtonModel

GRID         = [0.0, 0.25, 0.5, 0.75, 1.0]
PARAM_NAMES  = ["color", "size", "text"]
BUDGET       = 30     # random search call budget
N_RUNS       = 20     # how many seeds to average over


# Search strategies
def grid_search(box: BlackBox) -> dict:
    """Exhaustive grid search over all 5^3 = 125 combinations."""
    best_ctr    = -1.0
    best_params = {}
    for color_val, size_val, text_val in product(GRID, GRID, GRID):
        params = {"color": color_val, "size": size_val, "text": text_val}
        ctr    = box(params).ctr
        if ctr > best_ctr:
            best_ctr    = ctr
            best_params = dict(params)
    return {"best_ctr": best_ctr, "best_params": best_params, "calls": box.call_count}


def random_search(box: BlackBox, budget: int, rng: np.random.Generator) -> dict:
    """Random search with a fixed call budget."""
    best_ctr    = -1.0
    best_params = {}
    for _ in range(budget):
        params = {p: float(rng.choice(GRID)) for p in PARAM_NAMES}
        ctr    = box(params).ctr
        if ctr > best_ctr:
            best_ctr    = ctr
            best_params = dict(params)
    return {"best_ctr": best_ctr, "best_params": best_params, "calls": box.call_count}


def greedy_sweep(box: BlackBox) -> dict:
    """Greedy 1-D sweep: optimise one parameter at a time."""
    current = {"color": 0.0, "size": 0.0, "text": 0.0}
    for param in PARAM_NAMES:
        best_val = current[param]
        best_ctr = box(current).ctr
        for val in GRID[1:]:
            ctr = box({**current, param: val}).ctr
            if ctr > best_ctr:
                best_ctr = ctr
                best_val = val
        current[param] = best_val
    return {"best_ctr": best_ctr, "best_params": dict(current), "calls": box.call_count}


# Multi-run averaging
def run_strategy(strategy_fn, model, n_users, noise_std, seeds, **kwargs):
    """Run a search strategy over multiple seeds and return mean ± std CTR."""
    ctrs   = []
    params_list = []
    for seed in seeds:
        box = BlackBox(model=model, n_users=n_users, noise_std=noise_std, seed=seed)
        box.reset_call_count()
        if strategy_fn == random_search:
            rng    = np.random.default_rng(seed)
            result = strategy_fn(box, budget=BUDGET, rng=rng)
        else:
            result = strategy_fn(box, **kwargs)
        ctrs.append(result["best_ctr"])
        params_list.append(result["best_params"])
    return {
        "mean_ctr":   float(np.mean(ctrs)),
        "std_ctr":    float(np.std(ctrs)),
        "calls":      result["calls"],
        "best_params": params_list[-1],   # last run's config (typically same)
    }


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["demo", "real"], default="real",
        help="'demo' uses synthetic model; 'real' calibrates from dataset"
    )
    args = parser.parse_args()

    # Build model
    if args.mode == "real":
        print("=" * 60)
        print("MODE: REAL DATA CALIBRATION")
        print("=" * 60)

        df          = pd.read_csv("ab_test_results_aggregated_views_clicks_2.csv")
        ctr_control = (df[df["group"] == "control"]["clicks"] >= 1).mean()
        ctr_test    = (df[df["group"] == "test"]["clicks"] >= 1).mean()
        w_observed  = ctr_test - ctr_control

        print(f"  Control CTR:     {ctr_control:.4f}")
        print(f"  Test CTR:        {ctr_test:.4f}")
        print(f"  Observed effect: {w_observed:+.4f}  (w_color from data)")
        print(f"\n  NOTE: w_size and w_text are not observed in the dataset.")
        print(f"  They are set equal to w_color as a baseline assumption with")
        print(f"  no empirical basis. Optimisation results are ILLUSTRATIVE only.")
        print(f"  Follow-up experiments varying size and text independently")
        print(f"  would be needed to estimate these weights from data.")

        model = LinearButtonModel(
            base_click=ctr_control,
            w_color=w_observed,
            w_size=w_observed,
            w_text=w_observed,
            base_conv=0.0,
            w_conv_color=0.0,
            w_conv_size=0.0,
            w_conv_text=0.0,
        )
        reference_ctr = ctr_test
        reference_label = "real experiment CTR"

    else:
        print("=" * 60)
        print("MODE: SYNTHETIC DEMO")
        print("=" * 60)
        model = LinearButtonModel(
            base_click=0.1470, w_color=0.0147, w_size=0.0080, w_text=0.0120,
            base_conv=0.0, w_conv_color=0.0, w_conv_size=0.0, w_conv_text=0.0,
        )
        reference_ctr   = model._eval(model.click_expr, {"color": 1.0, "size": 0.0, "text": 0.0})
        reference_label = "color=1 baseline CTR"

    # Theoretical optimum via SymPy (zero calls, no noise)
    theo = model.theoretical_effect(
        {"color": 0.0, "size": 0.0, "text": 0.0},
        {"color": 1.0, "size": 1.0, "text": 1.0},
    )
    theo_ctr = theo["p_click_B"]
    print(f"\nTheoretical optimum (SymPy, all params=1): CTR = {theo_ctr:.4f}")

    # Multi-run optimisation
    n_users   = 5000
    noise_std = 0.003
    seeds     = list(range(N_RUNS))

    print(f"\nRunning each strategy {N_RUNS} times (seeds 0..{N_RUNS-1})")
    print(f"n_users={n_users}, noise_std={noise_std}")

    print("\n" + "=" * 60)
    print("GRID SEARCH (125 calls)")
    print("=" * 60)
    gs = run_strategy(grid_search, model, n_users, noise_std, seeds)
    print(f"Best config (last run): {gs['best_params']}")
    print(f"Mean CTR: {gs['mean_ctr']:.4f} +/- {gs['std_ctr']:.4f}")
    print(f"Theoretical CTR at that config: {theo_ctr:.4f}")

    print("\n" + "=" * 60)
    print(f"RANDOM SEARCH ({BUDGET} calls)")
    print("=" * 60)
    rs = run_strategy(random_search, model, n_users, noise_std, seeds)
    print(f"Best config (last run): {rs['best_params']}")
    print(f"Mean CTR: {rs['mean_ctr']:.4f} +/- {rs['std_ctr']:.4f}")

    print("\n" + "=" * 60)
    print("GREEDY 1-D SWEEP (~15 calls)")
    print("=" * 60)
    gr = run_strategy(greedy_sweep, model, n_users, noise_std, seeds)
    print(f"Best config (last run): {gr['best_params']}")
    print(f"Mean CTR: {gr['mean_ctr']:.4f} +/- {gr['std_ctr']:.4f}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nNote: 'Mean CTR' is the average of best-found CTR over {N_RUNS} seeds.")
    print(f"All three methods find config (1,1,1); CTR differences reflect sampling noise.")
    print(f"The theoretical CTR at (1,1,1) is {theo_ctr:.4f} (SymPy, no noise).\n")

    print(f"{'Method':<25} {'Calls':>6}  {'Mean CTR':>9}  {'Std':>7}  "
          f"{'Lift vs ' + reference_label[:12]:>20}")
    print("-" * 75)
    print(f"{'Reference (' + reference_label[:10] + ')':<25} {'—':>6}  "
          f"{reference_ctr:>9.4f}  {'—':>7}  {'(reference)':>20}")
    for name, res in [("Grid search", gs), ("Random search", rs), ("Greedy sweep", gr)]:
        lift = (res['mean_ctr'] - reference_ctr) / reference_ctr
        print(f"{name:<25} {res['calls']:>6}  {res['mean_ctr']:>9.4f}  "
              f"{res['std_ctr']:>7.4f}  {lift:>+19.1%}")
    print(f"{'SymPy theoretical':<25} {'0':>6}  {theo_ctr:>9.4f}  {'—':>7}  "
          f"{(theo_ctr - reference_ctr) / reference_ctr:>+19.1%}")


if __name__ == "__main__":
    main()
