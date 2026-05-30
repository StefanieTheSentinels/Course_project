"""
Compare 5 search/optimisation methods:
    1. Random Search
    2. Grid Search
    3. DIRECT
    4. Differential Evolution
    5. Bayesian Optimisation (GP + EI)

Outputs:
    optimization_results.csv
    convergence.png
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from ab_blackbox import FullSyntheticModel
from itertools import product
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.optimize import differential_evolution, direct
from skopt import gp_minimize
from skopt.space import Real

from ab_blackbox import (
    BlackBox,
    TrainedMLModel,
    build_feature_vector,
    FEATURE_NAMES,
)


PARAM_NAMES = [
    "contrast_score",
    "btn_w",
    "btn_h",
    "font_size",
    "text_quality",
    "whitespace_ratio",
    "scroll_to_button",
    "hour",
]

def get_bounds(dataset_path: str = "synthetic_dataset.csv"):
    try:
        df = pd.read_csv(dataset_path)
        return [
            (0.0, 1.0),
            (float(df["btn_w"].quantile(0.05)),    float(df["btn_w"].quantile(0.95))),
            (float(df["btn_h"].quantile(0.05)),    float(df["btn_h"].quantile(0.95))),
            (float(df["font_size"].quantile(0.05)),float(df["font_size"].quantile(0.95))),
            (0.0, 1.0),
            (0.0, 0.5),
            (0.0, 1.0),
            (0.0, 23.0),
        ]
    except FileNotFoundError:
        return [
            (0.0, 1.0), (20.0, 300.0), (20.0, 120.0), (8.0, 48.0),
            (0.0, 1.0), (0.0, 0.5),    (0.0, 1.0),    (0.0, 23.0),
        ]

BOUNDS_CONT = get_bounds()


def vector_to_params(x: np.ndarray, device: str = "desktop") -> Dict:
    contrast_score = float(np.clip(x[0], 0.0, 1.0))
    bg_val   = 255
    text_val = int(255 * (1.0 - contrast_score))
    return {
        "rgb_bg":           (bg_val, bg_val, bg_val),
        "rgb_text":         (text_val, text_val, text_val),
        "btn_w":            float(x[1]),
        "btn_h":            float(x[2]),
        "font_size":        float(x[3]),
        "text_quality":     float(x[4]),
        "whitespace_ratio": float(x[5]),
        "scroll_to_button": float(x[6]),
        "hour":             int(round(x[7])),
        "device":           device,
    }


class Objective:
    """Wraps BlackBox calls, records convergence history. Returns -CTR."""

    def __init__(self, box: BlackBox, device: str = "desktop"):
        self.box     = box
        self.device  = device
        self.history = []
        self.best_so_far = []

    def __call__(self, x) -> float:
        x = np.asarray(x, dtype=float)
        params = vector_to_params(x, device=self.device)
        result = self.box(params)
        ctr    = result.ctr
        self.history.append(ctr)
        running_best = max(self.best_so_far[-1], ctr) if self.best_so_far else ctr
        self.best_so_far.append(running_best)
        return -ctr

    def reset(self):
        self.history     = []
        self.best_so_far = []


def random_search(box: BlackBox, n_calls: int, device: str,
                  seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    obj = Objective(box, device=device)
    best_x   = None
    best_ctr = -1.0
    for _ in range(n_calls):
        x = np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS_CONT])
        ctr = -obj(x)
        if ctr > best_ctr:
            best_ctr = ctr
            best_x   = x
    return {
        "method":      "Random Search",
        "best_ctr":    best_ctr,
        "best_params": vector_to_params(best_x, device=device),
        "calls":       len(obj.history),
        "history":     obj.history,
        "best_so_far": obj.best_so_far,
    }


def grid_search(box: BlackBox, device: str,
                grid_size: int = 3) -> Dict:
    """Coarse grid: grid_size^8 calls. Default 3^8 = 6561."""
    obj = Objective(box, device=device)
    grids = [np.linspace(lo, hi, grid_size) for lo, hi in BOUNDS_CONT]
    best_x   = None
    best_ctr = -1.0
    for combo in product(*grids):
        x = np.array(combo)
        ctr = -obj(x)
        if ctr > best_ctr:
            best_ctr = ctr
            best_x   = x
    return {
        "method":      "Grid Search",
        "best_ctr":    best_ctr,
        "best_params": vector_to_params(best_x, device=device),
        "calls":       len(obj.history),
        "history":     obj.history,
        "best_so_far": obj.best_so_far,
    }


def run_direct(box: BlackBox, device: str,
               max_calls: int = 100) -> Dict:
    obj = Objective(box, device=device)
    res = direct(func=obj, bounds=BOUNDS_CONT, maxfun=max_calls, eps=1e-4)
    return {
        "method":      "DIRECT",
        "best_ctr":    float(-res.fun),
        "best_params": vector_to_params(res.x, device=device),
        "calls":       len(obj.history),
        "history":     obj.history,
        "best_so_far": obj.best_so_far,
    }


def run_de(box: BlackBox, device: str,
           maxiter: int = 30, popsize: int = 10,
           seed: int = 42) -> Dict:
    obj = Objective(box, device=device)
    res = differential_evolution(
        func=obj, bounds=BOUNDS_CONT,
        maxiter=maxiter, popsize=popsize,
        seed=seed, tol=1e-4, polish=False, workers=1,
    )
    return {
        "method":      "Differential Evolution",
        "best_ctr":    float(-res.fun),
        "best_params": vector_to_params(res.x, device=device),
        "calls":       len(obj.history),
        "history":     obj.history,
        "best_so_far": obj.best_so_far,
    }


def run_bo(box: BlackBox, device: str,
           n_calls: int = 50, n_initial: int = 10,
           seed: int = 42) -> Dict:
    obj = Objective(box, device=device)
    space = [Real(lo, hi) for lo, hi in BOUNDS_CONT]
    res = gp_minimize(
        func=obj, dimensions=space,
        n_calls=n_calls, n_initial_points=n_initial,
        acq_func="EI", random_state=seed,
    )
    return {
        "method":      "Bayesian Optimisation (GP)",
        "best_ctr":    float(-res.fun),
        "best_params": vector_to_params(np.array(res.x), device=device),
        "calls":       len(obj.history),
        "history":     obj.history,
        "best_so_far": obj.best_so_far,
    }


def plot_convergence(all_results: List[Dict], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    plt.figure(figsize=(9, 5.5))
    for res in all_results:
        plt.plot(
            range(1, len(res["best_so_far"]) + 1),
            res["best_so_far"],
            label=f"{res['method']} (best CTR = {res['best_ctr']:.4f})",
            linewidth=1.6,
        )
    plt.xlabel("BlackBox calls")
    plt.ylabel("Best CTR found so far")
    plt.title("Convergence: best CTR vs. number of calls")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"Saved convergence plot to {out_path}")


def flatten_params(params: Dict) -> Dict:
    """Convert params dict (including RGB tuples) to a flat dict for CSV."""
    flat = {}
    for k, v in params.items():
        if isinstance(v, tuple):
            for i, c in enumerate(("r", "g", "b")):
                flat[f"param_{k}_{c}"] = v[i] if i < len(v) else None
        else:
            flat[f"param_{k}"] = v
    return flat


def load_pickled_model(path: str):
    """Supports both new dict format and raw Pipeline (backward compat)."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "pipeline" in obj:
        if obj.get("feature_names") != FEATURE_NAMES:
            print("[WARNING] Saved feature_names differ from current FEATURE_NAMES.")
            print(f"  Saved:   {obj.get('feature_names')}")
            print(f"  Current: {FEATURE_NAMES}")
        return obj["pipeline"]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare optimisation methods.")
    parser.add_argument("--model",   type=str, default="best_model.pkl")
    parser.add_argument("--n-users", type=int, default=5000)
    parser.add_argument("--noise",   type=float, default=0.01)
    parser.add_argument("--device",  type=str, default="desktop",
                        choices=["mobile", "desktop"])
    parser.add_argument("--rs-calls",     type=int, default=30)
    parser.add_argument("--grid-size",    type=int, default=3,
                        help="Grid points per dim (8D -> grid_size^8 calls).")
    parser.add_argument("--direct-calls", type=int, default=100)
    parser.add_argument("--de-iter",      type=int, default=30)
    parser.add_argument("--de-pop",       type=int, default=10)
    parser.add_argument("--bo-calls",     type=int, default=50)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--csv-out",  type=str, default="optimization_results.csv")
    parser.add_argument("--plot-out", type=str, default="convergence.png")
    parser.add_argument("--skip-grid", action="store_true",
                        help="Skip Grid Search (3^8 = 6561 calls can be slow).")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Train it first with: python train_models.py")
        sys.exit(1)

    print("=" * 60)
    print("OPTIMISATION METHOD COMPARISON")
    print("=" * 60)

    sklearn_pipeline = load_pickled_model(args.model)
    ml_model = TrainedMLModel(
        classifier=sklearn_pipeline,
        feature_builder=build_feature_vector,
        feature_names=FEATURE_NAMES,
    )
    box = BlackBox(
        model=ml_model, n_users=args.n_users,
        noise_std=args.noise, seed=args.seed,
    )

    print(f"Loaded {args.model}")
    print(f"Device: {args.device}, n_users: {args.n_users}, noise: {args.noise}\n")

    methods = [
        ("Random Search", lambda: random_search(
            box, n_calls=args.rs_calls, device=args.device, seed=args.seed)),
        ("Oracle (ground truth)", lambda: run_bo(
        BlackBox(model=FullSyntheticModel(), n_users=args.n_users,
                 noise_std=args.noise, seed=args.seed),
        device=args.device, n_calls=args.bo_calls, seed=args.seed)),
    ]
    if not args.skip_grid:
        methods.append(("Grid Search", lambda: grid_search(
            box, device=args.device, grid_size=args.grid_size)))
    methods += [
        ("DIRECT", lambda: run_direct(
            box, device=args.device, max_calls=args.direct_calls)),
        ("Differential Evolution", lambda: run_de(
            box, device=args.device, maxiter=args.de_iter,
            popsize=args.de_pop, seed=args.seed)),
        ("Bayesian Optimisation (GP)", lambda: run_bo(
            box, device=args.device, n_calls=args.bo_calls, seed=args.seed)),
    ]

    all_results = []
    BOUNDS_CONT = get_bounds(args.model.replace("best_model.pkl", "synthetic_dataset.csv"))
    for name, fn in methods:
        print(f"--- {name} ---")
        box.reset_call_count()
        t0 = time.time()
        res = fn()
        res["runtime_sec"] = time.time() - t0
        all_results.append(res)
        print(f"  Best CTR: {res['best_ctr']:.4f}")
        print(f"  Calls:    {res['calls']}")
        print(f"  Runtime:  {res['runtime_sec']:.2f}s\n")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<28} {'Calls':>6} {'Best CTR':>10} {'Runtime':>10}")
    print("-" * 58)
    rows = []
    for res in sorted(all_results, key=lambda r: -r["best_ctr"]):
        print(f"{res['method']:<28} {res['calls']:>6} "
              f"{res['best_ctr']:>10.4f} {res['runtime_sec']:>9.2f}s")
        rows.append({
            "method":      res["method"],
            "calls":       res["calls"],
            "best_ctr":    res["best_ctr"],
            "runtime_sec": res["runtime_sec"],
            **flatten_params(res["best_params"]),
        })

    pd.DataFrame(rows).to_csv(args.csv_out, index=False)
    print(f"\nSaved results CSV to {args.csv_out}")

    plot_convergence(all_results, args.plot_out)


if __name__ == "__main__":
    main()
