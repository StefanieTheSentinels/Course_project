"""
Compare 3 optimisation methods:
    1. DIRECT
    2. Differential Evolution
    3. Bayesian Optimisation (GP + EI)

CLI:
    --target {oracle, ml, both}   which backend(s) to optimise against
    --no-gif                      skip convergence animation
    --csv-oracle / --csv-ml / --csv-crossval  per-backend CSV outputs
    --gif-oracle / --gif-ml       per-backend GIF outputs
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.optimize import differential_evolution, direct
from skopt import gp_minimize
from skopt.space import Real

from ab_blackbox import (
    BlackBox,
    TrainedMLModel,
    FullSyntheticModel,
    build_feature_vector,
    FEATURE_NAMES,
)


# ===========================================================================
# Bounds & vector <-> params mapping
# ===========================================================================

def get_bounds(dataset_path: str = "synthetic_dataset.csv"):
    try:
        df = pd.read_csv(dataset_path)
        return [
            (0.0, 255.0), (0.0, 255.0), (0.0, 255.0),
            (0.0, 255.0), (0.0, 255.0), (0.0, 255.0),
            (float(df["btn_w"].quantile(0.05)),     float(df["btn_w"].quantile(0.95))),
            (float(df["btn_h"].quantile(0.05)),     float(df["btn_h"].quantile(0.95))),
            (float(df["font_size"].quantile(0.05)), float(df["font_size"].quantile(0.95))),
            (0.0, 1.0),
            (0.0, 0.5),
            (0.0, 1.0),
        ]
    except FileNotFoundError:
        return [
            (0.0, 255.0), (0.0, 255.0), (0.0, 255.0),
            (0.0, 255.0), (0.0, 255.0), (0.0, 255.0),
            (20.0, 300.0), (20.0, 120.0), (8.0, 48.0),
            (0.0, 1.0), (0.0, 0.5), (0.0, 1.0),
        ]


BOUNDS_CONT = get_bounds()


def vector_to_params(x: np.ndarray, device: str = "desktop") -> Dict:
    return {
        "rgb_bg":           (int(np.clip(x[0], 0, 255)),
                             int(np.clip(x[1], 0, 255)),
                             int(np.clip(x[2], 0, 255))),
        "rgb_text":         (int(np.clip(x[3], 0, 255)),
                             int(np.clip(x[4], 0, 255)),
                             int(np.clip(x[5], 0, 255))),
        "btn_w":            float(x[6]),
        "btn_h":            float(x[7]),
        "font_size":        float(x[8]),
        "text_quality":     float(x[9]),
        "whitespace_ratio": float(x[10]),
        "scroll_to_button": float(x[11]),
        "hour":             12,
        "device":           device,
    }


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


# ===========================================================================
# Objective wrapper
# ===========================================================================

class Objective:
    """Wraps BlackBox calls, records convergence history. Returns -CTR."""

    def __init__(self, box: BlackBox, device: str = "desktop"):
        self.box            = box
        self.device         = device
        self.history        = []
        self.best_so_far    = []
        self.params_history = []

    def __call__(self, x) -> float:
        x      = np.asarray(x, dtype=float)
        params = vector_to_params(x, device=self.device)
        ctr    = self.box(params).ctr
        self.history.append(ctr)
        self.params_history.append(params)
        running_best = max(self.best_so_far[-1], ctr) if self.best_so_far else ctr
        self.best_so_far.append(running_best)
        return -ctr

    def reset(self):
        self.history        = []
        self.best_so_far    = []
        self.params_history = []


# ===========================================================================
# Three optimisers
# ===========================================================================

def run_direct(box: BlackBox, device: str, max_calls: int = 100) -> Dict:
    obj = Objective(box, device=device)
    res = direct(func=obj, bounds=BOUNDS_CONT, maxfun=max_calls, eps=1e-4)
    return {
        "method":         "DIRECT",
        "best_ctr":       float(-res.fun),
        "best_params":    vector_to_params(res.x, device=device),
        "calls":          len(obj.history),
        "history":        obj.history,
        "best_so_far":    obj.best_so_far,
        "params_history": obj.params_history,
    }


def run_de(box: BlackBox, device: str,
           maxiter: int = 30, popsize: int = 10, seed: int = 42) -> Dict:
    obj = Objective(box, device=device)
    res = differential_evolution(
        func=obj, bounds=BOUNDS_CONT,
        maxiter=maxiter, popsize=popsize,
        seed=seed, tol=1e-4, polish=False, workers=1,
    )
    return {
        "method":         "Differential Evolution",
        "best_ctr":       float(-res.fun),
        "best_params":    vector_to_params(res.x, device=device),
        "calls":          len(obj.history),
        "history":        obj.history,
        "best_so_far":    obj.best_so_far,
        "params_history": obj.params_history,
    }


def run_bo(box: BlackBox, device: str,
           n_calls: int = 50, n_initial: int = 10, seed: int = 42) -> Dict:
    obj   = Objective(box, device=device)
    space = [Real(lo, hi) for lo, hi in BOUNDS_CONT]
    res   = gp_minimize(
        func=obj, dimensions=space,
        n_calls=n_calls, n_initial_points=n_initial,
        acq_func="EI", random_state=seed,
    )
    return {
        "method":         "Bayesian Optimisation (GP)",
        "best_ctr":       float(-res.fun),
        "best_params":    vector_to_params(np.array(res.x), device=device),
        "calls":          len(obj.history),
        "history":        obj.history,
        "best_so_far":    obj.best_so_far,
        "params_history": obj.params_history,
    }


# ===========================================================================
# Convergence GIF
# ===========================================================================

def plot_convergence(all_results: List[Dict], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import matplotlib.gridspec as gridspec
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    COLORS = {
        "DIRECT":                    "#2980b9",
        "Differential Evolution":    "#27ae60",
        "Bayesian Optimisation (GP)": "#8e44ad",
    }
    STYLES = {
        "DIRECT":                    "--",
        "Differential Evolution":    "-.",
        "Bayesian Optimisation (GP)": ":",
    }

    max_calls = max(len(r["best_so_far"]) for r in all_results)
    histories = {res["method"]: res["best_so_far"] for res in all_results}
    all_vals  = [v for h in histories.values() for v in h]

    def get_params_at(result, call_idx):
        history = result["history"]
        actual  = min(call_idx, len(history))
        if actual == 0:
            return result["params_history"][0]
        best_idx = int(np.argmax(history[:actual]))
        return result["params_history"][best_idx]

    def get_ctr_at(result, call_idx):
        bsf = result["best_so_far"]
        idx = min(call_idx, len(bsf)) - 1
        return bsf[max(idx, 0)]

    def draw_button(ax, params, ctr, method_name):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("#f4f4f4")

        bg_hex  = "#{:02x}{:02x}{:02x}".format(*[int(c) for c in params["rgb_bg"]])
        txt_hex = "#{:02x}{:02x}{:02x}".format(*[int(c) for c in params["rgb_text"]])

        btn_w = min(0.18 + (params["btn_w"] / 300) * 0.60, 0.82)
        btn_h = min(0.06 + (params["btn_h"] / 120) * 0.22, 0.30)
        bx    = 0.5 - btn_w / 2
        by    = 0.5 - btn_h / 2

        ax.add_patch(FancyBboxPatch(
            (bx + 0.008, by - 0.012), btn_w, btn_h,
            boxstyle="round,pad=0.01", linewidth=0, facecolor="#cccccc", zorder=1,
        ))
        ax.add_patch(FancyBboxPatch(
            (bx, by), btn_w, btn_h,
            boxstyle="round,pad=0.01", linewidth=1.2,
            edgecolor="#aaaaaa", facecolor=bg_hex, zorder=2,
        ))

        tq    = params["text_quality"]
        label = "Buy Now" if tq >= 0.5 else "buy now!!!"
        font_scale = max(6, min(params["font_size"] / 16.0 * 9, 16))
        ax.text(0.5, 0.5, label,
                ha="center", va="center", fontsize=font_scale, color=txt_hex,
                alpha=0.3 + 0.7 * tq,
                fontweight="bold" if tq >= 0.6 else "normal", zorder=3)

        ctr_color = "#2e7d32" if ctr >= 0.15 else "#f57c00" if ctr >= 0.06 else "#c62828"
        ax.text(0.5, 0.06, f"CTR = {ctr:.3f}",
                ha="center", va="bottom", fontsize=8,
                color=ctr_color, fontweight="bold")
        ax.text(0.5, 0.94, method_name,
                ha="center", va="top", fontsize=7,
                color=COLORS.get(method_name, "#555555"), fontweight="bold")

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Optimisation Method Comparison", fontsize=13, fontweight="bold")

    gs   = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.4)
    ax1  = fig.add_subplot(gs[:, 0])
    ax1.set_xlim(1, max_calls)
    ax1.set_ylim(min(all_vals) * 0.95, max(all_vals) * 1.05)
    ax1.set_xlabel("BlackBox calls", fontsize=10)
    ax1.set_ylabel("Best CTR found so far", fontsize=10)
    ax1.set_title("Convergence", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    lines = {}
    for res in all_results:
        name = res["method"]
        line, = ax1.plot(
            [], [], label=name,
            color=COLORS.get(name.split(" [")[0], "#555555"),
            linestyle=STYLES.get(name.split(" [")[0], "-"),
            linewidth=2,
        )
        lines[name] = line
    ax1.legend(fontsize=8)

    btn_positions = [gs[0, 1], gs[0, 2], gs[1, 1], gs[1, 2]]
    btn_axes = [fig.add_subplot(btn_positions[i])
                for i in range(min(4, len(all_results)))]

    N_FRAMES = 240
    step     = max(1, max_calls // N_FRAMES)

    def update(frame):
        call_idx = min((frame + 1) * step, max_calls)
        for name, line in lines.items():
            actual_len = len(histories[name])
            xs = list(range(1, min(call_idx, actual_len) + 1))
            line.set_data(xs, histories[name][:len(xs)])
        for i, res in enumerate(all_results[:4]):
            draw_button(btn_axes[i],
                        get_params_at(res, call_idx),
                        get_ctr_at(res, call_idx),
                        res["method"])
        return list(lines.values())

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=50, blit=False)

    gif_path = out_path if out_path.endswith(".gif") else out_path.replace(".png", ".gif")
    anim.save(gif_path, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"Saved convergence animation to {gif_path}")


# ===========================================================================
# Model loading
# ===========================================================================

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


# ===========================================================================
# BlackBox factories
# ===========================================================================

def build_oracle_box(args) -> BlackBox:
    return BlackBox(
        model=FullSyntheticModel(),
        n_users=args.n_users,
        noise_std=args.noise,
        seed=args.seed,
    )


def build_ml_box(args, ml_model) -> BlackBox:
    return BlackBox(
        model=ml_model,
        n_users=args.n_users,
        noise_std=args.noise,
        seed=args.seed,
    )


def build_oracle_clean_box(args) -> BlackBox:
    """Noise-free oracle for cross-validating ML-found configs."""
    return BlackBox(
        model=FullSyntheticModel(),
        n_users=args.n_users,
        noise_std=0.0,
        seed=args.seed,
    )


# ===========================================================================
# Method runner
# ===========================================================================

def _budget_split(budget: int) -> Dict:
    de_popsize = 7
    return {
        "direct_calls": budget,
        "de_iter":      max(1, budget // (de_popsize * 12)),
        "de_popsize":   de_popsize,
        "bo_calls":     budget,
    }


def run_all_methods(box_factory: Callable[[], BlackBox],
                    args, label: str) -> List[Dict]:
    """Run DIRECT / DE / BO against a fresh box per method; tag with label."""
    b = _budget_split(args.budget)
    methods = [
        ("DIRECT", lambda: run_direct(
            box_factory(), device=args.device,
            max_calls=b["direct_calls"])),
        ("Differential Evolution", lambda: run_de(
            box_factory(), device=args.device,
            maxiter=b["de_iter"], popsize=b["de_popsize"], seed=args.seed)),
        ("Bayesian Optimisation (GP)", lambda: run_bo(
            box_factory(), device=args.device,
            n_calls=b["bo_calls"], seed=args.seed)),
    ]

    print("=" * 60)
    print(f"BLACKBOX: {label}")
    print("=" * 60)

    results = []
    for name, fn in methods:
        print(f"--- {name} ---")
        t0  = time.time()
        res = fn()
        res["runtime_sec"] = time.time() - t0
        res["blackbox"]    = label
        res["method"]      = f"{name} [{label}]"
        results.append(res)
        print(f"  Best CTR: {res['best_ctr']:.4f}")
        print(f"  Calls:    {res['calls']}")
        print(f"  Runtime:  {res['runtime_sec']:.2f}s\n")
    return results


# ===========================================================================
# Output: print + CSV
# ===========================================================================

def print_summary(results: List[Dict], label: str) -> None:
    print("=" * 60)
    print(f"SUMMARY [{label}]")
    print("=" * 60)
    print(f"{'Method':<30} {'Calls':>6} {'Best CTR':>10} {'Runtime':>10}")
    print("-" * 60)
    for res in sorted(results, key=lambda r: -r["best_ctr"]):
        method_short = res["method"].split(" [")[0]
        print(f"{method_short:<30} {res['calls']:>6} "
              f"{res['best_ctr']:>10.4f} {res['runtime_sec']:>9.2f}s")
    print()


def write_results_csv(results: List[Dict], path: str) -> None:
    rows = []
    for res in sorted(results, key=lambda r: -r["best_ctr"]):
        rows.append({
            "method":      res["method"],
            "calls":       res["calls"],
            "best_ctr":    res["best_ctr"],
            "runtime_sec": res["runtime_sec"],
            **flatten_params(res["best_params"]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved results CSV to {path}")


# ===========================================================================
# Cross-validation: ML-found configs re-evaluated on the oracle
# ===========================================================================

def print_trajectory(ml_results: List[Dict], args) -> None:
    """Start / mid / final gap on ML trajectory vs oracle."""
    oracle_box = build_oracle_clean_box(args)

    print("=" * 60)
    print("TRAJECTORY: start / mid / final on ML vs ORACLE")
    print("=" * 60)
    print(f"{'Method':<35} {'Phase':<8} {'BlackBox CTR':>14} "
          f"{'Oracle CTR':>12} {'Gap':>10}")
    print("-" * 90)

    for res in ml_results:
        history        = res["history"]
        params_history = res["params_history"]
        n              = len(history)
        if n == 0:
            continue
        points = {"start": 0, "mid": n // 2,
                  "final": int(np.argmax(history))}
        for phase, idx in points.items():
            params   = params_history[idx]
            bb_ctr   = history[idx]
            true_ctr = oracle_box(params).ctr
            print(f"{res['method']:<35} {phase:<8} "
                  f"{bb_ctr:>14.4f} {true_ctr:>12.4f} "
                  f"{bb_ctr - true_ctr:>+10.4f}")
        print()


def cross_validate(ml_results: List[Dict], args) -> List[Dict]:
    """Re-evaluate every ML-found best_params on the noise-free oracle."""
    oracle_box = build_oracle_clean_box(args)

    print("=" * 60)
    print("CROSS-VALIDATION: ML-optimum evaluated on ORACLE")
    print("=" * 60)
    print(f"{'Method':<35} {'CTR (ML claims)':>18} "
          f"{'CTR (oracle truth)':>20} {'Gap':>10}")
    print("-" * 90)

    rows = []
    for res in ml_results:
        ml_ctr   = res["best_ctr"]
        true_ctr = oracle_box(res["best_params"]).ctr
        gap      = ml_ctr - true_ctr
        print(f"{res['method']:<35} {ml_ctr:>18.4f} "
              f"{true_ctr:>20.4f} {gap:>+10.4f}")
        rows.append({
            "method":   res["method"],
            "ml_ctr":   ml_ctr,
            "true_ctr": true_ctr,
            "gap":      gap,
        })
    print()
    return rows


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare optimisation methods.")
    p.add_argument("--model",    type=str,   default="best_model.pkl")
    p.add_argument("--budget",   type=int,   default=200)
    p.add_argument("--n-users",  type=int,   default=10_000)
    p.add_argument("--noise",    type=float, default=0.01)
    p.add_argument("--device",   type=str,   default="desktop",
                   choices=["mobile", "desktop"])
    p.add_argument("--seed",     type=int,   default=42)

    # what to run
    p.add_argument("--target", type=str, default="both",
                   choices=["oracle", "ml", "both"],
                   help="Which simulator backend(s) to optimise against.")

    # outputs (split per backend)
    p.add_argument("--csv-oracle",   type=str,
                   default="optimization_results_oracle.csv")
    p.add_argument("--csv-ml",       type=str,
                   default="optimization_results_ml.csv")
    p.add_argument("--csv-crossval", type=str,
                   default="optimization_results_crossval.csv")

    # gif (split per backend, optional)
    p.add_argument("--no-gif", action="store_true",
                   help="Skip GIF generation entirely.")
    p.add_argument("--gif-oracle", type=str, default="convergence_oracle.gif")
    p.add_argument("--gif-ml",     type=str, default="convergence_ml.gif")
    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    if args.target in ("ml", "both") and not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Train it first with: python train_models.py")
        sys.exit(1)

    print("=" * 60)
    print("OPTIMISATION METHOD COMPARISON")
    print(f"Target: {args.target} | Device: {args.device} | "
          f"n_users: {args.n_users} | noise: {args.noise} | seed: {args.seed}")
    print("=" * 60)

    ml_model = None
    if args.target in ("ml", "both"):
        ml_model = TrainedMLModel(
            classifier=load_pickled_model(args.model),
            feature_builder=build_feature_vector,
            feature_names=FEATURE_NAMES,
        )
        print(f"Loaded {args.model}\n")

    oracle_results: List[Dict] = []
    ml_results:     List[Dict] = []

    # ---------- ORACLE ----------
    if args.target in ("oracle", "both"):
        oracle_results = run_all_methods(
            lambda: build_oracle_box(args), args, label="ORACLE")
        print_summary(oracle_results, label="ORACLE")
        write_results_csv(oracle_results, args.csv_oracle)
        if not args.no_gif:
            plot_convergence(oracle_results, args.gif_oracle)

    # ---------- ML ----------
    if args.target in ("ml", "both"):
        ml_results = run_all_methods(
            lambda: build_ml_box(args, ml_model), args, label="ML")
        print_summary(ml_results, label="ML")
        write_results_csv(ml_results, args.csv_ml)
        if not args.no_gif:
            plot_convergence(ml_results, args.gif_ml)

    # ---------- CROSS-VALIDATION (only when both backends ran) ----------
    if args.target == "both" and ml_results:
        print_trajectory(ml_results, args)
        cv_rows = cross_validate(ml_results, args)
        pd.DataFrame(cv_rows).to_csv(args.csv_crossval, index=False)
        print(f"Saved cross-validation to {args.csv_crossval}")


if __name__ == "__main__":
    main()