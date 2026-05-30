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
    "bg_r", "bg_g", "bg_b",
    "text_r", "text_g", "text_b",
    "btn_w", "btn_h", "font_size",
    "text_quality", "whitespace_ratio", "scroll_to_button",
]

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
        "hour":             12,  # зафиксировано
        "device":           device,
    }


class Objective:
    """Wraps BlackBox calls, records convergence history. Returns -CTR."""

    def __init__(self, box: BlackBox, device: str = "desktop"):
        self.box     = box
        self.device  = device
        self.history = []
        self.best_so_far = []
        self.params_history = []  # добавить

    def __call__(self, x) -> float:
        x = np.asarray(x, dtype=float)
        params = vector_to_params(x, device=self.device)
        result = self.box(params)
        ctr    = result.ctr
        self.history.append(ctr)
        self.params_history.append(params)  # добавить
        running_best = max(self.best_so_far[-1], ctr) if self.best_so_far else ctr
        self.best_so_far.append(running_best)
        return -ctr

    def reset(self):
        self.history     = []
        self.best_so_far = []
        self.params_history = []


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
        "method":         "Random Search",
        "best_ctr":       best_ctr,
        "best_params":    vector_to_params(best_x, device=device),
        "calls":          len(obj.history),
        "history":        obj.history,
        "best_so_far":    obj.best_so_far,
        "params_history": obj.params_history,
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
        "method":         "Grid Search",
        "best_ctr":       best_ctr,
        "best_params":    vector_to_params(best_x, device=device),
        "calls":          len(obj.history),
        "history":        obj.history,
        "best_so_far":    obj.best_so_far,
        "params_history": obj.params_history,
    }


def run_direct(box: BlackBox, device: str,
               max_calls: int = 100) -> Dict:
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
           maxiter: int = 30, popsize: int = 10,
           seed: int = 42) -> Dict:
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
        "method":         "Bayesian Optimisation (GP)",
        "best_ctr":       float(-res.fun),
        "best_params":    vector_to_params(np.array(res.x), device=device),
        "calls":          len(obj.history),
        "history":        obj.history,
        "best_so_far":    obj.best_so_far,
        "params_history": obj.params_history,
    }


def plot_convergence(all_results: List[Dict], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    COLORS = {
        "Grid Search":              "#e67e22",
        "DIRECT":                   "#2980b9",
        "Differential Evolution":   "#27ae60",
        "Bayesian Optimisation":    "#8e44ad",
        "Oracle (upper bound)":     "#c0392b",
    }
    STYLES = {
        "Grid Search":              "-",
        "DIRECT":                   "--",
        "Differential Evolution":   "-.",
        "Bayesian Optimisation":    ":",
        "Oracle (upper bound)":     "-",
    }

    max_calls = max(len(r["best_so_far"]) for r in all_results)
    histories = {res["method"]: res["best_so_far"] for res in all_results}

    best_result = max(
        [r for r in all_results if r["method"] != "Grid Search"],
        key=lambda r: r["best_ctr"]
    )
    params_history = best_result["params_history"]
    best_so_far    = best_result["best_so_far"]

    def get_best_params_at(call_idx):
        history = best_result["history"]
        actual_idx = min(call_idx, len(history))
        if actual_idx == 0:
            return params_history[0]
        best_idx = int(np.argmax(history[:actual_idx]))
        return params_history[best_idx]

    def draw_button(ax, params, ctr, method_name):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("#f4f4f4")

        bg_hex  = "#{:02x}{:02x}{:02x}".format(*[int(c) for c in params["rgb_bg"]])
        txt_hex = "#{:02x}{:02x}{:02x}".format(*[int(c) for c in params["rgb_text"]])

        btn_w = 0.18 + (params["btn_w"] / 300) * 0.60
        btn_h = 0.06 + (params["btn_h"] / 120) * 0.22
        btn_w = min(btn_w, 0.82)
        btn_h = min(btn_h, 0.30)
        bx = 0.5 - btn_w / 2
        by = 0.5  - btn_h / 2

        shadow = FancyBboxPatch(
            (bx + 0.008, by - 0.012), btn_w, btn_h,
            boxstyle="round,pad=0.01",
            linewidth=0, facecolor="#cccccc", zorder=1,
        )
        ax.add_patch(shadow)

        btn = FancyBboxPatch(
            (bx, by), btn_w, btn_h,
            boxstyle="round,pad=0.01",
            linewidth=1.2, edgecolor="#aaaaaa",
            facecolor=bg_hex, zorder=2,
        )
        ax.add_patch(btn)

        tq = params["text_quality"]
        label = "Buy Now" if tq >= 0.5 else "buy now!!!"
        font_scale = max(6, min(params["font_size"] / 16.0 * 9, 16))
        ax.text(
            0.5, 0.5, label,
            ha="center", va="center",
            fontsize=font_scale, color=txt_hex,
            alpha=0.3 + 0.7 * tq,
            fontweight="bold" if tq >= 0.6 else "normal",
            zorder=3,
        )

        color = "#2e7d32" if ctr >= 0.15 else "#f57c00" if ctr >= 0.06 else "#c62828"
        ax.text(
            0.5, 0.06, f"CTR = {ctr:.3f}",
            ha="center", va="bottom",
            fontsize=8, color=color, fontweight="bold",
        )
        ax.text(
            0.5, 0.94, method_name,
            ha="center", va="top",
            fontsize=7, color=COLORS.get(method_name, "#555555"),
            fontweight="bold",
        )

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Optimisation Method Comparison", fontsize=13, fontweight="bold")

    # левый график — конвергенция
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(1, max_calls)
    all_vals = [v for h in histories.values() for v in h]
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
            [], [],
            label=name,
            color=COLORS.get(name, "#555555"),
            linestyle=STYLES.get(name, "-"),
            linewidth=2,
        )
        lines[name] = line
    ax1.legend(fontsize=8)

    # правая сетка 2x2 — по одной кнопке на метод
    n_methods = len(all_results)
    cols = 2
    rows = (n_methods + 1) // 2
    btn_axes = []
    for i in range(n_methods):
        ax = fig.add_subplot(rows, 4, i + cols + 1 + (i // cols) * cols)
        btn_axes.append(ax)

    # правильная сетка через GridSpec
    import matplotlib.gridspec as gridspec
    fig.clear()
    fig.suptitle("Optimisation Method Comparison", fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.4)
    ax1 = fig.add_subplot(gs[:, 0])  # левая колонка целиком
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
            [], [],
            label=name,
            color=COLORS.get(name, "#555555"),
            linestyle=STYLES.get(name, "-"),
            linewidth=2,
        )
        lines[name] = line
    ax1.legend(fontsize=8)

    # 4 кнопки в правых 2x2 ячейках
    btn_positions = [gs[0, 1], gs[0, 2], gs[1, 1], gs[1, 2]]
    btn_axes = [fig.add_subplot(btn_positions[i]) for i in range(min(4, n_methods))]

    # помощник: лучший params метода на шаге call_idx
    def get_params_at(result, call_idx):
        history = result["history"]
        actual_idx = min(call_idx, len(history))
        if actual_idx == 0:
            return result["params_history"][0]
        best_idx = int(np.argmax(history[:actual_idx]))
        return result["params_history"][best_idx]

    def get_ctr_at(result, call_idx):
        bsf = result["best_so_far"]
        idx = min(call_idx, len(bsf)) - 1
        return bsf[max(idx, 0)]

    N_FRAMES = 240
    step = max(1, max_calls // N_FRAMES)

    def update(frame):
        call_idx = min((frame + 1) * step, max_calls)

        # конвергенция
        for name, line in lines.items():
            actual_len = len(histories[name])
            xs = list(range(1, min(call_idx, actual_len) + 1))
            line.set_data(xs, histories[name][:len(xs)])

        # кнопки
        for i, res in enumerate(all_results[:4]):
            cur_params = get_params_at(res, call_idx)
            cur_ctr    = get_ctr_at(res, call_idx)
            draw_button(btn_axes[i], cur_params, cur_ctr, res["method"])

        return list(lines.values())

    anim = FuncAnimation(
        fig, update,
        frames=N_FRAMES,
        interval=50,
        blit=False,
    )

    gif_path = out_path.replace(".png", ".gif")
    anim.save(gif_path, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"Saved convergence animation to {gif_path}")

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

def make_box(args, model):
    return BlackBox(
        model=model, n_users=args.n_users,
        noise_std=args.noise, seed=args.seed,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare optimisation methods.")
    parser.add_argument("--model",        type=str,   default="best_model.pkl")
    parser.add_argument("--n-users",      type=int,   default=10000)
    parser.add_argument("--noise",        type=float, default=0.01)
    parser.add_argument("--device",       type=str,   default="desktop",
                        choices=["mobile", "desktop"])
    parser.add_argument("--grid-size",    type=int,   default=3)
    parser.add_argument("--direct-calls", type=int,   default=100)
    parser.add_argument("--de-iter",      type=int,   default=30)
    parser.add_argument("--de-pop",       type=int,   default=10)
    parser.add_argument("--bo-calls",     type=int,   default=50)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--csv-out",      type=str,   default="optimization_results.csv")
    parser.add_argument("--plot-out",     type=str,   default="convergence.png")
    parser.add_argument("--skip-grid",    action="store_true")
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
    oracle_box = BlackBox(
        model=FullSyntheticModel(), n_users=args.n_users,
        noise_std=args.noise, seed=args.seed,
    )

    print(f"Loaded {args.model}")
    print(f"Device: {args.device}, n_users: {args.n_users}, noise: {args.noise}\n")

    methods = [
    ("DIRECT", lambda: run_direct(
        make_box(args, ml_model), device=args.device, max_calls=args.direct_calls)),
    ("Differential Evolution", lambda: run_de(
        make_box(args, ml_model), device=args.device, maxiter=args.de_iter,
        popsize=args.de_pop, seed=args.seed)),
    ("Bayesian Optimisation", lambda: run_bo(
        make_box(args, ml_model), device=args.device,
        n_calls=args.bo_calls, seed=args.seed)),
    ]

    all_results = []
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
    print(f"\n{'Method':<30} {'Calls':>6} {'Best CTR':>10} {'Runtime':>10}")
    print("-" * 60)
    rows = []
    for res in sorted(all_results, key=lambda r: -r["best_ctr"]):
        print(f"{res['method']:<30} {res['calls']:>6} "
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
