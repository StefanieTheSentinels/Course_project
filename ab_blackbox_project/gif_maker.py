"""
make_gif.py
-----------
Генерирует анимированный GIF: слева — кнопка меняется в реальном времени,
справа — график p(click) показывает текущее значение.

Анимация проходит через 4 сценария по кругу:
  1. contrast  : белое на белом → чёрное на белом
  2. scroll    : кнопка внизу страницы → вверху
  3. size      : маленькая кнопка → большая
  4. text qual : плохой текст → хороший

Запуск:
    python make_gif.py
    python make_gif.py --out my_button.gif --fps 25 --dpi 120
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generating_formula import p_click, contrast_ratio


# ── Параметры анимации ─────────────────────────────────────────────────────

N_STEPS   = 40      # кадров на один сценарий
SCENARIOS = [
    "contrast",
    "scroll",
    "size",
    "text_quality",
]
TOTAL_FRAMES = N_STEPS * len(SCENARIOS)

# Базовый «хороший» конфиг — отправная точка каждого сценария
BASE = {
    "rgb_bg":           (255, 255, 255),
    "rgb_text":         (30,  30,  30),
    "btn_w":            160,
    "btn_h":            52,
    "font_size":        16,
    "text_quality":     0.85,
    "whitespace_ratio": 0.30,
    "scroll_to_button": 0.05,
    "hour":             12,
    "device":           "desktop",
}

SCENARIO_LABELS = {
    "contrast":     "контраст текста",
    "scroll":       "позиция на странице",
    "size":         "размер кнопки",
    "text_quality": "качество текста",
}

SCENARIO_XLABELS = {
    "contrast":     "contrast ratio (1 → 21)",
    "scroll":       "scroll_to_button (0 → 1)",
    "size":         "btn_w × btn_h (px²)",
    "text_quality": "text_quality (0 → 1)",
}


# ── Вычисление p(click) вдоль оси сценария ────────────────────────────────

def make_curve(scenario: str, n: int = 200):
    """Кривая p(click) по оси выбранного сценария."""
    xs, ys = [], []
    for t in np.linspace(0, 1, n):
        cfg = dict(BASE)
        cfg = apply_scenario(cfg, scenario, t)
        xs.append(t)
        ys.append(p_click(cfg))
    return np.array(xs), np.array(ys)


def apply_scenario(cfg: dict, scenario: str, t: float) -> dict:
    """t ∈ [0,1] — позиция вдоль оси сценария."""
    cfg = dict(cfg)
    if scenario == "contrast":
        # t=0: белое на белом (ratio≈1), t=1: чёрное на белом (ratio≈21)
        text_val = int(255 * (1 - t))
        cfg["rgb_text"] = (text_val, text_val, text_val)
    elif scenario == "scroll":
        # t=0: кнопка вверху, t=1: внизу
        cfg["scroll_to_button"] = float(t)
    elif scenario == "size":
        # t=0: маленькая (40×24), t=1: большая (280×80)
        cfg["btn_w"] = int(40  + t * (280 - 40))
        cfg["btn_h"] = int(24  + t * (80  - 24))
    elif scenario == "text_quality":
        cfg["text_quality"] = float(t)
    return cfg


def get_x_axis_value(scenario: str, t: float) -> float:
    """Значение по оси X для подписи точки на графике."""
    if scenario == "contrast":
        text_val = int(255 * (1 - t))
        return contrast_ratio((255,255,255), (text_val,text_val,text_val))
    elif scenario == "scroll":
        return t
    elif scenario == "size":
        w = 40  + t * (280 - 40)
        h = 24  + t * (80  - 24)
        return w * h
    elif scenario == "text_quality":
        return t
    return t


# ── Отрисовка кнопки ──────────────────────────────────────────────────────

def draw_button(ax, cfg: dict, p: float, scenario: str):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    bg_r, bg_g, bg_b = cfg["rgb_bg"]
    tx_r, tx_g, tx_b = cfg["rgb_text"]
    bg_hex = "#{:02x}{:02x}{:02x}".format(bg_r, bg_g, bg_b)
    tx_hex = "#{:02x}{:02x}{:02x}".format(tx_r, tx_g, tx_b)

    # Фон «страницы»
    ax.set_facecolor("#f4f4f4")

    # Позиция кнопки по вертикали зависит от scroll
    scroll = cfg.get("scroll_to_button", 0.05)
    # scroll=0 → кнопка высоко (y≈0.65), scroll=1 → низко (y≈0.08)
    btn_cy = 0.72 - scroll * 0.55

    # Масштаб кнопки
    raw_w = cfg["btn_w"]
    raw_h = cfg["btn_h"]
    max_dim = max(raw_w, raw_h, 1)
    btn_w = 0.18 + (raw_w / 300) * 0.60   # [0.18 … 0.78]
    btn_h = 0.06 + (raw_h / 80)  * 0.18   # [0.06 … 0.24]
    btn_w = min(btn_w, 0.82)
    btn_h = min(btn_h, 0.28)

    bx = 0.5 - btn_w / 2
    by = btn_cy - btn_h / 2

    # Тень (лёгкая)
    shadow = FancyBboxPatch(
        (bx + 0.008, by - 0.012), btn_w, btn_h,
        boxstyle="round,pad=0.01",
        linewidth=0, facecolor="#cccccc", zorder=1,
    )
    ax.add_patch(shadow)

    # Сама кнопка
    btn = FancyBboxPatch(
        (bx, by), btn_w, btn_h,
        boxstyle="round,pad=0.01",
        linewidth=1.2,
        edgecolor="#aaaaaa",
        facecolor=bg_hex,
        zorder=2,
    )
    ax.add_patch(btn)

    # Текст кнопки
    font_scale = cfg["font_size"] / 16.0
    tq = cfg["text_quality"]
    label = "Купить сейчас" if tq >= 0.5 else "купить сейчас!!!"
    alpha_text = 0.3 + 0.7 * tq
    ax.text(
        0.5, btn_cy, label,
        ha="center", va="center",
        fontsize=8.5 * min(font_scale, 1.4),
        color=tx_hex, alpha=alpha_text,
        fontweight="bold" if tq >= 0.6 else "normal",
        zorder=3,
    )

    # Индикатор позиции прокрутки (маленький скроллбар справа)
    bar_x = 0.92
    ax.plot([bar_x, bar_x], [0.08, 0.92], color="#cccccc", lw=2, zorder=1)
    thumb_y = 0.92 - scroll * 0.84
    ax.plot([bar_x, bar_x], [thumb_y - 0.04, thumb_y + 0.04],
            color="#888888", lw=4, solid_capstyle="round", zorder=2)

    # Метка сценария
    ax.text(
        0.5, 0.97,
        SCENARIO_LABELS[scenario],
        ha="center", va="top",
        fontsize=8.5, color="#555555",
        fontweight="bold",
    )

    # CTR badge
    color = "#2e7d32" if p >= 0.15 else "#f57c00" if p >= 0.06 else "#c62828"
    ax.text(
        0.5, 0.03,
        f"p(click) = {p:.3f}",
        ha="center", va="bottom",
        fontsize=10, color=color,
        fontweight="bold",
    )


# ── Главный скрипт ────────────────────────────────────────────────────────

def build_animation(fps: int, dpi: int):
    # Предвычисляем все кривые
    curves = {s: make_curve(s) for s in SCENARIOS}

    fig, (ax_btn, ax_plot) = plt.subplots(
        1, 2,
        figsize=(9, 4.2),
        gridspec_kw={"width_ratios": [1, 1.5]},
    )
    fig.patch.set_facecolor("#ffffff")
    plt.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.14, wspace=0.28)

    # Заголовок
    title = fig.suptitle("", fontsize=11, fontweight="bold", color="#222222", y=0.97)

    # Настройка правого графика
    ax_plot.set_facecolor("#fafafa")
    for spine in ax_plot.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")

    # Элементы графика, которые будем обновлять
    curve_line, = ax_plot.plot([], [], lw=2.0, color="#1565c0", zorder=2)
    fill_poly   = ax_plot.fill_between([], [], alpha=0)
    dot,        = ax_plot.plot([], [], "o", ms=9, color="#e53935", zorder=5)
    vline       = ax_plot.axvline(x=0, color="#e53935", lw=1.0, ls="--",
                                   alpha=0.5, zorder=3)
    ax_plot.set_ylabel("p(click)", fontsize=9, color="#444444")
    ax_plot.tick_params(labelsize=8, colors="#666666")
    ax_plot.grid(True, alpha=0.25, lw=0.5)

    def update(frame):
        nonlocal fill_poly

        scenario_idx = frame // N_STEPS
        step         = frame  % N_STEPS
        scenario     = SCENARIOS[scenario_idx]
        t            = step / (N_STEPS - 1)

        cfg = apply_scenario(dict(BASE), scenario, t)
        p   = p_click(cfg)

        # Обновить кнопку
        draw_button(ax_btn, cfg, p, scenario)

        # Обновить кривую и точку
        xs_norm, ys = curves[scenario]
        x_val = get_x_axis_value(scenario, t)

        # Пересчитать ось X в «настоящих» единицах
        if scenario == "contrast":
            xs_real = np.array([
                get_x_axis_value(scenario, tt) for tt in xs_norm
            ])
            xlim = (1, 21)
            x_cur = x_val
        elif scenario == "scroll":
            xs_real = xs_norm
            xlim = (0, 1)
            x_cur = t
        elif scenario == "size":
            xs_real = np.array([
                get_x_axis_value(scenario, tt) for tt in xs_norm
            ])
            xlim = (40*24, 280*80)
            x_cur = x_val
        else:
            xs_real = xs_norm
            xlim = (0, 1)
            x_cur = t

        curve_line.set_data(xs_real, ys)
        ax_plot.set_xlim(*xlim)
        ax_plot.set_ylim(-0.01, max(ys) * 1.18 + 0.01)
        ax_plot.set_xlabel(SCENARIO_XLABELS[scenario], fontsize=8.5, color="#444444")

        # Перерисовать fill_between
        fill_poly.remove()
        fill_poly = ax_plot.fill_between(
            xs_real, ys, alpha=0.10, color="#1565c0", zorder=1
        )

        # Текущая точка
        dot.set_data([x_cur], [p])
        vline.set_xdata([x_cur])

        # Заголовок
        pct = int(t * 100)
        title.set_text(
            f"Сценарий {scenario_idx+1}/{len(SCENARIOS)}: {SCENARIO_LABELS[scenario]}  "
            f"({pct}%)"
        )

        return curve_line, dot, vline, fill_poly

    anim = FuncAnimation(
        fig, update,
        frames=TOTAL_FRAMES,
        interval=1000 / fps,
        blit=False,
    )
    return fig, anim


def main():
    parser = argparse.ArgumentParser(description="Генератор GIF: кнопка + p(click)")
    parser.add_argument("--out", default="button_animation.gif")
    parser.add_argument("--fps", type=int,   default=20)
    parser.add_argument("--dpi", type=int,   default=110)
    args = parser.parse_args()

    print(f"Генерирую {TOTAL_FRAMES} кадров ({len(SCENARIOS)} сценария × {N_STEPS} шагов)...")
    print(f"FPS={args.fps}, DPI={args.dpi} → ~{TOTAL_FRAMES/args.fps:.1f}s анимации")

    fig, anim = build_animation(fps=args.fps, dpi=args.dpi)

    writer = PillowWriter(fps=args.fps)
    anim.save(args.out, writer=writer, dpi=args.dpi)
    plt.close(fig)

    size_kb = os.path.getsize(args.out) // 1024
    print(f"Сохранено: {args.out}  ({size_kb} KB)")


if __name__ == "__main__":
    main()