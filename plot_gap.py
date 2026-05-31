"""
Plot the surrogate-truth gap: what each optimizer CLAIMED (ML surrogate)
vs what was actually TRUE (oracle evaluation of the found configuration).

Run:
    python plot_gap.py
Output:
    graphics/surrogate_gap.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ----- data from your 5-seed runs -----
methods       = ["DIRECT", "DE", "BO (GP)"]
ml_claim_mean = [0.2781, 0.2202, 0.6374]
ml_claim_std  = [0.0290, 0.0693, 0.0894]
oracle_mean   = [0.1910, 0.1409, 0.1653]
oracle_std    = [0.0202, 0.0933, 0.1271]  # gap std as proxy

# ----- styling -----
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(8, 5))

x         = np.arange(len(methods))
bar_w     = 0.35
color_clm = "#c0392b"   # red — the lie
color_tru = "#27ae60"   # green — the truth

bars_clm = ax.bar(x - bar_w/2, ml_claim_mean, bar_w,
                  yerr=ml_claim_std, capsize=4,
                  color=color_clm, label="ML claim (what optimiser reported)",
                  edgecolor="white", linewidth=0.8)
bars_tru = ax.bar(x + bar_w/2, oracle_mean,   bar_w,
                  yerr=oracle_std, capsize=4,
                  color=color_tru, label="Oracle truth (real CTR of that config)",
                  edgecolor="white", linewidth=0.8)

# annotate gap above each pair
for i, (clm, tru) in enumerate(zip(ml_claim_mean, oracle_mean)):
    gap = clm - tru
    y   = max(clm, tru) + 0.04
    ax.annotate(
        f"gap = +{gap:.2f}",
        xy=(x[i], y), ha="center", va="bottom",
        fontsize=10, fontweight="bold",
        color="#c0392b" if gap > 0.2 else "#555555",
    )
    # connecting line so the eye reads the difference
    ax.plot([x[i] - bar_w/2, x[i] + bar_w/2],
            [clm, tru], color="#888888", linestyle=":", linewidth=1.2)

# analytical ceiling line
ax.axhline(0.39, color="#444444", linestyle="--", linewidth=1, alpha=0.6)
ax.text(len(methods) - 0.5, 0.40,
        "analytical ceiling (≈0.39)",
        fontsize=9, color="#444444", ha="right", va="bottom", style="italic")

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel("CTR")
ax.set_ylim(0, 0.85)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.set_title(
    "Surrogate vs Oracle: what optimisers report and what is actually true\n"
    "(configurations found on ML surrogate, re-evaluated on the ground-truth formula)",
    fontsize=11, pad=12,
)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.grid(axis="y", alpha=0.25)

os.makedirs("graphics", exist_ok=True)
out = "graphics/surrogate_gap.png"
plt.tight_layout()
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
