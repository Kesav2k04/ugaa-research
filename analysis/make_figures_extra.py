"""
Generate three more publication-quality figures from real experiment data:
1. precision_recall_methods.pdf  - 9-method P-R scatter on adversarial
2. yes_rate_all_splits.pdf       - running yes-rate per split (adv/pop/rand)
3. logit_gap_two_panel.pdf       - dual-panel: per-cat means + density
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
OUT_DIRS = [
    REPO / "paper" / "neurips" / "figures",
    REPO / "paper" / "arxiv" / "figures",
]
HTML_DIR = REPO / "paper" / "html"

TWO   = "#b43232"
EIGHT = "#2864a0"
GRAY  = "#5a5a5a"
LIGHT = "#e6e6e6"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":   9,
    "axes.linewidth": 0.7,
    "axes.spines.right": False,
    "axes.spines.top":   False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "pdf.fonttype": 42,
})

def save_all(fig, name):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight", dpi=180)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(HTML_DIR / f"{name}.png", bbox_inches="tight", dpi=180)
    print(f"  saved {name}")


# FIG 1: Precision/Recall scatter for 9 correction methods + baseline
print("[fig] precision_recall_methods")
methods = [
    ("Baseline",            0.8221, 0.8658, 0.7827, EIGHT, "o"),
    ("CLIP-B/32 b=1.0",     0.8171, 0.9072, 0.7433, TWO,   "s"),
    ("CLIP-B/32 b=1.5",     0.8104, 0.9278, 0.7193, TWO,   "s"),
    ("CLIP-L/14-336",       0.8159, 0.9031, 0.7453, TWO,   "v"),
    ("Entropy gating",      0.8164, 0.9023, 0.7453, TWO,   "v"),
    ("Magnitude gating",    0.8148, 0.8981, 0.7467, TWO,   "v"),
    ("CLS similarity",      0.8153, 0.8963, 0.7460, TWO,   "v"),
    ("Object-noun CLIP",    0.8166, 0.9017, 0.7453, TWO,   "v"),
    ("Uniform suppression", 0.8110, 0.9221, 0.7247, TWO,   "x"),
    ("VCD-noise",           0.8169, 0.9009, 0.7440, TWO,   "D"),
]

fig, ax = plt.subplots(figsize=(5.5, 4.2))
# F1 isolines as background
P_grid = np.linspace(0.83, 0.95, 200)
R_grid = np.linspace(0.70, 0.82, 200)
PP, RR = np.meshgrid(P_grid, R_grid)
F1 = 2 * PP * RR / (PP + RR + 1e-12)
cs = ax.contour(PP, RR, F1, levels=[0.78, 0.80, 0.81, 0.8221, 0.83, 0.84],
                colors="gray", linewidths=0.4, linestyles=":", alpha=0.6)
ax.clabel(cs, inline=True, fontsize=6.5, fmt="F1=%.3f")

for name, f1, p, r, c, mk in methods:
    is_baseline = name == "Baseline"
    ms = 100 if is_baseline else 55
    edge = "black" if is_baseline else c
    lw = 1.0 if is_baseline else 0.5
    ax.scatter(p, r, marker=mk, s=ms, color=c, alpha=0.85,
               edgecolors=edge, linewidths=lw, zorder=3)
    # Label
    offset_y = 0.005 if not is_baseline else -0.012
    offset_x = 0.001
    ax.annotate(name, (p, r), xytext=(p + offset_x, r + offset_y),
                fontsize=6.5, color=GRAY, ha="left")

ax.set_xlabel("Precision (POPE adversarial, 3000 questions)")
ax.set_ylabel("Recall (POPE adversarial)")
ax.set_xlim(0.85, 0.94)
ax.set_ylim(0.71, 0.80)
ax.grid(True, ls=":", color="gray", alpha=0.3, lw=0.4)
ax.set_title("Inference-time corrections on the precision-recall plane", fontsize=10)
fig.tight_layout()
save_all(fig, "precision_recall_methods")
plt.close(fig)


# FIG 2: Running yes-rate across all three splits under 8-token
print("[fig] yes_rate_all_splits")
def running_yes(results, key):
    yes = 0
    out = []
    for i, r in enumerate(results):
        if r[key] == "yes":
            yes += 1
        out.append(yes / (i + 1))
    return out

with open(EXP / "pope_adversarial_2tok_vs_8tok.json") as f:
    adv_d = json.load(f)
with open(EXP / "pope_popular_2tok_vs_8tok.json") as f:
    pop_d = json.load(f)
with open(EXP / "pope_random_2tok_vs_8tok.json") as f:
    rnd_d = json.load(f)

adv_r = adv_d["results"]
pop_r = pop_d["results"]
rnd_r = rnd_d["results"]
xs = np.arange(1, 3001)

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
for ax, key, title in [
    (axes[0], "pred_2tok", "Two-token protocol"),
    (axes[1], "pred_8tok", "Eight-token protocol"),
]:
    ax.plot(xs, running_yes(adv_r, key), color="#a82828", lw=1.0, label="Adversarial")
    ax.plot(xs, running_yes(pop_r, key), color="#c87830", lw=1.0, label="Popular")
    ax.plot(xs, running_yes(rnd_r, key), color="#4878a8", lw=1.0, label="Random")
    ax.axhline(0.5, color=GRAY, ls="--", lw=0.7, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Question index")
    ax.set_xlim(0, 3000)
    ax.set_ylim(0.38, 0.82)
    ax.grid(True, axis="y", ls=":", color="gray", alpha=0.4, lw=0.5)
axes[0].set_ylabel("Running yes-rate")
axes[1].legend(loc="lower right", fontsize=8.5)
fig.tight_layout()
save_all(fig, "yes_rate_all_splits")
plt.close(fig)


# FIG 3: Logit gap two-panel
print("[fig] logit_gap_two_panel")
with open(EXP / "ugaa_v6_100q_validation.json") as f:
    diag = json.load(f)

cats = {"TP": [], "FP": [], "TN": [], "FN": []}
for r in diag["results"]:
    if r.get("prediction") == "error" or "gap_raw" not in r:
        continue
    p = r["baseline_prediction"]
    l = r["label"]
    cat = ("TP" if p == "yes" and l == "yes" else
           "FP" if p == "yes" and l == "no" else
           "TN" if p == "no" and l == "no" else "FN")
    cats[cat].append(abs(r["gap_raw"]))

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))

# Left: bar of means
import statistics
order = ["TP", "FP", "TN", "FN"]
colors = [EIGHT, TWO, "#6a8da8", "#c87878"]
means = [statistics.mean(cats[k]) for k in order]
stds  = [statistics.pstdev(cats[k]) for k in order]
ns    = [len(cats[k]) for k in order]
xs = np.arange(len(order))
axes[0].bar(xs, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
            edgecolor="white", lw=0.8, error_kw={"lw": 0.7, "ecolor": GRAY})
for i, (m, n) in enumerate(zip(means, ns)):
    axes[0].text(i, m + 0.08, f"{m:.2f}", ha="center", fontsize=9, color="#333")
    axes[0].text(i, -0.10, f"n={n}", ha="center", fontsize=7.5, color=GRAY)
axes[0].set_xticks(xs)
axes[0].set_xticklabels(order)
axes[0].set_ylabel(r"Mean $|\ell_{\rm yes} - \ell_{\rm no}|$")
axes[0].set_ylim(-0.3, 3.4)
axes[0].grid(True, axis="y", ls=":", color="gray", alpha=0.4, lw=0.5)
axes[0].set_title("Per-category mean logit gap", fontsize=10)

# Right: density curves
from scipy.stats import gaussian_kde
xg = np.linspace(0, 5.5, 400)
for label, color in zip(order, colors):
    arr = np.array(cats[label])
    if len(arr) < 2:
        continue
    if len(arr) >= 3:
        kde = gaussian_kde(arr, bw_method=0.45)
        ys = kde(xg)
        axes[1].plot(xg, ys, color=color, lw=1.3, label=f"{label} (n={len(arr)})")
        axes[1].fill_between(xg, ys, alpha=0.18, color=color)
axes[1].set_xlabel(r"$|\ell_{\rm yes} - \ell_{\rm no}|$")
axes[1].set_ylabel("Density")
axes[1].set_xlim(0, 5)
axes[1].legend(loc="upper right", fontsize=8)
axes[1].grid(True, ls=":", color="gray", alpha=0.4, lw=0.5)
axes[1].set_title("Logit-gap density by category", fontsize=10)

fig.tight_layout()
save_all(fig, "logit_gap_two_panel")
plt.close(fig)

print("Done.")
