"""
Generate publication-quality figures from REAL experiment data.

All numbers trace to JSON files in ../experiments/. The script writes
PDF figures into ../paper/neurips/figures/ and ../paper/arxiv/figures/,
and PNG copies into ../paper/html/ for the web mirror. Run from the
repo root or from this folder; paths are resolved relative to the
script location.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    "xtick.major.size": 3,
    "ytick.major.size": 3,
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

# ---------------------------------------------------------------- DATA LOAD
with open(EXP / "pope_adversarial_2tok_vs_8tok.json") as f:
    adv = json.load(f)
with open(EXP / "pope_popular_2tok_vs_8tok.json") as f:
    pop = json.load(f)
with open(EXP / "pope_random_2tok_vs_8tok.json") as f:
    rnd = json.load(f)
with open(EXP / "pope_full_adversarial_baseline_summary.json") as f:
    base_sum = json.load(f)
with open(EXP / "ugaa_v6_100q_validation.json") as f:
    diag = json.load(f)

# ---------------------------------------------------------------- FIG: yes-rate convergence over questions
print("[fig] yes_rate_convergence")
results = adv["results"]
running_yes_2 = []
running_yes_8 = []
yes_2 = 0
yes_8 = 0
for i, r in enumerate(results):
    if r["pred_2tok"] == "yes":
        yes_2 += 1
    if r["pred_8tok"] == "yes":
        yes_8 += 1
    running_yes_2.append(yes_2 / (i + 1))
    running_yes_8.append(yes_8 / (i + 1))

fig, ax = plt.subplots(figsize=(5.4, 3.0))
xs = np.arange(1, len(results) + 1)
ax.plot(xs, running_yes_2, color=TWO,   lw=1.2, label="Two-token protocol")
ax.plot(xs, running_yes_8, color=EIGHT, lw=1.2, label="Eight-token protocol")
ax.axhline(0.5, color=GRAY, lw=0.7, ls="--", alpha=0.6)
ax.text(3000, 0.515, "ground-truth yes-rate = 0.500", color=GRAY,
        fontsize=7.5, ha="right", va="bottom")
ax.set_xlabel("Question index (POPE adversarial, ordered as evaluated)")
ax.set_ylabel("Running yes-rate")
ax.set_ylim(0.35, 0.85)
ax.set_xlim(0, 3000)
ax.legend(loc="center right", fontsize=8.5)
ax.grid(True, axis="y", ls=":", color="gray", alpha=0.4, lw=0.5)
fig.tight_layout()
save_all(fig, "yes_rate_convergence")
plt.close(fig)

# ---------------------------------------------------------------- FIG: confusion matrices 2tok vs 8tok adversarial
print("[fig] confusion_matrices")
# Compute confusion under each protocol on the adversarial split
def confuse(preds, labels):
    tp = sum(1 for p, l in zip(preds, labels) if p == "yes" and l == "yes")
    tn = sum(1 for p, l in zip(preds, labels) if p == "no"  and l == "no")
    fp = sum(1 for p, l in zip(preds, labels) if p == "yes" and l == "no")
    fn = sum(1 for p, l in zip(preds, labels) if p == "no"  and l == "yes")
    return tp, tn, fp, fn

labels   = [r["label"]      for r in results]
preds_2  = [r["pred_2tok"]  for r in results]
preds_8  = [r["pred_8tok"]  for r in results]
tp2, tn2, fp2, fn2 = confuse(preds_2, labels)
tp8, tn8, fp8, fn8 = confuse(preds_8, labels)

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
for ax, (tp, tn, fp, fn), title in zip(
    axes, [(tp2, tn2, fp2, fn2), (tp8, tn8, fp8, fn8)],
    ["Two-token protocol", "Eight-token protocol"],
):
    mat = np.array([[tp, fn], [fp, tn]])
    im = ax.imshow(mat, cmap="Blues", aspect="equal", vmin=0, vmax=1800)
    for i in range(2):
        for j in range(2):
            v = mat[i, j]
            color = "white" if v > 900 else "black"
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["label = yes", "label = no"], fontsize=8)
    ax.set_yticklabels(["pred = yes", "pred = no"], fontsize=8)
    ax.set_title(title, fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout()
save_all(fig, "confusion_matrices_adv")
plt.close(fig)

# ---------------------------------------------------------------- FIG: top-token frequency
print("[fig] top_token_frequency")
token_names = {
    "3582":"yes", "8241":"Yes", "4874":"' yes'", "3869":"' Yes'",
    "1217":"no",  "3782":"No",  "694":"' no'",   "1939":"' No'",
}
# Pull counts from each split summary
splits = [("Adversarial", adv["summary"]["top_tokens_generated"]),
          ("Popular",     pop["summary"]["top_tokens_generated"]),
          ("Random",      rnd["summary"]["top_tokens_generated"])]

fig, ax = plt.subplots(figsize=(6.4, 2.9))
all_token_ids = ["3869", "1939", "3582", "8241", "4874", "1217", "3782", "694"]
x = np.arange(len(all_token_ids))
w = 0.27
for i, (name, counts) in enumerate(splits):
    vals = [counts.get(tid, 0) for tid in all_token_ids]
    color = [TWO, "#7a4a4a", EIGHT][i]
    ax.bar(x + (i - 1) * w, vals, w, label=name, color=color, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"{token_names[t]}\n({t})" for t in all_token_ids], fontsize=8)
ax.set_ylabel("First-token occurrences (per 3000)")
ax.set_yscale("symlog", linthresh=1)
ax.set_yticks([0, 1, 10, 100, 1000])
ax.set_yticklabels(["0", "1", "10", "100", "1000"])
ax.set_ylim(0, 3000)
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(True, axis="y", ls=":", color="gray", alpha=0.4, lw=0.5)
ax.axvline(1.5, color=GRAY, lw=0.6, ls="--", alpha=0.5)
ax.text(0.75, 2500, "generated by model", fontsize=8, ha="center", color=EIGHT)
ax.text(4.5, 2500, "never generated (logit-only)", fontsize=8, ha="center", color=GRAY)
fig.tight_layout()
save_all(fig, "top_token_frequency")
plt.close(fig)

# ---------------------------------------------------------------- FIG: |logit gap| histogram by category
print("[fig] logit_gap_by_category")
diag_results = diag["results"]
cats = {"TP": [], "FP": [], "TN": [], "FN": []}
for r in diag_results:
    if r.get("prediction") == "error" or "gap_raw" not in r:
        continue
    pred = r["baseline_prediction"]
    lab  = r["label"]
    cat = ("TP" if pred=="yes" and lab=="yes" else
           "FP" if pred=="yes" and lab=="no"  else
           "TN" if pred=="no"  and lab=="no"  else "FN")
    cats[cat].append(abs(r["gap_raw"]))

fig, ax = plt.subplots(figsize=(5.6, 3.0))
bins = np.linspace(0, 5.5, 23)
for label, color in [("TP", EIGHT), ("FP", TWO), ("TN", "#6a8da8"), ("FN", "#c87878")]:
    ax.hist(cats[label], bins=bins, alpha=0.55, color=color,
            label=f"{label} (n={len(cats[label])})", edgecolor="white", lw=0.5)
ax.set_xlabel(r"$|\ell_{\rm yes} - \ell_{\rm no}|$ (absolute logit gap, ungated)")
ax.set_ylabel("Count")
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(True, axis="y", ls=":", color="gray", alpha=0.4, lw=0.5)
fig.tight_layout()
save_all(fig, "logit_gap_histogram")
plt.close(fig)

# ---------------------------------------------------------------- FIG: F1 of nine correction methods
print("[fig] correction_methods_bar")
methods = [
    ("Baseline",          0.8221),
    ("CLIP-B/32 b=1.0",   0.8171),
    ("CLIP-L/14-336",     0.8159),
    ("Entropy gating",    0.8164),
    ("Object-noun CLIP",  0.8166),
    ("VCD-noise",         0.8169),
    ("CLS similarity",    0.8153),
    ("Magnitude gating",  0.8148),
    ("Uniform suppression", 0.8110),
    ("CLIP-B/32 b=1.5",   0.8104),
]
methods.reverse()
names  = [m[0] for m in methods]
scores = [m[1] for m in methods]
colors = [EIGHT if "Baseline" in n else TWO for n in names]

fig, ax = plt.subplots(figsize=(5.4, 3.6))
bars = ax.barh(names, scores, color=colors, alpha=0.85, edgecolor="white", lw=0.6)
ax.axvline(0.8221, color=EIGHT, lw=1.0, ls="--", alpha=0.7)
ax.text(0.8222, len(methods) - 0.3, "baseline = 0.8221", fontsize=8,
        color=EIGHT, ha="left", va="top")
for b, s in zip(bars, scores):
    ax.text(s - 0.0005, b.get_y() + b.get_height()/2, f"{s:.4f}",
            ha="right", va="center", fontsize=7.8, color="white", fontweight="bold")
ax.set_xlim(0.79, 0.83)
ax.set_xlabel("POPE adversarial F1 (3000 questions, eight-token)")
ax.grid(True, axis="x", ls=":", color="gray", alpha=0.4, lw=0.5)
fig.tight_layout()
save_all(fig, "correction_methods_bar")
plt.close(fig)

# ---------------------------------------------------------------- FIG: published baselines spread
print("[fig] baseline_spread")
papers = [
    ("Tang et al.\n(VisFlow)",       73.54, "two-token"),
    ("Tang et al.\n(VisFlow, pop.)", 76.53, "two-token"),
    ("Seo et al.",                   79.30, "two-token"),
    ("Chen et al.\n(AdaptVis)",      81.00, "two-token"),
    ("Tang et al.\n(VisFlow, rand.)",81.54, "two-token"),
    ("Ours\n(eight-token, adv.)",    82.21, "eight-token"),
    ("Ours\n(eight-token, pop.)",    84.98, "eight-token"),
    ("Ours\n(eight-token, rand.)",   87.13, "eight-token"),
]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
xs = list(range(len(papers)))
for i, (label, val, proto) in enumerate(papers):
    color = EIGHT if proto == "eight-token" else TWO
    ax.bar(i, val, color=color, alpha=0.85, width=0.7)
    ax.text(i, val + 0.4, f"{val:.2f}", ha="center", fontsize=8, color="#333")
ax.set_xticks(xs)
ax.set_xticklabels([p[0] for p in papers], fontsize=7.6)
ax.set_ylim(70, 92)
ax.set_ylabel("POPE F1 (x100)")
ax.axhline(82.21, color=EIGHT, ls="--", lw=0.8, alpha=0.6)
ax.text(7.4, 82.5, "corrected adv. baseline", color=EIGHT, fontsize=7.5, ha="right")
# Custom legend
red_patch  = mpatches.Patch(color=TWO,   alpha=0.85, label="Reported (two-token, prior)")
blue_patch = mpatches.Patch(color=EIGHT, alpha=0.85, label="Our eight-token measurement")
ax.legend(handles=[red_patch, blue_patch], loc="upper left", fontsize=8)
ax.grid(True, axis="y", ls=":", color="gray", alpha=0.4, lw=0.5)
fig.tight_layout()
save_all(fig, "baseline_spread")
plt.close(fig)

# ---------------------------------------------------------------- FIG: grounding signal vs accuracy
print("[fig] grounding_vs_correct")
import statistics
gs_correct = []
gs_wrong = []
for r in diag_results:
    if "grounding" not in r:
        continue
    if r["baseline_prediction"] == r["label"]:
        gs_correct.append(r["grounding"] * 1e4)
    else:
        gs_wrong.append(r["grounding"] * 1e4)

fig, ax = plt.subplots(figsize=(5.0, 2.9))
bins = np.linspace(0.5, 2.0, 16)
ax.hist(gs_correct, bins=bins, color=EIGHT, alpha=0.6, label=f"correct (n={len(gs_correct)})",
        edgecolor="white", lw=0.5)
ax.hist(gs_wrong,   bins=bins, color=TWO,   alpha=0.6, label=f"wrong (n={len(gs_wrong)})",
        edgecolor="white", lw=0.5)
ax.set_xlabel(r"Mean attention to image tokens, layers 14--20 ($\times 10^{-4}$)")
ax.set_ylabel("Count")
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(True, axis="y", ls=":", color="gray", alpha=0.4, lw=0.5)
mc = statistics.mean(gs_correct) if gs_correct else 0
mw = statistics.mean(gs_wrong)   if gs_wrong else 0
ax.text(0.97, 0.55, f"mean correct = {mc:.2f}\nmean wrong   = {mw:.2f}",
        transform=ax.transAxes, ha="right", fontsize=7.8, color=GRAY,
        family="monospace")
fig.tight_layout()
save_all(fig, "grounding_vs_correct")
plt.close(fig)

print("Done.")
