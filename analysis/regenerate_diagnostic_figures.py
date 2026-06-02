"""
regenerate_diagnostic_figures.py
================================

Regenerate the two Section 5.3 diagnostic figures from the verified
3,000-question adversarial diagnostic artifact:

    experiments/ugaa_full_adversarial_3000q_diagnostic.json

Figure 7  attention histogram (correct vs wrong, mean lines)
Figure 8  logit gap (left: per-category bar chart with 1-SD error bars;
          right: per-category kernel-density estimates)

Both figures are written as 300-dpi PNG and PDF into every figure
directory used by the papers (paper/arxiv/figures and
paper/neurips/figures), so that \\includegraphics{figures/...} resolves
when each main.tex is compiled in place.

matplotlib only (no seaborn, no plotly). The kernel-density estimate is
a self-contained Gaussian KDE in numpy so the script carries no extra
dependency. No GPU is used. Run from the repo root:

    python analysis/regenerate_diagnostic_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
DIAG = REPO / "experiments" / "ugaa_full_adversarial_3000q_diagnostic.json"
FIG_DIRS = [
    REPO / "paper" / "arxiv" / "figures",
    REPO / "paper" / "neurips" / "figures",
    REPO / "paper" / "html",
]

# Color map (matches Section 5.3 caption text).
C_CORRECT = "#2563EB"   # blue
C_WRONG   = "#DC2626"   # red
C_TP      = "#16A34A"   # green
C_FP      = "#DC2626"   # red
C_TN      = "#2563EB"   # blue
C_FN      = "#D97706"   # amber

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Data loading / grouping
# ---------------------------------------------------------------------------

def load_groups(path: Path):
    with open(path) as f:
        rows = json.load(f)["results"]
    groups = {"correct": [], "wrong": [],
              "TP": [], "TN": [], "FP": [], "FN": []}
    for r in rows:
        pred = r.get("baseline_prediction") or r.get("prediction")
        label = r.get("label")
        ground = r.get("grounding")
        gap = r.get("gap_raw")
        if pred is None or label is None or pred == "error":
            continue
        if ground is None or gap is None:
            continue
        rec = {"grounding": float(ground), "gap_abs": abs(float(gap))}
        (groups["correct"] if pred == label else groups["wrong"]).append(rec)
        if pred == "yes" and label == "yes":
            groups["TP"].append(rec)
        elif pred == "no" and label == "no":
            groups["TN"].append(rec)
        elif pred == "yes" and label == "no":
            groups["FP"].append(rec)
        elif pred == "no" and label == "yes":
            groups["FN"].append(rec)
    return groups


def _vals(group, field):
    return np.array([r[field] for r in group], dtype=float)


def gaussian_kde(samples, grid, bw=None):
    """Self-contained Gaussian KDE (Silverman bandwidth) in numpy."""
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if n < 2:
        return np.zeros_like(grid)
    if bw is None:
        std = samples.std(ddof=1)
        bw = 1.06 * std * n ** (-1.0 / 5.0)
        if bw <= 0:
            bw = 1e-3
    u = (grid[:, None] - samples[None, :]) / bw
    k = np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
    return k.sum(axis=1) / (n * bw)


# ---------------------------------------------------------------------------
# Figure 7: attention histogram (correct vs wrong)
# ---------------------------------------------------------------------------

def figure7(groups):
    correct = _vals(groups["correct"], "grounding")
    wrong = _vals(groups["wrong"], "grounding")
    mc, mw = correct.mean(), wrong.mean()

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    lo = min(correct.min(), wrong.min())
    hi = max(correct.max(), wrong.max())
    bins = np.linspace(lo, hi, 41)
    ax.hist(correct, bins=bins, color=C_CORRECT, alpha=0.6,
            label=f"correct (n={correct.size})", density=True)
    ax.hist(wrong, bins=bins, color=C_WRONG, alpha=0.6,
            label=f"wrong (n={wrong.size})", density=True)

    ax.axvline(mc, color=C_CORRECT, linestyle="--", linewidth=1.4)
    ax.axvline(mw, color=C_WRONG, linestyle="--", linewidth=1.4)
    ymax = ax.get_ylim()[1]
    ax.annotate(f"correct mean = {mc:.4f}", xy=(mc, ymax * 0.92),
                xytext=(mc, ymax * 0.92), color=C_CORRECT, fontsize=9,
                ha="right", va="top")
    ax.annotate(f"wrong mean = {mw:.4f}", xy=(mw, ymax * 0.80),
                xytext=(mw, ymax * 0.80), color=C_WRONG, fontsize=9,
                ha="left", va="top")

    ax.set_xlabel("Mean attention to image tokens, layers 14–20")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 8: logit gap (bar chart + KDE)
# ---------------------------------------------------------------------------

def figure8(groups):
    order = ["TP", "TN", "FP", "FN"]
    colors = {"TP": C_TP, "TN": C_TN, "FP": C_FP, "FN": C_FN}
    vals = {k: _vals(groups[k], "gap_abs") for k in order}
    means = {k: vals[k].mean() for k in order}
    stds = {k: vals[k].std(ddof=1) for k in order}

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.0, 3.5))

    # Left: per-category bar chart with 1-SD error bars.
    x = np.arange(len(order))
    axl.bar(x, [means[k] for k in order],
            yerr=[stds[k] for k in order], capsize=4,
            color=[colors[k] for k in order], alpha=0.85,
            error_kw=dict(ecolor="#404040", lw=1.0))
    axl.set_xticks(x)
    axl.set_xticklabels([f"{k}\n(n={vals[k].size})" for k in order])
    axl.set_ylabel(r"Mean $|\mathrm{logit_{yes}} - \mathrm{logit_{no}}|$")
    for xi, k in zip(x, order):
        axl.annotate(f"{means[k]:.2f}", xy=(xi, means[k]),
                     xytext=(xi, means[k] + stds[k] + 0.05),
                     ha="center", va="bottom", fontsize=9, color="#202020")

    # Right: per-category KDE curves.
    allv = np.concatenate([vals[k] for k in order])
    grid = np.linspace(allv.min(), allv.max(), 256)
    for k in order:
        dens = gaussian_kde(vals[k], grid)
        axr.plot(grid, dens, color=colors[k], lw=1.6, label=k)
        axr.fill_between(grid, dens, color=colors[k], alpha=0.08)
    axr.set_xlabel(r"$|\mathrm{logit_{yes}} - \mathrm{logit_{no}}|$")
    axr.set_ylabel("Density")
    axr.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save(fig, stem):
    written = []
    for d in FIG_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            out = d / f"{stem}.{ext}"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            written.append(out)
    plt.close(fig)
    return written


def main():
    if not DIAG.exists():
        raise SystemExit(f"ERROR: diagnostic JSON not found at {DIAG}")
    groups = load_groups(DIAG)
    print("Group sizes: " + ", ".join(
        f"{k}={len(groups[k])}" for k in ("TP", "TN", "FP", "FN",
                                          "correct", "wrong")))

    for w in save(figure7(groups), "fig7_attention_histogram_3000q"):
        print(f"  wrote {w.relative_to(REPO)}")
    for w in save(figure8(groups), "fig8_logit_gap_3000q"):
        print(f"  wrote {w.relative_to(REPO)}")
    print("done.")


if __name__ == "__main__":
    main()
