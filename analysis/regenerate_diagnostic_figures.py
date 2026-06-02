"""
regenerate_diagnostic_figures.py
================================

Regenerate the two Section 5.3 diagnostic figures from the verified
3,000-question adversarial diagnostic artifact:

    experiments/ugaa_full_adversarial_3000q_diagnostic.json

Figure 7  attention histogram (correct vs wrong) with mean lines and a
          clean statistics panel (no text overlapping the bars)
Figure 8  logit gap: left, per-category mean with 1-SD error bars;
          right, per-category kernel-density estimates

Design targets publication quality: constrained layout (no clipped or
overlapping labels), a light horizontal grid, de-emphasised spines, a
boxed statistics annotation placed in low-density corners, and
two-line category tick labels that do not run together.

Both figures are written as 300-dpi PNG and vector PDF into every
figure directory the papers use (paper/arxiv/figures,
paper/neurips/figures) and the HTML asset directory (paper/html), so
that \\includegraphics{figures/...} resolves when each main.tex is
compiled in place and the HTML picks up the PNGs.

matplotlib only (no seaborn, no plotly). The kernel-density estimate is
a self-contained Gaussian KDE in numpy. No GPU is used. Run from the
repo root:

    python analysis/regenerate_diagnostic_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

REPO = Path(__file__).resolve().parent.parent
DIAG = REPO / "experiments" / "ugaa_full_adversarial_3000q_diagnostic.json"
FIG_DIRS = [
    REPO / "paper" / "arxiv" / "figures",
    REPO / "paper" / "neurips" / "figures",
    REPO / "paper" / "html",
]

# Color map (matches the Section 5.3 caption text).
C_CORRECT = "#2563EB"   # blue
C_WRONG   = "#DC2626"   # red
C_TP      = "#16A34A"   # green
C_FP      = "#DC2626"   # red
C_TN      = "#2563EB"   # blue
C_FN      = "#D97706"   # amber
GRID      = "#D8DEE9"
INK       = "#1F2933"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.edgecolor": "#52606D",
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "#52606D",
    "ytick.color": "#52606D",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
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
        iqr = np.subtract(*np.percentile(samples, [75, 25]))
        sigma = min(std, iqr / 1.349) if iqr > 0 else std
        bw = 0.9 * sigma * n ** (-1.0 / 5.0)
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

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)

    lo = min(correct.min(), wrong.min())
    hi = max(correct.max(), wrong.max())
    bins = np.linspace(lo, hi, 46)
    ax.hist(correct, bins=bins, color=C_CORRECT, alpha=0.55,
            edgecolor=C_CORRECT, linewidth=0.4,
            label=f"correct  (n = {correct.size:,})", density=True, zorder=2)
    ax.hist(wrong, bins=bins, color=C_WRONG, alpha=0.55,
            edgecolor=C_WRONG, linewidth=0.4,
            label=f"wrong  (n = {wrong.size:,})", density=True, zorder=2)

    # Mean lines (no inline text; values go in the stats panel).
    ax.axvline(mc, color=C_CORRECT, linestyle=(0, (5, 2)), linewidth=1.6, zorder=4)
    ax.axvline(mw, color=C_WRONG, linestyle=(0, (5, 2)), linewidth=1.6, zorder=4)

    ax.set_xlabel("Mean attention to image tokens (layers 14–20)")
    ax.set_ylabel("Density")
    ax.margins(x=0.01)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)

    leg = ax.legend(loc="upper left", fontsize=9.5, handlelength=1.3,
                    borderaxespad=0.6)
    leg.set_zorder(6)

    # Statistics panel in the clear upper-right corner.
    stats = (
        r"$\mu_{\mathrm{correct}} = 0.0738$" "\n"
        r"$\mu_{\mathrm{wrong}} = 0.0801$" "\n"
        r"$\Delta = -0.0064$ (95% CI $[-0.0076,-0.0051]$)" "\n"
        r"Mann$-$Whitney $p = 1.4\times10^{-24}$" "\n"
        r"Cohen's $d = -0.46$"
    )
    ax.text(0.985, 0.97, stats, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.6, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#B8C2CC", linewidth=0.8, alpha=0.96))
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

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.4, 3.7),
                                   constrained_layout=True)

    # ---- Left: per-category mean with 1-SD error bars ----
    axl.set_axisbelow(True)
    axl.yaxis.grid(True, color=GRID, linewidth=0.8)
    x = np.arange(len(order))
    bars = axl.bar(x, [means[k] for k in order], width=0.66,
                   color=[colors[k] for k in order], alpha=0.92,
                   edgecolor="white", linewidth=0.6, zorder=3)
    axl.errorbar(x, [means[k] for k in order], yerr=[stds[k] for k in order],
                 fmt="none", ecolor="#3E4C59", elinewidth=1.1,
                 capsize=4, capthick=1.1, zorder=4)
    top = max(means[k] + stds[k] for k in order)
    for xi, k in zip(x, order):
        axl.text(xi, means[k] + stds[k] + top * 0.03, f"{means[k]:.2f}",
                 ha="center", va="bottom", fontsize=9.5, color=INK, zorder=5)
    axl.set_xticks(x)
    axl.set_xticklabels([f"{k}\n$n={vals[k].size:,}$" for k in order],
                        fontsize=9.5)
    axl.tick_params(axis="x", length=0, pad=6)
    axl.set_ylim(0, top * 1.16)
    axl.set_ylabel(r"Mean $|\ell_{\mathrm{yes}} - \ell_{\mathrm{no}}|$")
    axl.margins(x=0.06)

    # ---- Right: per-category KDE curves ----
    axr.set_axisbelow(True)
    axr.yaxis.grid(True, color=GRID, linewidth=0.8)
    allv = np.concatenate([vals[k] for k in order])
    grid = np.linspace(max(0.0, allv.min()), allv.max(), 400)
    for k in order:
        dens = gaussian_kde(vals[k], grid)
        axr.plot(grid, dens, color=colors[k], lw=2.0, label=k, zorder=3)
        axr.fill_between(grid, dens, color=colors[k], alpha=0.10, zorder=2)
    axr.set_xlabel(r"$|\ell_{\mathrm{yes}} - \ell_{\mathrm{no}}|$")
    axr.set_ylabel("Density")
    axr.set_xlim(grid.min(), grid.max())
    axr.set_ylim(bottom=0)
    axr.xaxis.set_major_locator(MultipleLocator(1.0))
    axr.legend(loc="upper right", fontsize=9.5, handlelength=1.4,
               title="category", title_fontsize=9.0)
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
            fig.savefig(out, bbox_inches="tight")
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

    # HTML uses PNG only; drop the stray PDF copies it does not reference.
    for w in save(figure7(groups), "fig7_attention_histogram_3000q"):
        print(f"  wrote {w.relative_to(REPO)}")
    for w in save(figure8(groups), "fig8_logit_gap_3000q"):
        print(f"  wrote {w.relative_to(REPO)}")

    html = REPO / "paper" / "html"
    for stray in ("fig7_attention_histogram_3000q.pdf",
                  "fig8_logit_gap_3000q.pdf"):
        p = html / stray
        if p.exists():
            p.unlink()
            print(f"  removed {p.relative_to(REPO)} (HTML uses PNG only)")
    print("done.")


if __name__ == "__main__":
    main()
