"""
diagnostic_stats.py
===================

Inferential statistics for the diagnostic JSON produced by the v6 100q
validation run (and, when present, the full-3000q diagnostic produced
by src/run_full_diagnostic_3000q.py).

For each split of the diagnostic data (correct vs wrong, TP vs FP,
TP vs FN, TN vs FP), this script reports:

  - n, mean, median, std for each group
  - 95% bootstrap confidence interval on the difference of means
  - Cohen's d effect size with its sign
  - two-sided Mann-Whitney U with the p-value

Bootstrap and Cohen's d are computed in pure numpy; Mann-Whitney
uses scipy when available and falls back to a pure-python normal-
approximation otherwise. No GPU is used. Run from the repo root:

    python analysis/diagnostic_stats.py
    python analysis/diagnostic_stats.py --diagnostic experiments/ugaa_full_3000q_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

DEFAULT_DIAG = REPO / "experiments" / "ugaa_v6_100q_validation.json"


# ---------------------------------------------------------------------------
# Mann-Whitney U
# ---------------------------------------------------------------------------

def _mann_whitney_scipy(a, b):
    from scipy.stats import mannwhitneyu  # type: ignore
    res = mannwhitneyu(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def _mann_whitney_pure(a, b):
    na, nb = len(a), len(b)
    if na < 5 or nb < 5:
        return float("nan"), float("nan")
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_a = sum(ranks[k] for k, (_, g) in enumerate(combined) if g == 0)
    U_a = rank_a - na * (na + 1) / 2.0
    U_b = na * nb - U_a
    U = min(U_a, U_b)
    mu = na * nb / 2.0
    sigma = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    if sigma == 0:
        return float(U), 1.0
    z = (U - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(U), float(p)


def mann_whitney_u(a, b):
    try:
        return _mann_whitney_scipy(a, b)
    except Exception:
        return _mann_whitney_pure(a, b)


# ---------------------------------------------------------------------------
# Effect size and bootstrap CI
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    """Pooled-variance Cohen's d. Sign is positive when mean(a) > mean(b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    s = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) if (na + nb - 2) > 0 else float("nan")
    if not s or math.isnan(s):
        return float("nan")
    return float((a.mean() - b.mean()) / s)


def bootstrap_ci_diff(a, b, n_boot=10000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    na, nb = a.size, b.size
    for i in range(n_boot):
        sa = rng.choice(a, size=na, replace=True)
        sb = rng.choice(b, size=nb, replace=True)
        diffs[i] = sa.mean() - sb.mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = a.mean() - b.mean()
    return float(point), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Diagnostic JSON loader / categoriser
# ---------------------------------------------------------------------------

def load_categories(diag_path: Path):
    """Group per-question records by TP/TN/FP/FN/correct/wrong on the
    baseline prediction. Supports the v6 100q schema and the
    full-3000q schema (which uses the same field names)."""
    with open(diag_path) as f:
        diag = json.load(f)
    rows = diag.get("results") or diag.get("records") or []

    cats = {"correct": [], "wrong": [],
            "TP": [], "TN": [], "FP": [], "FN": []}
    for r in rows:
        pred = r.get("baseline_prediction") or r.get("prediction")
        label = r.get("label")
        if pred is None or label is None or pred == "error":
            continue
        ground = r.get("grounding")
        gap = r.get("gap_raw")
        clip_l = r.get("clip_l_max_sim")
        if ground is None or gap is None:
            continue
        record = {
            "grounding": float(ground),
            "gap_abs":   abs(float(gap)),
            "clip_l":    float(clip_l) if clip_l is not None else float("nan"),
            "label":     label,
            "pred":      pred,
        }
        if pred == label:
            cats["correct"].append(record)
        else:
            cats["wrong"].append(record)
        if pred == "yes" and label == "yes":
            cats["TP"].append(record)
        elif pred == "no"  and label == "no":
            cats["TN"].append(record)
        elif pred == "yes" and label == "no":
            cats["FP"].append(record)
        elif pred == "no"  and label == "yes":
            cats["FN"].append(record)
    return cats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _summary(values):
    if not values:
        return dict(n=0, mean=float("nan"), median=float("nan"), std=float("nan"))
    return dict(
        n=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def report_pair(name, group_a_name, a_rows, group_b_name, b_rows, field,
                n_boot=10000):
    a = [r[field] for r in a_rows if not math.isnan(r[field])]
    b = [r[field] for r in b_rows if not math.isnan(r[field])]
    sa = _summary(a)
    sb = _summary(b)
    if sa["n"] >= 2 and sb["n"] >= 2:
        diff, lo, hi = bootstrap_ci_diff(a, b, n_boot=n_boot)
        d = cohens_d(a, b)
    else:
        diff = lo = hi = d = float("nan")
    if sa["n"] >= 5 and sb["n"] >= 5:
        U, p = mann_whitney_u(a, b)
    else:
        U = p = float("nan")
    direction = (group_a_name if sa["median"] > sb["median"]
                 else group_b_name)
    print(f"== {name} | field={field} ==")
    print(f"  {group_a_name:10s} n={sa['n']:<4d} mean={sa['mean']:.6g}  median={sa['median']:.6g}  std={sa['std']:.6g}")
    print(f"  {group_b_name:10s} n={sb['n']:<4d} mean={sb['mean']:.6g}  median={sb['median']:.6g}  std={sb['std']:.6g}")
    if not math.isnan(diff):
        print(f"  diff = mean({group_a_name}) - mean({group_b_name}) = {diff:.6g}  "
              f"95%% bootstrap CI [{lo:.6g}, {hi:.6g}]")
        print(f"  Cohen's d = {d:.4f}  (sign: {direction} larger)")
    if not math.isnan(U):
        print(f"  Mann-Whitney U = {U:.2f}, two-sided p = {p:.4g}  (direction: higher in {direction})")
    else:
        print(f"  Mann-Whitney U: skipped (n_min < 5)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", default=str(DEFAULT_DIAG),
                        help="Path to diagnostic JSON (default: 100q).")
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="Number of bootstrap resamples (default: 10000).")
    args = parser.parse_args()

    diag_path = Path(args.diagnostic)
    if not diag_path.exists():
        print(f"ERROR: diagnostic JSON not found at {diag_path}")
        return

    cats = load_categories(diag_path)
    print(f"Diagnostic: {diag_path.relative_to(REPO) if str(diag_path).startswith(str(REPO)) else diag_path}")
    print(f"Bootstrap resamples: {args.n_boot}")
    print(f"Group sizes: TP={len(cats['TP'])}, TN={len(cats['TN'])}, "
          f"FP={len(cats['FP'])}, FN={len(cats['FN'])}, "
          f"correct={len(cats['correct'])}, wrong={len(cats['wrong'])}")
    print()

    report_pair("(A) correct vs wrong (image-attention)",
                "correct", cats["correct"], "wrong", cats["wrong"],
                field="grounding", n_boot=args.n_boot)
    report_pair("(B) TP vs FP (image-attention)",
                "TP", cats["TP"], "FP", cats["FP"],
                field="grounding", n_boot=args.n_boot)
    report_pair("(C) TN vs FN (image-attention)",
                "TN", cats["TN"], "FN", cats["FN"],
                field="grounding", n_boot=args.n_boot)

    report_pair("(D) correct vs wrong (|logit gap|)",
                "correct", cats["correct"], "wrong", cats["wrong"],
                field="gap_abs", n_boot=args.n_boot)
    report_pair("(E) TP vs FN (|logit gap|, confidence vs correctness)",
                "TP", cats["TP"], "FN", cats["FN"],
                field="gap_abs", n_boot=args.n_boot)
    report_pair("(F) TN vs FP (|logit gap|, confidence vs correctness)",
                "TN", cats["TN"], "FP", cats["FP"],
                field="gap_abs", n_boot=args.n_boot)

    has_clip = any(
        not math.isnan(r["clip_l"])
        for grp in cats.values()
        for r in grp
    )
    if has_clip:
        report_pair("(G) TP vs FP (CLIP-L max patch-noun similarity)",
                    "TP", cats["TP"], "FP", cats["FP"],
                    field="clip_l", n_boot=args.n_boot)
    else:
        print("CLIP-L per-question max similarity is not stored in this")
        print("diagnostic JSON. Rerun src/run_full_diagnostic_3000q.py to")
        print("produce it; that script writes per-question clip_l_max_sim.")


if __name__ == "__main__":
    main()
