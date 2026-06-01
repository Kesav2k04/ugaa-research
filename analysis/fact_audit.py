"""
fact_audit.py
=============

Verify every numeric claim in the paper against the source JSON files in
../experiments/ and run a strict final submission gate on the LaTeX
source. Prints OK per row; any mismatch indicates a false-data risk.

The audit has three phases:

  Phase 1  Paper-claim audit: every paper-side number is compared
           against the JSON-derived ground truth.

  Phase 2  Inference-time correction audit: each method's reported F1,
           precision, and recall are compared against the per-method
           summary JSON.

  Phase 3  Submission gate (strict): scans paper/arxiv/main.tex and
           paper/neurips/main.tex for placeholders, TODO/FIXME markers,
           anonymous-author text in the arXiv version, em dashes,
           corrupted characters, and Table 7 confusion-matrix values
           that do not match the JSON.

  Phase 4  Cross-model audit (if artifact present): verifies that the
           500-question prompt-template table in the Multi-Model
           Validation section matches the JSON values in
           experiments/cross_model for both prompt templates and all
           four readouts.

  Phase 4b Multi-model master-table audit (if recorded results present):
           re-derives the LLaVA-1.6 block from the real 3000q
           per-question JSONs, and checks the headline claims (legacy
           8-token collapse to F1 0 on disjoint vocabularies, dynamic
           readout in [0.80, 0.92], and the InstructBLIP string-parse
           divergence) against experiments/multi_model/multi_model_results.json.

  Phase 5  String-parse equivalence (Theorem section): verifies that
           on every LLaVA-1.5 question across all three splits, the
           first-token argmax is in the eight-token set (otherwise the
           theorem's hypothesis fails and the paper claim is wrong).
           Also verifies pred_dynamic_single == pred_string_parse on
           every LLaVA-1.6 cross-model question.

  Phase 6  Practitioner snippet: verifies that the drop-in code in
           Listing 1 of both papers is syntactically valid Python.

Exit code is 0 if every check passes; otherwise non-zero.

Run from the repo root:

  python analysis/fact_audit.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
PAPER_ARXIV = REPO / "paper" / "arxiv" / "main.tex"
PAPER_NEURIPS = REPO / "paper" / "neurips" / "main.tex"

# ---------------------------------------------------------------------------
# Phase 0: load JSON ground truth and compute confusion matrices directly
# from the per-question prediction records.
# ---------------------------------------------------------------------------

def _confusion(records, pred_key):
    tp = tn = fp = fn = 0
    for r in records:
        l = r["label"]
        p = r[pred_key]
        if   p == "yes" and l == "yes": tp += 1
        elif p == "no"  and l == "no":  tn += 1
        elif p == "yes" and l == "no":  fp += 1
        elif p == "no"  and l == "yes": fn += 1
    return tp, tn, fp, fn


def _load_split(split):
    with open(EXP / f"pope_{split}_2tok_vs_8tok.json") as f:
        return json.load(f)


def _f1_from_cm(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(f1, 4), round(p, 4), round(r, 4)


def _phase1_paper_claim_audit():
    """Phase 1: every numeric claim used in the paper, against JSON truth."""
    splits = {s: _load_split(s) for s in ("adversarial", "popular", "random")}

    truths = {}
    for s, d in splits.items():
        recs = d["results"]
        tp8, tn8, fp8, fn8 = _confusion(recs, "pred_8tok")
        tp2, tn2, fp2, fn2 = _confusion(recs, "pred_2tok")
        f18, p8, r8 = _f1_from_cm(tp8, fp8, fn8)
        f12, p2, r2 = _f1_from_cm(tp2, fp2, fn2)
        yr8 = round(sum(1 for x in recs if x["pred_8tok"] == "yes") / len(recs), 4)
        yr2 = round(sum(1 for x in recs if x["pred_2tok"] == "yes") / len(recs), 4)
        truths[s] = dict(
            tp8=tp8, tn8=tn8, fp8=fp8, fn8=fn8, f18=f18, p8=p8, r8=r8, yr8=yr8,
            tp2=tp2, tn2=tn2, fp2=fp2, fn2=fn2, f12=f12, p2=p2, r2=r2, yr2=yr2,
            n=len(recs),
        )

    adv, pop, rnd = truths["adversarial"], truths["popular"], truths["random"]

    # Each claim: (label, paper_value, json_truth_value, tol)
    claims = [
        # adversarial 8tok
        ("adv 8tok F1",       0.8221, adv["f1"]   if False else adv["f18"], 5e-4),
        ("adv 8tok P",        0.8658, adv["p8"],  5e-4),
        ("adv 8tok R",        0.7827, adv["r8"],  5e-4),
        ("adv 8tok yes-rate", 0.452,  adv["yr8"], 1.5e-3),
        # adversarial 2tok
        ("adv 2tok F1",       0.7608, adv["f12"], 5e-4),
        ("adv 2tok P",        0.6307, adv["p2"],  5e-4),
        ("adv 2tok R",        0.9587, adv["r2"],  5e-4),
        ("adv 2tok yes-rate", 0.760,  adv["yr2"], 1.5e-3),
        ("adv F1 gap",        0.0613, round(adv["f18"] - adv["f12"], 4), 5e-4),
        # popular
        ("pop 8tok F1",       0.8498, pop["f18"], 5e-4),
        ("pop 8tok P",        0.9140, pop["p8"],  5e-4),
        ("pop 8tok R",        0.7940, pop["r8"],  5e-4),
        ("pop 2tok F1",       0.7961, pop["f12"], 5e-4),
        ("pop 2tok P",        0.6797, pop["p2"],  5e-4),
        ("pop 2tok R",        0.9607, pop["r2"],  5e-4),
        ("pop F1 gap",        0.0537, round(pop["f18"] - pop["f12"], 4), 5e-4),
        # random
        ("rnd 8tok F1",       0.8713, rnd["f18"], 5e-4),
        ("rnd 8tok P",        0.9652, rnd["p8"],  5e-4),
        ("rnd 8tok R",        0.7940, rnd["r8"],  5e-4),
        ("rnd 2tok F1",       0.8397, rnd["f12"], 5e-4),
        ("rnd 2tok P",        0.7459, rnd["p2"],  5e-4),
        ("rnd 2tok R",        0.9607, rnd["r2"],  5e-4),
        ("rnd F1 gap",        0.0316, round(rnd["f18"] - rnd["f12"], 4), 5e-4),
        # Confusion matrix integers (8tok)
        ("adv TP", 1174, adv["tp8"], 0),
        ("adv TN", 1318, adv["tn8"], 0),
        ("adv FP",  182, adv["fp8"], 0),
        ("adv FN",  326, adv["fn8"], 0),
        ("pop TP", 1191, pop["tp8"], 0),
        ("pop TN", 1388, pop["tn8"], 0),
        ("pop FP",  112, pop["fp8"], 0),
        ("pop FN",  309, pop["fn8"], 0),
        ("rnd TP", 1191, rnd["tp8"], 0),
        ("rnd TN", 1457, rnd["tn8"], 0),
        ("rnd FP",   43, rnd["fp8"], 0),
        ("rnd FN",  309, rnd["fn8"], 0),
        # split sizes
        ("adv n",  3000, adv["n"], 0),
        ("pop n",  3000, pop["n"], 0),
        ("rnd n",  3000, rnd["n"], 0),
        # Random FP rate (text claim "43/1500 = 2.9%")
        ("rnd FP rate %", 2.9, round(100 * rnd["fp8"] / 1500, 1), 0.1),
        # Adversarial/random FP ratio
        ("adv/rnd FP ratio", 4.2, round(adv["fp8"] / max(rnd["fp8"], 1), 1), 0.2),
        # Token counts (Table 2)
        ("adv tok3869", 1356, sum(1 for x in splits["adversarial"]["results"]
                                   if x["top_token_id"] == 3869), 0),
        ("adv tok1939", 1644, sum(1 for x in splits["adversarial"]["results"]
                                   if x["top_token_id"] == 1939), 0),
        ("pop tok3869", 1303, sum(1 for x in splits["popular"]["results"]
                                   if x["top_token_id"] == 3869), 0),
        ("pop tok1939", 1697, sum(1 for x in splits["popular"]["results"]
                                   if x["top_token_id"] == 1939), 0),
        ("rnd tok3869", 1234, sum(1 for x in splits["random"]["results"]
                                   if x["top_token_id"] == 3869), 0),
        ("rnd tok1939", 1766, sum(1 for x in splits["random"]["results"]
                                   if x["top_token_id"] == 1939), 0),
    ]

    print("Phase 1: Paper-claim audit (paper-value vs. JSON-truth)")
    print("-" * 78)
    all_ok = True
    for name, paper_val, truth_val, tol in claims:
        if isinstance(paper_val, int):
            ok = (paper_val == truth_val)
            delta = abs(paper_val - truth_val)
        else:
            delta = abs(paper_val - truth_val)
            ok = delta <= tol + 1e-9
        flag = "OK " if ok else "BAD"
        print(f"  [{flag}] {name:22s} paper={str(paper_val):<8s} "
              f"truth={str(truth_val):<8s} delta={delta:.4f}")
        if not ok:
            all_ok = False
    print("-" * 78)
    return all_ok


# ---------------------------------------------------------------------------
# Phase 2: inference-time correction summaries
# ---------------------------------------------------------------------------

def _phase2_correction_audit():
    expected = {
        "baseline":   ("baseline",  (0.8221, 0.8658, 0.7827)),
        "beta1.0":    ("beta1.0",   (0.8164, 0.9023, 0.7453)),
        "clip_b1.0":  ("clip_b1.0", (0.8171, 0.9072, 0.7433)),
        "clip_b1.5":  ("clip_b1.5", (0.8104, 0.9278, 0.7193)),
    }
    print()
    print("Phase 2: Inference-time correction audit")
    print("-" * 78)
    all_ok = True
    for label, (tag, paper) in expected.items():
        p = EXP / f"pope_full_adversarial_{tag}_summary.json"
        if not p.exists():
            print(f"  [SKIP] {label:18s} file not found: {p.name}")
            continue
        with open(p) as f:
            s = json.load(f)
        truth = (s["f1"], s["precision"], s["recall"])
        deltas = [abs(p_ - t) for p_, t in zip(paper, truth)]
        ok = all(d < 5e-4 for d in deltas)
        flag = "OK " if ok else "BAD"
        print(f"  [{flag}] {label:18s} paper={paper}  truth={truth}")
        if not ok:
            all_ok = False
    print("-" * 78)
    return all_ok


# ---------------------------------------------------------------------------
# Phase 3: strict submission gate on the LaTeX source.
# ---------------------------------------------------------------------------

PLACEHOLDER_TOKENS = [
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bXXX+\b",
    r"\bplaceholder\b",
    r"\bTBA\b",
    r"\bTBD\b",
    r"\\todo\{",
]

ARXIV_ANONYMOUS_TOKENS = [
    "Anonymous Author",
    "[repository URL withheld",
    "withheld for double-blind",
    "Acknowledgments withheld",
]

EM_DASH = "—"          # —
REPLACEMENT_CHAR = "�"  #

ADV_ROW_RE = re.compile(
    r"Adversarial\s*&\s*1\{,\}174\s*&\s*1\{,\}318\s*&\s*182\s*&\s*326"
)
POP_ROW_RE = re.compile(
    r"Popular\s*&\s*1\{,\}191\s*&\s*1\{,\}388\s*&\s*112\s*&\s*309"
)
RND_ROW_RE = re.compile(
    r"Random\s*&\s*1\{,\}191\s*&\s*1\{,\}457\s*&\s*43\s*&\s*309"
)


def _scan_paper(path: Path, is_arxiv: bool):
    issues = []
    if not path.exists():
        return [f"missing source file: {path}"]
    text = path.read_text(encoding="utf-8", errors="replace")

    # Placeholders
    for pat in PLACEHOLDER_TOKENS:
        m = re.search(pat, text)
        if m:
            line = text[: m.start()].count("\n") + 1
            issues.append(f"placeholder match {pat!r} at line {line}")

    # arXiv must not contain anonymous-author or withheld text.
    if is_arxiv:
        for tok in ARXIV_ANONYMOUS_TOKENS:
            if tok in text:
                line = text[: text.find(tok)].count("\n") + 1
                issues.append(f"anonymous/withheld text {tok!r} at line {line}")

    # Em dash forbidden in both papers (per project style).
    if EM_DASH in text:
        line = text[: text.find(EM_DASH)].count("\n") + 1
        issues.append(f"em dash present at line {line}")

    # UTF-8 replacement char indicates corruption.
    if REPLACEMENT_CHAR in text:
        line = text[: text.find(REPLACEMENT_CHAR)].count("\n") + 1
        issues.append(f"replacement character (corrupted byte) at line {line}")

    # Table 7 numbers must match JSON truth.
    if not ADV_ROW_RE.search(text):
        issues.append("Table 7 adversarial row missing or mismatched (expected 1174 / 1318 / 182 / 326)")
    if not POP_ROW_RE.search(text):
        issues.append("Table 7 popular row missing or mismatched (expected 1191 / 1388 / 112 / 309)")
    if not RND_ROW_RE.search(text):
        issues.append("Table 7 random row missing or mismatched (expected 1191 / 1457 / 43 / 309)")

    return issues


def _phase3_submission_gate():
    print()
    print("Phase 3: Submission gate (strict)")
    print("-" * 78)
    arxiv_issues   = _scan_paper(PAPER_ARXIV,   is_arxiv=True)
    neurips_issues = _scan_paper(PAPER_NEURIPS, is_arxiv=False)

    all_ok = True
    for path, issues in (("arxiv", arxiv_issues), ("neurips", neurips_issues)):
        if not issues:
            print(f"  [OK ] {path:7s} : no placeholders, no forbidden tokens, "
                  f"Table 7 matches")
        else:
            all_ok = False
            print(f"  [BAD] {path:7s} :")
            for it in issues:
                print(f"          - {it}")
    print("-" * 78)
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 4: cross-model validation (LLaVA-1.6-Mistral-7B, 500q).
# ---------------------------------------------------------------------------

CROSS_DIR = REPO / "experiments" / "cross_model"
CROSS_BASE = "llava-hf_llava-v1.6-mistral-7b-hf_pope_adversarial_500q"

CROSS_EXPECTED = {
    "paper_template": {
        "tokenizer_class": "LlamaTokenizerFast",
        "dynamic_yes_ids": [5081, 5592],
        "dynamic_no_ids":  [708, 1770],
        "n_valid": 500,
        "n_errors": 0,
        "eval_legacy_2tok":     {"f1": 0.6667, "precision": 0.5000, "recall": 1.0000, "yes_rate": 1.0000},
        "eval_legacy_8tok":     {"f1": 0.0000, "precision": 0.0000, "recall": 0.0000, "yes_rate": 0.0000},
        "eval_dynamic_single":  {"f1": 0.8719, "precision": 0.8352, "recall": 0.9120, "yes_rate": 0.5460},
        "eval_string_parse":    {"f1": 0.8719, "precision": 0.8352, "recall": 0.9120, "yes_rate": 0.5460},
    },
    "native_chat_template": {
        "tokenizer_class": "LlamaTokenizerFast",
        "dynamic_yes_ids": [5081, 5592],
        "dynamic_no_ids":  [708, 1770],
        "n_valid": 500,
        "n_errors": 0,
        "eval_legacy_2tok":     {"f1": 0.6925, "precision": 0.5297, "recall": 1.0000, "yes_rate": 0.9440},
        "eval_legacy_8tok":     {"f1": 0.0000, "precision": 0.0000, "recall": 0.0000, "yes_rate": 0.0000},
        "eval_dynamic_single":  {"f1": 0.8621, "precision": 0.9517, "recall": 0.7880, "yes_rate": 0.4140},
        "eval_string_parse":    {"f1": 0.8621, "precision": 0.9517, "recall": 0.7880, "yes_rate": 0.4140},
    },
}


def _phase4_cross_model_audit():
    print()
    print("Phase 4: Cross-model validation (LLaVA-1.6-Mistral-7B, 500q)")
    print("-" * 78)

    found_any = False
    all_ok = True
    for mode, expected in CROSS_EXPECTED.items():
        path = CROSS_DIR / f"{CROSS_BASE}_{mode}_token_audit.json"
        if not path.exists():
            print(f"  [SKIP] {mode:22s} (artifact not present at {path.name})")
            continue
        found_any = True
        with open(path) as f:
            d = json.load(f)
        s = d["summary"]

        # Scalars
        scalar_checks = [
            ("tokenizer_class", expected["tokenizer_class"], s.get("tokenizer_class")),
            ("n_valid",         expected["n_valid"],         s.get("n_valid")),
            ("n_errors",        expected["n_errors"],        s.get("n_errors")),
            ("dynamic_yes_ids", expected["dynamic_yes_ids"], sorted(s.get("dynamic_yes_ids", []))),
            ("dynamic_no_ids",  expected["dynamic_no_ids"],  sorted(s.get("dynamic_no_ids",  []))),
        ]
        for name, exp, got in scalar_checks:
            ok = exp == got
            flag = "OK " if ok else "BAD"
            print(f"  [{flag}] {mode}/{name:18s} expected={exp} got={got}")
            if not ok:
                all_ok = False

        # Readouts
        for key in ("eval_legacy_2tok", "eval_legacy_8tok",
                    "eval_dynamic_single", "eval_string_parse"):
            exp = expected[key]
            got = s.get(key, {})
            for metric in ("f1", "precision", "recall", "yes_rate"):
                e = exp[metric]
                g = got.get(metric, float("nan"))
                ok = abs(e - g) <= 5e-4
                flag = "OK " if ok else "BAD"
                print(f"  [{flag}] {mode}/{key}/{metric:9s} expected={e} got={g}")
                if not ok:
                    all_ok = False

    print("-" * 78)
    if not found_any:
        print("  Phase 4 has no artifacts to check; skipping cross-model audit.")
        return True
    return all_ok


# ---------------------------------------------------------------------------
# Phase 4b: multi-model validation (Section "Multi-Model Validation").
# ---------------------------------------------------------------------------

MULTI_MODEL_JSON = REPO / "experiments" / "multi_model" / "multi_model_results.json"
LLAVA16_3000Q_TMPL = str(REPO / "experiments" / "cross_model" /
                         "llava16_mistral_pope_{split}_3000q_paper_template_token_audit.json")


def _phase4b_multimodel_audit():
    """Phase 4b: verify the multi-model master table against the recorded
    results file, independently re-deriving the LLaVA-1.6 block from the
    real per-question JSONs, and confirming the headline claims (legacy
    8-token collapse, dynamic-readout band, InstructBLIP string-parse
    divergence). Skips cleanly if the recorded file is absent."""
    print()
    print("Phase 4b: Multi-model validation (Multi-Model Validation section)")
    print("-" * 78)
    if not MULTI_MODEL_JSON.exists():
        print(f"  no recorded results file ({MULTI_MODEL_JSON.name}); skipping.")
        return True

    with open(MULTI_MODEL_JSON) as f:
        rec = json.load(f)["results"]
    all_ok = True

    # 4b.1 Re-derive the LLaVA-1.6 block from the real per-question JSONs.
    for split in ("adversarial", "popular", "random"):
        path = Path(LLAVA16_3000Q_TMPL.format(split=split))
        if not path.exists():
            print(f"  [SKIP] llava16 {split:11s} per-question JSON not present")
            continue
        with open(path) as f:
            s = json.load(f)["summary"]
        exp = rec["llava16_mistral"]["splits"][split]
        for key in ("eval_legacy_2tok", "eval_legacy_8tok",
                    "eval_dynamic_single", "eval_string_parse"):
            got = s.get(key, {}).get("f1")
            want = exp[key]["f1"]
            ok = got is not None and abs(got - want) <= 5e-4
            flag = "OK " if ok else "BAD"
            print(f"  [{flag}] llava16 {split:11s} {key:20s} record={want} json={got}")
            all_ok = all_ok and ok

    # 4b.2 legacy_8tok collapses to F1 0 on the disjoint-vocabulary models.
    for model in ("llava16_mistral", "qwen2_vl"):
        for split, blocks in rec[model]["splits"].items():
            f1 = blocks["eval_legacy_8tok"]["f1"]
            ok = f1 == 0.0
            print(f"  [{'OK ' if ok else 'BAD'}] {model:16s} {split:11s} legacy_8tok F1==0  got={f1}")
            all_ok = all_ok and ok

    # 4b.3 dynamic_single stays in [0.80, 0.92] on every model and split.
    for model, blob in rec.items():
        for split, blocks in blob["splits"].items():
            f1 = blocks["eval_dynamic_single"]["f1"]
            ok = 0.80 <= f1 <= 0.92
            print(f"  [{'OK ' if ok else 'BAD'}] {model:16s} {split:11s} dynamic in[0.80,0.92] got={f1}")
            all_ok = all_ok and ok

    # 4b.4 string_parse == dynamic_single on three models; diverges on
    # InstructBLIP (collapses to the 0.6667 all-yes floor).
    for model in ("llava16_mistral", "mplug_owl2", "qwen2_vl"):
        for split, blocks in rec[model]["splits"].items():
            d = blocks["eval_dynamic_single"]["f1"]
            sp = blocks["eval_string_parse"]["f1"]
            ok = abs(d - sp) <= 5e-4
            print(f"  [{'OK ' if ok else 'BAD'}] {model:16s} {split:11s} string==dynamic  d={d} sp={sp}")
            all_ok = all_ok and ok
    for split, blocks in rec["instructblip"]["splits"].items():
        d = blocks["eval_dynamic_single"]["f1"]
        sp = blocks["eval_string_parse"]["f1"]
        ok = (abs(sp - 0.6667) <= 5e-4) and ((d - sp) > 0.05)
        print(f"  [{'OK ' if ok else 'BAD'}] instructblip {split:11s} string diverges -> 0.6667  d={d} sp={sp}")
        all_ok = all_ok and ok

    print("-" * 78)
    return all_ok


# ---------------------------------------------------------------------------
# Phase 5: string-parse / single-token-readout equivalence (Theorem section).
# ---------------------------------------------------------------------------

LLAVA15_DYNAMIC = {3582, 8241, 4874, 3869, 1217, 3782, 694, 1939}


def _phase5_equivalence_audit():
    print()
    print("Phase 5: String-parse / single-token equivalence (Theorem section)")
    print("-" * 78)

    all_ok = True

    # 5a. LLaVA-1.5: every top_token_id must be in the eight-token set,
    # else the paper's "9000/9000" claim is false.
    total_in = total = 0
    for split in ("adversarial", "popular", "random"):
        path = EXP / f"pope_{split}_2tok_vs_8tok.json"
        if not path.exists():
            print(f"  [SKIP] {split} JSON not found at {path.name}")
            continue
        with open(path) as f:
            d = json.load(f)
        rows = d["results"]
        in_set = sum(1 for r in rows if r["top_token_id"] in LLAVA15_DYNAMIC)
        total_in += in_set; total += len(rows)
        ok = in_set == len(rows)
        flag = "OK " if ok else "BAD"
        print(f"  [{flag}] LLaVA-1.5 {split:12s} in-set {in_set}/{len(rows)}")
        if not ok:
            all_ok = False
    ok_overall = total_in == total and total > 0
    if total > 0:
        flag = "OK " if ok_overall else "BAD"
        print(f"  [{flag}] LLaVA-1.5 overall {total_in}/{total}")
        if not ok_overall:
            all_ok = False

    # 5b. LLaVA-1.6 cross-model: pred_dynamic_single == pred_string_parse
    # on every question in both prompt modes.
    cross = REPO / "experiments" / "cross_model"
    base = "llava-hf_llava-v1.6-mistral-7b-hf_pope_adversarial_500q"
    for mode in ("paper_template", "native_chat_template"):
        path = cross / f"{base}_{mode}_token_audit.json"
        if not path.exists():
            print(f"  [SKIP] LLaVA-1.6 {mode} JSON not found")
            continue
        with open(path) as f:
            d = json.load(f)
        rows = [r for r in d["results"] if "pred_string_parse" in r]
        agree = sum(1 for r in rows
                    if r["pred_dynamic_single"] == r["pred_string_parse"])
        ok = agree == len(rows) and len(rows) > 0
        flag = "OK " if ok else "BAD"
        print(f"  [{flag}] LLaVA-1.6 {mode:22s} agreement {agree}/{len(rows)}")
        if not ok:
            all_ok = False

    # 5c. LLaVA-1.6 3000q full-split runs: same per-question agreement.
    for split in ("adversarial", "popular", "random"):
        path = Path(LLAVA16_3000Q_TMPL.format(split=split))
        if not path.exists():
            print(f"  [SKIP] LLaVA-1.6 3000q {split} JSON not found")
            continue
        with open(path) as f:
            d = json.load(f)
        rows = [r for r in d["results"] if "pred_string_parse" in r]
        agree = sum(1 for r in rows
                    if r["pred_dynamic_single"] == r["pred_string_parse"])
        ok = agree == len(rows) and len(rows) > 0
        print(f"  [{'OK ' if ok else 'BAD'}] LLaVA-1.6 3000q {split:11s} agreement {agree}/{len(rows)}")
        if not ok:
            all_ok = False

    print("-" * 78)
    return all_ok


# ---------------------------------------------------------------------------
# Phase 6: practitioner-snippet syntax (Section "Practitioner Recommendations").
# ---------------------------------------------------------------------------

SNIPPET = """\
tok = processor.tokenizer
enc = lambda s: tok.encode(s, add_special_tokens=False)
yes_ids = [enc(s)[0] for s in ["yes", "Yes", " yes", " Yes"] if len(enc(s)) == 1]
no_ids  = [enc(s)[0] for s in ["no", "No", " no", " No"]   if len(enc(s)) == 1]
out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                     return_dict_in_generate=True, output_scores=True)
logit = out.scores[0][0]
pred = "yes" if max(logit[i] for i in yes_ids) > max(logit[j] for j in no_ids) else "no"
"""


def _phase6_snippet_audit():
    import ast
    print()
    print("Phase 6: Practitioner snippet syntax (Listing in both papers)")
    print("-" * 78)
    try:
        ast.parse(SNIPPET)
        print("  [OK ] snippet parses as valid Python")
        ok = True
    except SyntaxError as exc:
        print(f"  [BAD] snippet parse error: {exc}")
        ok = False

    # also confirm the same code text appears in both papers
    for label, path in (("arxiv", PAPER_ARXIV),
                        ("neurips", PAPER_NEURIPS)):
        text = path.read_text(encoding="utf-8")
        has_snippet = "out = model.generate(**inputs, max_new_tokens=1" in text
        flag = "OK " if has_snippet else "BAD"
        print(f"  [{flag}] {label}: drop-in snippet present in main.tex")
        if not has_snippet:
            ok = False
    print("-" * 78)
    return ok


def main():
    ok1 = _phase1_paper_claim_audit()
    ok2 = _phase2_correction_audit()
    ok3 = _phase3_submission_gate()
    ok4 = _phase4_cross_model_audit()
    ok4b = _phase4b_multimodel_audit()
    ok5 = _phase5_equivalence_audit()
    ok6 = _phase6_snippet_audit()
    all_ok = ok1 and ok2 and ok3 and ok4 and ok4b and ok5 and ok6
    print()
    print("ALL FACTS VERIFIED" if all_ok else "FACT MISMATCH - DO NOT SUBMIT")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
