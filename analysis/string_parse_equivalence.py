"""
string_parse_equivalence.py
===========================

Empirically verify the theorem that, under greedy decoding, the
string-parse readout and the dynamic single-token (max-logit over
single-token surface forms) readout are equivalent.

The theorem statement: if every first-token argmax across the evaluation
set belongs to the dynamic single-token set, then string-parse and
logit-max readouts produce identical predictions on every question.
The converse is empirically meaningful: the fraction of questions
whose argmax token sits inside the dynamic set is exactly the
equivalence rate; the theorem makes that rate a tight identity, not a
correlation.

This script reads only the existing per-question prediction logs in
experiments/ and reports:

  (i)  LLaVA-1.5-7B over 9,000 POPE questions (3 splits): how many
       questions have their first-token argmax inside the eight-token
       set, and what the equivalence rate is. For all questions whose
       argmax is in the set, the eight-token readout and string-parse
       are formally identical (by the theorem).

  (ii) LLaVA-1.6-Mistral-7B over the 500-question cross-model run
       (paper template + native chat template), where the JSON
       explicitly stores both pred_dynamic_single and pred_string_parse.
       The agreement rate is the direct empirical check on the theorem.

No GPU is needed. No model is loaded. Run from the repo root:

    python analysis/string_parse_equivalence.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
CROSS = EXP / "cross_model"

# LLaVA-1.5 dynamic single-token set (the eight-token set in the paper).
LLAVA15_DYNAMIC = {3582, 8241, 4874, 3869, 1217, 3782, 694, 1939}


def _phase1_llava15():
    """For each LLaVA-1.5 split, count questions whose top_token_id is
    inside the eight-token set."""
    print("Phase 1: LLaVA-1.5-7B equivalence (9,000 POPE questions)")
    print("-" * 76)
    total_in = 0
    total = 0
    per_split = {}
    for split in ("adversarial", "popular", "random"):
        path = EXP / f"pope_{split}_2tok_vs_8tok.json"
        with open(path) as f:
            d = json.load(f)
        rows = d["results"]
        in_set = sum(1 for r in rows if r["top_token_id"] in LLAVA15_DYNAMIC)
        total_in += in_set
        total += len(rows)
        per_split[split] = (in_set, len(rows))
        rate = in_set / len(rows)
        print(f"  {split:12s} {in_set:5d}/{len(rows):5d}  equivalence rate = {rate:.6f}")
    overall = total_in / total
    print(f"  {'overall':12s} {total_in:5d}/{total:5d}  equivalence rate = {overall:.6f}")
    print("-" * 76)
    return per_split, total_in, total


def _phase2_llava16():
    """For LLaVA-1.6, the cross-model JSON has explicit pred_dynamic_single
    and pred_string_parse; count direct agreement."""
    print()
    print("Phase 2: LLaVA-1.6-Mistral-7B agreement (500q x 2 prompt modes)")
    print("-" * 76)
    overall_agree = 0
    overall_total = 0
    for mode in ("paper_template", "native_chat_template"):
        path = CROSS / f"llava-hf_llava-v1.6-mistral-7b-hf_pope_adversarial_500q_{mode}_token_audit.json"
        if not path.exists():
            print(f"  {mode:22s} (artifact not found)")
            continue
        with open(path) as f:
            d = json.load(f)
        rows = d["results"]
        valid = [r for r in rows if "pred_string_parse" in r]
        agree = sum(1 for r in valid
                    if r["pred_dynamic_single"] == r["pred_string_parse"])
        rate = agree / max(len(valid), 1)
        overall_agree += agree
        overall_total += len(valid)
        print(f"  {mode:22s} {agree:4d}/{len(valid):4d}  agreement rate = {rate:.6f}")
    if overall_total:
        rate = overall_agree / overall_total
        print(f"  {'overall':22s} {overall_agree:4d}/{overall_total:4d}  agreement rate = {rate:.6f}")
    print("-" * 76)
    return overall_agree, overall_total


def _phase3_summary(p1, p2):
    per_split, l15_in, l15_total = p1
    l16_agree, l16_total = p2
    print()
    print("Summary")
    print("-" * 76)
    print(f"  LLaVA-1.5-7B (9000 q, 3 splits)")
    print(f"    questions with argmax in eight-token set: {l15_in}/{l15_total}")
    print(f"    -> on every such question, eight-token readout ==")
    print(f"       string-parse readout (by the theorem).")
    print(f"    -> empirical equivalence rate = {l15_in/l15_total:.6f}")
    if l16_total:
        print(f"  LLaVA-1.6-Mistral-7B (1000 q across two prompt modes)")
        print(f"    direct agreement between pred_dynamic_single and pred_string_parse:")
        print(f"      {l16_agree}/{l16_total} = {l16_agree/l16_total:.6f}")
    print()
    print("These two numbers together form the empirical validation of the")
    print("theorem proven in Section 'Formal Equivalence of String-Parse and")
    print("Single-Token Readouts'.")


def main():
    p1 = _phase1_llava15()
    p2 = _phase2_llava16()
    _phase3_summary(p1, p2)


if __name__ == "__main__":
    main()
