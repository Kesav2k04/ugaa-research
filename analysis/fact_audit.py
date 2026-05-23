"""
Audit every numeric claim in the paper against the source JSON files in
../experiments/. Prints OK per row; any mismatch indicates a
false-data risk. Run from the repo root or from this folder.
"""
import json, os
from pathlib import Path

EXP = str(Path(__file__).resolve().parent.parent / "experiments")

# Load ground truth
with open(f"{EXP}/pope_full_adversarial_baseline_summary.json") as f:
    adv = json.load(f)
with open(f"{EXP}/pope_adversarial_2tok_vs_8tok.json") as f:
    adv_pp = json.load(f)["summary"]
with open(f"{EXP}/pope_popular_2tok_vs_8tok.json") as f:
    pop_pp = json.load(f)["summary"]
with open(f"{EXP}/pope_random_2tok_vs_8tok.json") as f:
    rnd_pp = json.load(f)["summary"]

claims = {
    "adv 8tok F1":       (0.8221, adv_pp["eval_8token"]["f1"]),
    "adv 8tok P":        (0.8658, adv_pp["eval_8token"]["precision"]),
    "adv 8tok R":        (0.7827, adv_pp["eval_8token"]["recall"]),
    "adv 8tok yes-rate": (0.452,  adv_pp["eval_8token"]["yes_rate"]),
    "adv 2tok F1":       (0.7608, adv_pp["eval_2token"]["f1"]),
    "adv 2tok P":        (0.6307, adv_pp["eval_2token"]["precision"]),
    "adv 2tok R":        (0.9587, adv_pp["eval_2token"]["recall"]),
    "adv 2tok yes-rate": (0.760,  adv_pp["eval_2token"]["yes_rate"]),
    "adv gap":           (0.0613, adv_pp["f1_gap_8tok_minus_2tok"]),
    "pop 8tok F1":       (0.8498, pop_pp["eval_8token"]["f1"]),
    "pop 8tok P":        (0.9140, pop_pp["eval_8token"]["precision"]),
    "pop 8tok R":        (0.7940, pop_pp["eval_8token"]["recall"]),
    "pop 2tok F1":       (0.7961, pop_pp["eval_2token"]["f1"]),
    "pop 2tok P":        (0.6797, pop_pp["eval_2token"]["precision"]),
    "pop 2tok R":        (0.9607, pop_pp["eval_2token"]["recall"]),
    "pop gap":           (0.0537, pop_pp["f1_gap_8tok_minus_2tok"]),
    "rnd 8tok F1":       (0.8713, rnd_pp["eval_8token"]["f1"]),
    "rnd 8tok P":        (0.9652, rnd_pp["eval_8token"]["precision"]),
    "rnd 8tok R":        (0.7940, rnd_pp["eval_8token"]["recall"]),
    "rnd 2tok F1":       (0.8397, rnd_pp["eval_2token"]["f1"]),
    "rnd 2tok P":        (0.7459, rnd_pp["eval_2token"]["precision"]),
    "rnd 2tok R":        (0.9607, rnd_pp["eval_2token"]["recall"]),
    "rnd gap":           (0.0316, rnd_pp["f1_gap_8tok_minus_2tok"]),
    "TP adv":            (1174,   adv["tp"]),
    "TN adv":            (1318,   adv["tn"]),
    "FP adv":            (182,    adv["fp"]),
    "FN adv":            (326,    adv["fn"]),
    "n adv":             (3000,   adv["n_total"]),
}

print("Paper-claim audit (paper-value vs. JSON-truth):")
print("-" * 70)
all_ok = True
for name, (paper_val, truth_val) in claims.items():
    delta = abs(paper_val - truth_val)
    if isinstance(paper_val, int):
        ok = paper_val == truth_val
    else:
        ok = delta < 0.0005
    flag = "OK " if ok else "BAD"
    print(f"  [{flag}] {name:25s} paper={paper_val:<9} truth={truth_val:<9} delta={delta:.4f}")
    if not ok:
        all_ok = False
print("-" * 70)

# Inference-time correction results
files = {
    "CLIP-B/32 b=1.0":  ("clip_b1.0",   (0.8171, 0.9072, 0.7433)),
    "CLIP-B/32 b=1.5":  ("clip_b1.5",   (0.8104, 0.9278, 0.7193)),
    "UGAA beta=1.0":    ("beta1.0",     (0.8164, 0.9023, 0.7453)),
    "baseline":         ("baseline",    (0.8221, 0.8658, 0.7827)),
}
print()
print("Inference-time correction audit:")
print("-" * 70)
for label, (tag, expected) in files.items():
    p = f"{EXP}/pope_full_adversarial_{tag}_summary.json"
    if not os.path.exists(p):
        print(f"  [SKIP] {label}: file not found ({p})")
        continue
    with open(p) as f:
        s = json.load(f)
    truth = (s["f1"], s["precision"], s["recall"])
    paper = expected
    deltas = [abs(p_-t) for p_, t in zip(paper, truth)]
    ok = all(d < 0.0005 for d in deltas)
    flag = "OK " if ok else "BAD"
    print(f"  [{flag}] {label:18s} paper={paper}  truth={truth}")
    if not ok:
        all_ok = False
print("-" * 70)
print()
print("ALL FACTS VERIFIED" if all_ok else "FACT MISMATCH - DO NOT SUBMIT")
