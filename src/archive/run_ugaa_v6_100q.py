"""
UGAA v6 — 100-question validation gate on POPE adversarial.

Purpose: prove the gate operator does not regress baseline (F1=0.8221) on a
cheap sample BEFORE committing 30 minutes per split to the full 3000-q run.

Logs per-question diagnostics (grounding, complexity, zone, raw vs modified
yes/no maxes) so any regression is debuggable without re-running.

Run:  D:\\UGAA-MASTER\\ugaa_env\\python.exe src\\run_ugaa_v6_100q.py
"""

import json
import os
import sys
import time
import gc

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import compute_f1
from ugaa_gate_v6 import (
    UGAAGateV6,
    YES_TOKEN_IDS,
    NO_TOKEN_IDS,
    load_llava_with_gate,
)


DATA_PATH = "datasets/pope/pope_adversarial_full.json"
N_SAMPLES = 100
BASELINE_F1 = 0.8221  # POPE adversarial 8-token baseline, 3000q
OUT_PATH = "experiments/ugaa_v6_100q_validation.json"


@torch.no_grad()
def predict_with_gate(model, processor, gate, image, question, device):
    """Single forward pass; extract logits + per-layer attention; apply gate."""
    prompt = f"USER: <image>\n{question} Answer yes or no only.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    out = model(**inputs, output_attentions=True, return_dict=True)
    # Last-token logits = prediction for the NEXT token (yes/no).
    logits = out.logits[0, -1, :].float().clone()
    # Tuple of length n_layers, each (1, n_heads, seq, seq).
    attentions = out.attentions

    diag = gate.process(logits, attentions, inputs["input_ids"])

    # Aggressive cleanup — output_attentions is the dominant allocation.
    del out, logits, attentions
    return diag


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")
    with open(DATA_PATH) as f:
        items = json.load(f)[:N_SAMPLES]

    print(f"=== UGAA v6 100q Validation | POPE adversarial | n={len(items)} ===")
    print("Loading LLaVA (4-bit, eager attention)...")
    model, processor = load_llava_with_gate()
    device = str(model.device)
    print(f"Ready on {device}.\n")

    gate = UGAAGateV6(model)
    gate.register_hooks()
    print(
        f"Gate: alpha={gate.alpha}  beta={gate.beta}  "
        f"high={gate.high_thresh}  low={gate.low_thresh}  "
        f"conf_margin={gate.confidence_margin}  layers=({gate.layer_lo},{gate.layer_hi})\n"
    )

    results = []
    errors = 0
    t0 = time.time()

    for i, item in enumerate(items):
        try:
            image = Image.open(item["local_path"]).convert("RGB")
        except Exception as e:
            print(f"  [{i+1}] image-open failed: {e}")
            results.append({
                "question_id": item["question_id"],
                "label": item["label"],
                "prediction": "error",
            })
            errors += 1
            continue

        try:
            d = predict_with_gate(model, processor, gate, image, item["question"], device)
        except Exception as e:
            print(f"  [{i+1}] forward failed: {e}")
            results.append({
                "question_id": item["question_id"],
                "label": item["label"],
                "prediction": "error",
            })
            errors += 1
            continue

        results.append({
            "question_id": item["question_id"],
            "label": item["label"],
            "prediction": d["prediction"],
            "baseline_prediction": d["baseline_prediction"],
            "grounding": d["grounding"],
            "complexity": d["complexity"],
            "zone": d["zone"],
            "yes_raw": d["yes_raw"],
            "no_raw": d["no_raw"],
            "gap_raw": d["gap_raw"],
            "yes_mod": d["yes_mod"],
            "no_mod": d["no_mod"],
            "visual_span": d["visual_span"],
        })

        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            valid = [r for r in results if r["prediction"] != "error"]
            m = compute_f1([r["prediction"] for r in valid], [r["label"] for r in valid])
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(items)}] F1={m['f1']:.4f}  errors={errors}  "
                  f"{elapsed/60:.1f}m elapsed")

    gate.remove_hooks()

    # ---- Aggregate metrics ----
    valid = [r for r in results if r["prediction"] != "error"]
    preds = [r["prediction"] for r in valid]
    labels = [r["label"] for r in valid]
    m = compute_f1(preds, labels)

    base_preds = [r["baseline_prediction"] for r in valid]
    bm = compute_f1(base_preds, labels)

    # Zone distribution
    zone_counts = {"CONFIDENT": 0, "HIGH": 0, "LOW": 0, "NEUTRAL": 0}
    for r in valid:
        zone_counts[r["zone"]] = zone_counts.get(r["zone"], 0) + 1

    # Modifications (gate changed the answer?)
    flipped = sum(1 for r in valid if r["prediction"] != r["baseline_prediction"])
    flipped_to_yes = sum(1 for r in valid if r["prediction"] == "yes" and r["baseline_prediction"] == "no")
    flipped_to_no = sum(1 for r in valid if r["prediction"] == "no" and r["baseline_prediction"] == "yes")

    # Was the flip correct?
    flips_correct = sum(1 for r in valid
                        if r["prediction"] != r["baseline_prediction"]
                        and r["prediction"] == r["label"])
    flips_wrong = flipped - flips_correct

    # Score stats
    import statistics as st
    gs = [r["grounding"] for r in valid]
    cs = [r["complexity"] for r in valid]
    gap_abs = [abs(r["gap_raw"]) for r in valid]

    delta = m["f1"] - bm["f1"]
    delta_baseline_3000 = m["f1"] - BASELINE_F1

    summary = {
        "n_total": len(results),
        "n_valid": len(valid),
        "n_errors": errors,
        "gate_metrics": m,
        "ungated_metrics_same_100q": bm,
        "baseline_f1_3000q_reference": BASELINE_F1,
        "delta_vs_ungated_100q": round(delta, 4),
        "delta_vs_baseline_3000q": round(delta_baseline_3000, 4),
        "zone_counts": zone_counts,
        "flipped_total": flipped,
        "flipped_to_yes": flipped_to_yes,
        "flipped_to_no": flipped_to_no,
        "flips_correct": flips_correct,
        "flips_wrong": flips_wrong,
        "grounding_stats": {
            "min": round(min(gs), 4) if gs else 0.0,
            "max": round(max(gs), 4) if gs else 0.0,
            "mean": round(st.mean(gs), 4) if gs else 0.0,
            "median": round(st.median(gs), 4) if gs else 0.0,
        },
        "complexity_stats": {
            "min": round(min(cs), 4) if cs else 0.0,
            "max": round(max(cs), 4) if cs else 0.0,
            "mean": round(st.mean(cs), 4) if cs else 0.0,
        },
        "logit_gap_abs_stats": {
            "min": round(min(gap_abs), 4) if gap_abs else 0.0,
            "max": round(max(gap_abs), 4) if gap_abs else 0.0,
            "mean": round(st.mean(gap_abs), 4) if gap_abs else 0.0,
            "median": round(st.median(gap_abs), 4) if gap_abs else 0.0,
        },
        "gate_config": {
            "alpha": gate.alpha,
            "beta": gate.beta,
            "high_thresh": gate.high_thresh,
            "low_thresh": gate.low_thresh,
            "confidence_margin": gate.confidence_margin,
            "complexity_min": gate.complexity_min,
            "complexity_max": gate.complexity_max,
            "layers": (gate.layer_lo, gate.layer_hi),
        },
    }

    os.makedirs("experiments", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    # ---- Print final block ----
    print()
    print("=" * 65)
    print("UGAA V6 — 100q VALIDATION | POPE ADVERSARIAL")
    print("=" * 65)
    print(f"  Ungated on same 100q : F1={bm['f1']:.4f}  P={bm['precision']:.4f}  R={bm['recall']:.4f}")
    print(f"  UGAA V6  on same 100q: F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")
    print(f"  Delta vs ungated 100q : {delta:+.4f} F1")
    print(f"  3000q baseline ref    : F1=0.8221  (Delta {delta_baseline_3000:+.4f})")
    print()
    print(f"  Zone distribution    : {zone_counts}")
    print(f"  Predictions flipped  : {flipped} ({flipped_to_yes} to YES, {flipped_to_no} to NO)")
    print(f"  Flips correct        : {flips_correct} / {flipped}")
    print(f"  Flips wrong          : {flips_wrong}")
    print()
    print(f"  Grounding stats: min={summary['grounding_stats']['min']:.4f} "
          f"max={summary['grounding_stats']['max']:.4f} "
          f"mean={summary['grounding_stats']['mean']:.4f}")
    print(f"  Complexity stats: min={summary['complexity_stats']['min']:.4f} "
          f"max={summary['complexity_stats']['max']:.4f} "
          f"mean={summary['complexity_stats']['mean']:.4f}")
    print(f"  |yes-no| stats : mean={summary['logit_gap_abs_stats']['mean']:.4f} "
          f"median={summary['logit_gap_abs_stats']['median']:.4f}")
    print()

    if delta >= 0.005:
        print("  GATE POSITIVE -> proceed to hyperparameter tune + full 3000q runs.")
    elif delta >= -0.002:
        print("  GATE NEUTRAL -> tune thresholds before committing to 3000q.")
    else:
        print("  GATE REGRESSION -> stop. Inspect zone counts and flip correctness.")
    print()
    print(f"  Saved: {OUT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
