# src/run_pope_2tok_baseline.py
# Measures the 2-token vs 8-token evaluation gap on the full 3000-question POPE.
# This is the core finding of the paper: 2-token eval creates systematic yes-bias,
# depressing the baseline and inflating reported method gains.
#
# 2-token (prior papers): YES={3582}, NO={1217}
# 8-token (our standard): YES={3582,8241,4874,3869}, NO={1217,3782,694,1939}
#
# Usage: python src/run_pope_2tok_baseline.py --split adversarial

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image
from transformers import (
    AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration,
)

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import compute_f1

MODEL_PATH = "D:/models/llava-1.5-7b"
DATA_DIR = "datasets/pope"

# Token ID sets
YES_2TOK = [3582]           # "yes" — what prior papers use
NO_2TOK  = [1217]           # "no"
YES_8TOK = [3582, 8241, 4874, 3869]   # yes Yes ▁yes ▁Yes
NO_8TOK  = [1217, 3782,  694, 1939]   # no  No  ▁no  ▁No


def load_llava():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


@torch.no_grad()
def predict_both(model, processor, image, question, device):
    prompt = f"USER: <image>\n{question} Answer yes or no only.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
    logits = out.scores[0][0].float()

    pred_2tok = "yes" if logits[YES_2TOK[0]] > logits[NO_2TOK[0]] else "no"
    yes_8 = max(logits[i].item() for i in YES_8TOK)
    no_8  = max(logits[i].item() for i in NO_8TOK)
    pred_8tok = "yes" if yes_8 > no_8 else "no"

    # Also record top token to understand what the model actually generates
    top_tok = int(logits.argmax())
    return pred_2tok, pred_8tok, top_tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="adversarial", choices=["adversarial", "popular", "random"])
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()

    path = os.path.join(DATA_DIR, f"pope_{args.split}_full.json")
    with open(path) as f:
        items = json.load(f)
    if args.samples:
        items = items[:args.samples]

    print(f"=== 2-token vs 8-token POPE Baseline | split={args.split} | n={len(items)} ===")
    print("Loading LLaVA...")
    model, processor = load_llava()
    device = str(model.device)
    print("Ready.\n")

    results = []
    t0 = time.time()
    top_tok_counts = {}

    for i, item in enumerate(items):
        image = Image.open(item["local_path"]).convert("RGB")
        pred_2, pred_8, top_tok = predict_both(model, processor, image, item["question"], device)
        top_tok_counts[top_tok] = top_tok_counts.get(top_tok, 0) + 1
        results.append({
            "question_id": item["question_id"],
            "label": item["label"],
            "pred_2tok": pred_2,
            "pred_8tok": pred_8,
            "top_token_id": top_tok,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            done = compute_f1([r["pred_8tok"] for r in results], [r["label"] for r in results])
            print(f"[{i+1}/{len(items)}] 8tok F1={done['f1']:.4f} | {elapsed/60:.1f}m")

    # Score both
    labels = [r["label"] for r in results]
    m2 = compute_f1([r["pred_2tok"] for r in results], labels)
    m8 = compute_f1([r["pred_8tok"] for r in results], labels)

    # Count yes/no predictions
    yes_2 = sum(1 for r in results if r["pred_2tok"] == "yes")
    yes_8 = sum(1 for r in results if r["pred_8tok"] == "yes")

    # Most common top tokens
    sorted_toks = sorted(top_tok_counts.items(), key=lambda x: -x[1])[:10]

    summary = {
        "split": args.split,
        "n": len(results),
        "eval_2token": {"f1": m2["f1"], "precision": m2["precision"], "recall": m2["recall"],
                        "yes_predictions": yes_2, "yes_rate": round(yes_2/len(results), 3)},
        "eval_8token": {"f1": m8["f1"], "precision": m8["precision"], "recall": m8["recall"],
                        "yes_predictions": yes_8, "yes_rate": round(yes_8/len(results), 3)},
        "f1_gap_8tok_minus_2tok": round(m8["f1"] - m2["f1"], 4),
        "top_tokens_generated": {str(tok_id): count for tok_id, count in sorted_toks[:8]},
    }

    os.makedirs("experiments", exist_ok=True)
    out = f"experiments/pope_{args.split}_2tok_vs_8tok.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n{'='*65}")
    print(f"POPE {args.split.upper()} — 2-TOKEN vs 8-TOKEN EVALUATION")
    print(f"{'='*65}")
    print(f"  2-token (prior papers): F1={m2['f1']:.4f}  P={m2['precision']:.4f}  R={m2['recall']:.4f}  YES_rate={yes_2/len(results):.3f}")
    print(f"  8-token (our standard): F1={m8['f1']:.4f}  P={m8['precision']:.4f}  R={m8['recall']:.4f}  YES_rate={yes_8/len(results):.3f}")
    print(f"  Gap: {m8['f1'] - m2['f1']:+.4f} F1 points")
    print(f"\n  Top generated tokens:")
    tok_names = {3582:"yes", 8241:"Yes", 4874:" yes", 3869:" Yes",
                 1217:"no", 3782:"No", 694:" no", 1939:" No"}
    for tok_id, count in sorted_toks[:8]:
        name = tok_names.get(tok_id, f"tok_{tok_id}")
        in_2tok = "✓ 2tok" if tok_id in YES_2TOK + NO_2TOK else "✗ MISSED by 2tok"
        print(f"    tok={tok_id} ({name!r}): {count} times  [{in_2tok}]")
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
