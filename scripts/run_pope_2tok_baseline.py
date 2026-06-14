# scripts/run_pope_2tok_baseline.py
# Measures the two-token vs eight-token evaluation gap on the full
# 3000-question POPE. Two-token reads YES={3582}, NO={1217}; eight-token
# reads YES={3582,8241,4874,3869}, NO={1217,3782,694,1939}.
#
# Usage:
#   python scripts/run_pope_2tok_baseline.py --split adversarial \
#       --model-path llava-hf/llava-1.5-7b-hf \
#       --data-dir datasets/pope --output-dir experiments \
#       --device cuda --cache-dir D:/models/hf_cache

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration,
)

# Resolve the project root one level above scripts/ so the package import
# and the default data/output paths work from any working directory, on
# Windows, Linux, or a cloud GPU instance.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pope_audit.evaluate import compute_f1

DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "datasets" / "pope")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "experiments")

YES_2TOK = [3582]
NO_2TOK = [1217]
YES_8TOK = [3582, 8241, 4874, 3869]
NO_8TOK = [1217, 3782, 694, 1939]


def load_llava(model_path: str, cache_dir: str | None = None,
               device: str = "cuda"):
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if device == "cuda":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, quantization_config=bnb, device_map="auto", **kwargs,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float32, **kwargs,
        )
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    return model, processor


@torch.no_grad()
def predict_both(model, processor, image, question, device):
    prompt = f"USER: <image>\n{question} Answer yes or no only.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    out = model.generate(
        **inputs, max_new_tokens=1,
        return_dict_in_generate=True, output_scores=True,
    )
    logits = out.scores[0][0].float()

    pred_2tok = "yes" if logits[YES_2TOK[0]] > logits[NO_2TOK[0]] else "no"
    yes_8 = max(logits[i].item() for i in YES_8TOK)
    no_8 = max(logits[i].item() for i in NO_8TOK)
    pred_8tok = "yes" if yes_8 > no_8 else "no"

    top_tok = int(logits.argmax())
    return pred_2tok, pred_8tok, top_tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="adversarial",
                        choices=["adversarial", "popular", "random"])
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit run to first N questions (default: all).")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="HF repo id or local path of the VLM.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Folder containing pope_{split}_full.json.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where to write the audit JSON.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace cache directory (e.g. D:/models/hf_cache).")
    args = parser.parse_args()

    path = Path(args.data_dir) / f"pope_{args.split}_full.json"
    with open(path) as f:
        items = json.load(f)
    if args.samples:
        items = items[: args.samples]

    print(f"=== Two-token vs eight-token POPE baseline | split={args.split} | n={len(items)} ===")
    print(f"Loading LLaVA from {args.model_path}...")
    model, processor = load_llava(args.model_path, args.cache_dir, args.device)
    device = str(model.device)
    print("Ready.\n")

    results = []
    t0 = time.time()
    top_tok_counts = {}

    for i, item in enumerate(items):
        image = Image.open(item["local_path"]).convert("RGB")
        pred_2, pred_8, top_tok = predict_both(
            model, processor, image, item["question"], device,
        )
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
            done = compute_f1(
                [r["pred_8tok"] for r in results],
                [r["label"] for r in results],
            )
            print(f"[{i+1}/{len(items)}] 8tok F1={done['f1']:.4f} | {elapsed/60:.1f}m")

    labels = [r["label"] for r in results]
    m2 = compute_f1([r["pred_2tok"] for r in results], labels)
    m8 = compute_f1([r["pred_8tok"] for r in results], labels)

    yes_2 = sum(1 for r in results if r["pred_2tok"] == "yes")
    yes_8 = sum(1 for r in results if r["pred_8tok"] == "yes")

    sorted_toks = sorted(top_tok_counts.items(), key=lambda x: -x[1])[:10]

    summary = {
        "split": args.split,
        "n": len(results),
        "model_path": args.model_path,
        "eval_2token": {"f1": m2["f1"], "precision": m2["precision"], "recall": m2["recall"],
                        "yes_predictions": yes_2,
                        "yes_rate": round(yes_2 / len(results), 3)},
        "eval_8token": {"f1": m8["f1"], "precision": m8["precision"], "recall": m8["recall"],
                        "yes_predictions": yes_8,
                        "yes_rate": round(yes_8 / len(results), 3)},
        "f1_gap_8tok_minus_2tok": round(m8["f1"] - m2["f1"], 4),
        "top_tokens_generated": {str(tok_id): count for tok_id, count in sorted_toks[:8]},
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"pope_{args.split}_2tok_vs_8tok.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    tok_names = {3582: "yes", 8241: "Yes", 4874: " yes", 3869: " Yes",
                 1217: "no", 3782: "No", 694: " no", 1939: " No"}
    print(f"\n{'='*65}")
    print(f"POPE {args.split.upper()} - two-token vs eight-token")
    print(f"{'='*65}")
    print(f"  2tok: F1={m2['f1']:.4f}  P={m2['precision']:.4f}  R={m2['recall']:.4f}  yes-rate={yes_2/len(results):.3f}")
    print(f"  8tok: F1={m8['f1']:.4f}  P={m8['precision']:.4f}  R={m8['recall']:.4f}  yes-rate={yes_8/len(results):.3f}")
    print(f"  Gap (8tok minus 2tok): {m8['f1'] - m2['f1']:+.4f} F1 points")
    print(f"\n  Top generated tokens:")
    for tok_id, count in sorted_toks[:8]:
        name = tok_names.get(tok_id, f"tok_{tok_id}")
        flag = "in 2tok set" if tok_id in YES_2TOK + NO_2TOK else "MISSED by 2tok"
        print(f"    tok={tok_id} ({name!r}): {count} times  [{flag}]")
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
