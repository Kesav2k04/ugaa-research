# src/run_pope_eval_full.py - KESAV
# Full POPE benchmark evaluation (3000 questions).
#
# Supports three modes:
#   --baseline               vanilla LLaVA, no intervention
#   --variant ugaa           UGAA v5 constant NO-bias (beta-tunable)
#   --variant clip_certainty CLIP-grounded certainty-modulated NO-bias
#
# CLIP certainty: max cosine similarity between the object noun and 4×4 image
# crops mapped through a sigmoid. Discriminates TPs (object present → high
# certainty → low bias) from FPs (object absent → low certainty → high bias).
# Overcomes constant beta=1.0 which broke recall by applying max bias to ALL
# questions equally.
#
# Requires: datasets/pope/pope_{split}_full.json + local image files
#
# Usage:
#   python src/run_pope_eval_full.py --split adversarial --baseline
#   python src/run_pope_eval_full.py --split adversarial --beta 1.0
#   python src/run_pope_eval_full.py --split adversarial --variant clip_certainty --beta 1.0
#   python src/run_pope_eval_full.py --split adversarial --variant clip_certainty --beta 1.5
#
# Outputs:
#   experiments/pope_full_{split}_{tag}_predictions.json
#   experiments/pope_full_{split}_{tag}_summary.json

import argparse
import json
import os
import re
import sys
import time

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    CLIPModel,
    CLIPProcessor,
    LlavaForConditionalGeneration,
)

sys.path.insert(0, os.path.dirname(__file__))
from ugaa_hook import UGAAHook, YES_TOKEN_IDS, NO_TOKEN_IDS, _get_yes_no_logits
from clip_l_grounding import clip_l_certainty, load_clip_l_components
from evaluate import compute_f1

# ---------------------------------------------------------------------------
# CLIP certainty - same logic as run_ablation_a.py
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_SIG_CENTER = 0.22   # CLIP-B/32 zero-shot threshold
CLIP_SIG_SCALE = 20.0    # sigmoid steepness
_clip_cache: dict = {}
_NOUN_RE = re.compile(r"[Ii]s there an?\s+(.+?)\s+in the image", re.I)


def _pope_noun(question: str) -> str:
    m = _NOUN_RE.search(question)
    return m.group(1).strip() if m else question.rstrip("?").split()[-1]


def _get_clip(device):
    key = str(device)
    if key not in _clip_cache:
        print(f"[CLIP] Loading {CLIP_MODEL_NAME} (fp16)...")
        model = CLIPModel.from_pretrained(
            CLIP_MODEL_NAME, torch_dtype=torch.float16
        ).to(device).eval()
        proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_cache[key] = (model, proc)
        print("[CLIP] Ready.")
    return _clip_cache[key]


@torch.no_grad()
def clip_certainty_for_question(image: Image.Image, question: str, device) -> float:
    """Max CLIP sim (4×4 crops + global) → sigmoid certainty ∈ (0,1)."""
    clip_model, clip_proc = _get_clip(device)
    noun = _pope_noun(question)
    w, h = image.size
    pw, ph = max(w // 4, 1), max(h // 4, 1)
    crops = [
        image.crop((c * pw, r * ph, min((c + 1) * pw, w), min((r + 1) * ph, h)))
        for r in range(4) for c in range(4)
    ]
    crops.append(image)
    img_in = clip_proc(images=crops, return_tensors="pt", padding=True)
    img_in = {k: v.to(device).half() for k, v in img_in.items() if isinstance(v, torch.Tensor)}
    img_feats = clip_model.get_image_features(**img_in).float()
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    txt_in = clip_proc(text=[noun], return_tensors="pt", padding=True)
    txt_in = {k: v.to(device) for k, v in txt_in.items() if isinstance(v, torch.Tensor)}
    txt_feat = clip_model.get_text_features(**txt_in).float()
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    max_sim = float((img_feats @ txt_feat.T).squeeze(-1).max())
    return float(torch.sigmoid(torch.tensor(CLIP_SIG_SCALE * (max_sim - CLIP_SIG_CENTER))))

DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
DEFAULT_DATA_DIR = "datasets/pope"
DEFAULT_OUTPUT_DIR = "experiments"
VISUAL_START = 1
VISUAL_END = 577
BASELINE_F1_100 = 0.8041  # 100-sample adversarial baseline (reference)


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


def load_items(data_dir: str, split: str, max_samples: int | None) -> list:
    path = os.path.join(data_dir, f"pope_{split}_full.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run: python scripts/download_pope_full.py --splits " + split
        )
    with open(path) as f:
        items = json.load(f)
    if max_samples and max_samples < len(items):
        items = items[:max_samples]
    return items


def open_image(item: dict) -> Image.Image | None:
    path = item.get("local_path", "")
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return None


@torch.no_grad()
def run_baseline(model, processor, image, question: str) -> str:
    prompt = f"USER: <image>\n{question} Answer yes or no only.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=5)
    text = processor.decode(out[0], skip_special_tokens=True)
    return text.split("ASSISTANT:")[-1].strip().lower()


def infer_clip_certainty(model, processor, image, question, beta, device):
    """Single inference step using CLIP-B/32 certainty-modulated NO-bias."""
    certainty = clip_certainty_for_question(image, question, device)
    real_yes, real_no = _get_yes_no_logits(model, processor, image, question, device)
    score = (real_yes - real_no) - beta * (1.0 - certainty)
    return "yes" if score > 0 else "no"


def infer_clip_l(model, processor, image, question, beta, device):
    """Single inference step using CLIP-L/14-336 per-patch certainty (best signal)."""
    certainty = clip_l_certainty(model, processor, image, question, device)
    real_yes, real_no = _get_yes_no_logits(model, processor, image, question, device)
    score = (real_yes - real_no) - beta * (1.0 - certainty)
    return "yes" if score > 0 else "no"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="adversarial",
                        choices=["adversarial", "popular", "random"])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit to first N questions (default: all 3000).")
    parser.add_argument("--baseline", action="store_true",
                        help="Vanilla LLaVA, no UGAA.")
    parser.add_argument("--variant", default="ugaa",
                        choices=["ugaa", "clip_certainty", "clip_l"],
                        help="ugaa=constant; clip_certainty=CLIP-B/32; clip_l=CLIP-L/14-336.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="HF repo id or local path of the VLM.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Directory containing pope_{split}_full.json.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for prediction and summary JSON outputs.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace cache directory (e.g. D:/models/hf_cache).")
    args = parser.parse_args()

    if args.baseline:
        tag = "baseline"
    elif args.variant == "clip_certainty":
        tag = f"clip_b{args.beta}"
    elif args.variant == "clip_l":
        tag = f"clip_l_b{args.beta}"
    else:
        tag = f"beta{args.beta}"

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_pred = f"{out_dir}/pope_full_{args.split}_{tag}_predictions.json"
    out_summ = f"{out_dir}/pope_full_{args.split}_{tag}_summary.json"

    print(f"=== Full POPE Eval | split={args.split} | {tag} ===")
    items = load_items(args.data_dir, args.split, args.samples)
    print(f"Loaded {len(items)} questions from {args.data_dir}/pope_{args.split}_full.json")

    print(f"Loading LLaVA from {args.model_path}...")
    model, processor = load_llava(args.model_path, args.cache_dir, args.device)
    ugaa = None
    if not args.baseline and args.variant == "ugaa":
        ugaa = UGAAHook(beta=args.beta)

    if args.baseline:
        mode = "baseline (no UGAA)"
    elif args.variant == "clip_certainty":
        mode = f"clip_certainty (CLIP-B/32) beta={args.beta}"
        _get_clip(model.device)
    elif args.variant == "clip_l":
        mode = f"clip_l (CLIP-L/14-336 per-patch) beta={args.beta}"
        load_clip_l_components(str(model.device))
    else:
        mode = f"UGAA v5 constant beta={args.beta}"
    print(f"Ready. Mode: {mode}\n")

    results = []
    errors = 0
    t0 = time.time()

    for i, item in enumerate(items):
        question = item["question"]

        image = open_image(item)
        if image is None:
            errors += 1
            results.append({
                "question_id": item["question_id"],
                "question": question,
                "label": item["label"],
                "prediction": "error",
            })
            continue

        try:
            if args.baseline:
                pred = run_baseline(model, processor, image, question)
            elif args.variant == "clip_certainty":
                pred = infer_clip_certainty(
                    model, processor, image,
                    question + " Answer yes or no only.",
                    args.beta, str(model.device),
                )
            elif args.variant == "clip_l":
                pred = infer_clip_l(
                    model, processor, image,
                    question + " Answer yes or no only.",
                    args.beta, str(model.device),
                )
            else:
                pred = ugaa.infer(
                    model, processor, image,
                    question + " Answer yes or no only.",
                    visual_start=VISUAL_START, visual_end=VISUAL_END,
                )
        except Exception as e:
            print(f"  [{i+1}] INFER ERROR: {e}")
            pred = "error"
            errors += 1

        results.append({
            "question_id": item["question_id"],
            "question": question,
            "label": item["label"],
            "prediction": pred,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(items) - i - 1) / rate
            valid = [r for r in results if r["prediction"] != "error"]
            interim = compute_f1(
                [r["prediction"] for r in valid],
                [r["label"] for r in valid],
            )
            print(
                f"[{i+1}/{len(items)}] "
                f"F1={interim['f1']:.4f} P={interim['precision']:.4f} R={interim['recall']:.4f} | "
                f"{elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m left | "
                f"errors={errors}"
            )

    # Final metrics (exclude errors)
    valid = [r for r in results if r["prediction"] != "error"]
    preds = [r["prediction"] for r in valid]
    labels = [r["label"] for r in valid]
    metrics = compute_f1(preds, labels)

    tp = sum(1 for r in valid if r["prediction"] == "yes" and r["label"] == "yes")
    tn = sum(1 for r in valid if r["prediction"] == "no"  and r["label"] == "no")
    fp = sum(1 for r in valid if r["prediction"] == "yes" and r["label"] == "no")
    fn = sum(1 for r in valid if r["prediction"] == "no"  and r["label"] == "yes")
    total_time = time.time() - t0

    summary = {
        "split": args.split,
        "mode": mode,
        "n_total": len(results),
        "n_valid": len(valid),
        "n_errors": errors,
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "baseline_f1_100sample_ref": BASELINE_F1_100,
        "delta_f1_vs_100sample_ref": round(metrics["f1"] - BASELINE_F1_100, 4),
        "total_time_min": round(total_time / 60, 1),
        "sec_per_question": round(total_time / max(len(valid), 1), 2),
    }

    with open(out_pred, "w") as f:
        json.dump(results, f, indent=2)
    with open(out_summ, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== FINAL RESULTS ({args.split}, {len(valid)}/{len(results)} valid) ===")
    print(json.dumps(summary, indent=2))
    print(f"\nPredictions -> {out_pred}")
    print(f"Summary     -> {out_summ}")


if __name__ == "__main__":
    main()
