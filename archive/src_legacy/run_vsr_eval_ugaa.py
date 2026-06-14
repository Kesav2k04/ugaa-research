# src/run_vsr_eval_ugaa.py — KESAV
# VSR eval with UGAA v5 (certainty-modulated NO-bias).
#
# Two run modes:
#   single beta:  python src/run_vsr_eval_ugaa.py --beta 0.5
#   sweep:        python src/run_vsr_eval_ugaa.py --sweep
#
# Sweep writes one JSON per beta: experiments/vsr_ugaa_v5_beta{β}.json
# Single mode also writes a canonical experiments/vsr_ugaa_v5_predictions.json.

import argparse
import json
import os
import sys

import requests
import torch
from PIL import Image
from io import BytesIO
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

sys.path.insert(0, os.path.dirname(__file__))
from ugaa_hook import UGAAHook

MODEL_PATH = "D:/models/llava-1.5-7b"
VISUAL_START = 1
VISUAL_END = 577
SWEEP_BETAS = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]
DATASET = "datasets/vsr/vsr_sample_100.json"
BASELINE_ACC = 0.66


def load_llava():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def fetch_image(item):
    img_data = requests.get(item["image_url"], timeout=10).content
    return Image.open(BytesIO(img_data)).convert("RGB")


def score(results):
    tp = sum(1 for r in results if r["prediction"] == "yes" and r["label"] == "yes")
    tn = sum(1 for r in results if r["prediction"] == "no" and r["label"] == "no")
    fp = sum(1 for r in results if r["prediction"] == "yes" and r["label"] == "no")
    fn = sum(1 for r in results if r["prediction"] == "no" and r["label"] == "yes")
    correct = tp + tn
    total = len(results)
    acc = correct / total if total else 0.0
    return {"accuracy": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total}


def run_one_beta(beta, samples, images, model, processor):
    """Run all VSR questions for a single beta. Caches images to avoid refetching."""
    ugaa = UGAAHook(beta=beta)
    print(f"\n========== UGAA v5 — beta={beta} ==========\n")

    results = []
    for i, item in enumerate(samples):
        statement = item["question"]
        question = (
            f"Is the following statement true about the image: "
            f"'{statement}'? Answer yes or no only."
        )
        print(f"\n[{i + 1}/{len(samples)}] {statement[:70]}")

        image = images[i]
        if image is None:
            pred = "error"
        else:
            try:
                pred = ugaa.infer(
                    model, processor, image, question,
                    visual_start=VISUAL_START, visual_end=VISUAL_END,
                )
            except Exception as e:
                print(f"  INFER ERROR: {e}")
                pred = "error"

        results.append({
            "question": statement,
            "label": item["label"],
            "prediction": pred,
        })

    return results


def write_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beta", type=float, default=None,
        help="Single beta to run. Ignored if --sweep is set."
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help=f"Sweep beta over {SWEEP_BETAS}.",
    )
    args = parser.parse_args()

    if not args.sweep and args.beta is None:
        args.beta = 0.5  # plan default

    print("Loading LLaVA...")
    model, processor = load_llava()
    print("LLaVA loaded.\n")

    with open(DATASET) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} VSR questions from {DATASET}")

    # Pre-fetch images once so the sweep doesn't re-download N*beta times.
    print("Pre-fetching images...")
    images = []
    for i, item in enumerate(samples):
        try:
            images.append(fetch_image(item))
        except Exception as e:
            print(f"  [{i + 1}] image fetch failed: {e}")
            images.append(None)
    print(f"Cached {sum(1 for x in images if x is not None)}/{len(samples)} images.\n")

    betas = SWEEP_BETAS if args.sweep else [args.beta]
    summary = {}

    for beta in betas:
        results = run_one_beta(beta, samples, images, model, processor)
        metrics = score(results)
        summary[beta] = metrics

        path = f"experiments/vsr_ugaa_v5_beta{beta}.json"
        write_results(results, path)
        if not args.sweep:
            write_results(results, "experiments/vsr_ugaa_v5_predictions.json")

        delta = metrics["accuracy"] - BASELINE_ACC
        print(
            f"\n--- beta={beta} | acc={metrics['accuracy']:.4f} | "
            f"TP={metrics['tp']} TN={metrics['tn']} "
            f"FP={metrics['fp']} FN={metrics['fn']} | "
            f"delta={delta:+.4f} ---\n"
        )

    print("\n========== SUMMARY ==========")
    print(f"Baseline VSR: {BASELINE_ACC} (TP=48, TN=18, FP=31, FN=3)")
    for beta, m in summary.items():
        delta = m["accuracy"] - BASELINE_ACC
        print(
            f"beta={beta}: acc={m['accuracy']:.4f} "
            f"(TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}) "
            f"delta={delta:+.4f}"
        )

    if args.sweep:
        best_beta = max(summary, key=lambda b: summary[b]["accuracy"])
        print(f"\nBest beta: {best_beta} → acc={summary[best_beta]['accuracy']:.4f}")


if __name__ == "__main__":
    main()
