# src/run_pope_eval_ugaa.py — KESAV
# POPE eval with UGAA v5 (certainty-modulated NO-bias).
#
# Usage:
#   single beta:  python src/run_pope_eval_ugaa.py --beta 0.5
#   sweep:        python src/run_pope_eval_ugaa.py --sweep

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
from evaluate import compute_f1

MODEL_PATH = "D:/models/llava-1.5-7b"
VISUAL_START = 1
VISUAL_END = 577
SWEEP_BETAS = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]
DATASET = "datasets/pope/pope_sample_100.json"
BASELINE_F1 = 0.8041


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


def run_one_beta(beta, samples, images, model, processor):
    ugaa = UGAAHook(beta=beta)
    print(f"\n========== UGAA v5 — beta={beta} ==========\n")

    results = []
    for i, item in enumerate(samples):
        question_raw = item["question"]
        # POPE questions are already phrased as "Is there a X in the image?".
        question = f"{question_raw} Answer yes or no only."
        print(f"\n[{i + 1}/{len(samples)}] {question_raw}")

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
            "question_id": item.get("question_id", i + 1),
            "question": question_raw,
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
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if not args.sweep and args.beta is None:
        args.beta = 0.5

    print("Loading LLaVA...")
    model, processor = load_llava()
    print("LLaVA loaded.\n")

    with open(DATASET) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} POPE questions from {DATASET}")

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

        preds = [r["prediction"] for r in results]
        labels = [r["label"] for r in results]
        metrics = compute_f1(preds, labels)
        summary[beta] = metrics

        path = f"experiments/pope_ugaa_v5_beta{beta}.json"
        write_results(results, path)
        if not args.sweep:
            write_results(results, "experiments/pope_ugaa_v5_predictions.json")

        delta = metrics.get("f1", 0.0) - BASELINE_F1
        print(
            f"\n--- beta={beta} | F1={metrics.get('f1', 0):.4f} | "
            f"delta_F1={delta:+.4f} ---\n"
        )

    print("\n========== SUMMARY ==========")
    print(f"Baseline POPE: F1={BASELINE_F1}")
    for beta, m in summary.items():
        delta = m.get("f1", 0.0) - BASELINE_F1
        print(
            f"beta={beta}: F1={m.get('f1', 0):.4f} "
            f"P={m.get('precision', 0):.4f} R={m.get('recall', 0):.4f} "
            f"delta_F1={delta:+.4f}"
        )

    if args.sweep:
        best_beta = max(summary, key=lambda b: summary[b].get("f1", 0.0))
        print(f"\nBest beta: {best_beta} → F1={summary[best_beta].get('f1', 0):.4f}")


if __name__ == "__main__":
    main()
