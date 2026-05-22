# src/run_whatsup_eval.py — KESAV
# What'sUp spatial benchmark baseline.
#
# Dataset: coco_qa_two_obj.json
#   Each entry: [image_id, correct_caption, wrong_caption]
#   caption_a is ALWAYS the correct spatial description.
#   caption_b is the spatial opposite (e.g. "left" ↔ "right").
#
# Why independent scoring (not forced-choice A/B):
#   LLaVA has a strong position bias toward option A in forced-choice
#   prompts — it always outputs "Option A", giving 100% trivially since
#   the label is always A.  We instead score each caption independently
#   and predict whichever caption the model assigns a higher yes-logit.
#   This is bias-free and directly comparable to POPE/VSR.
#
# Usage:
#   python src/run_whatsup_eval.py [--samples N]

import argparse
import json
import os
import random
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
from ugaa_hook import YES_TOKEN_IDS, NO_TOKEN_IDS

MODEL_PATH = "D:/models/llava-1.5-7b"
DATA_PATH = "datasets/whatsup/coco_qa_two_obj.json"
OUTPUT_PATH = "experiments/whatsup_predictions.json"
SEED = 42


def load_model():
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


def load_image(url):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


@torch.no_grad()
def score_caption(model, processor, image, caption: str) -> float:
    """
    Returns the yes-logit minus no-logit for the question
    'Does this image show: [caption]?'
    Uses the same 8-token set as POPE/VSR for consistency.
    """
    question = f"Does this image show: {caption}? Answer yes or no only."
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = out.scores[0][0].float()
    yes_score = max(logits[i].item() for i in YES_TOKEN_IDS)
    no_score = max(logits[i].item() for i in NO_TOKEN_IDS)
    return yes_score - no_score


def load_dataset(max_samples: int):
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    random.seed(SEED)
    sample = random.sample(raw, min(max_samples, len(raw)))
    items = []
    for idx, entry in enumerate(sample):
        image_id = int(entry[0])
        items.append({
            "question_id": idx,
            "image_id": image_id,
            "image_url": f"http://images.cocodataset.org/val2017/{image_id:012d}.jpg",
            "correct_caption": entry[1],
            "wrong_caption": entry[2],
        })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    print("=" * 50)
    print("What'sUp Spatial Evaluation (independent scoring)")
    print("=" * 50)

    items = load_dataset(args.samples)
    print(f"Loaded {len(items)} questions from {DATA_PATH}")

    print("\nLoading LLaVA...")
    model, processor = load_model()
    print("Model loaded.\n")

    results = []
    correct = 0

    for i, item in enumerate(items):
        try:
            image = load_image(item["image_url"])
        except Exception as e:
            print(f"[{i+1}/{len(items)}] image fetch failed: {e}")
            results.append({**item, "prediction": "ERROR", "correct": False})
            continue

        score_correct = score_caption(model, processor, image, item["correct_caption"])
        score_wrong = score_caption(model, processor, image, item["wrong_caption"])
        pred = "correct" if score_correct > score_wrong else "wrong"
        is_correct = pred == "correct"
        if is_correct:
            correct += 1

        print(
            f"[{i+1}/{len(items)}] {pred.upper()} | "
            f"correct={score_correct:.2f} wrong={score_wrong:.2f} | "
            f"{item['correct_caption'][:50]}"
        )

        results.append({
            "question_id": item["question_id"],
            "image_id": item["image_id"],
            "correct_caption": item["correct_caption"],
            "wrong_caption": item["wrong_caption"],
            "score_correct": score_correct,
            "score_wrong": score_wrong,
            "prediction": pred,
            "correct": is_correct,
        })

    os.makedirs("experiments", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    valid = [r for r in results if r["prediction"] != "ERROR"]
    acc = correct / len(valid) if valid else 0.0
    print(f"\n========== RESULTS ==========")
    print(json.dumps({
        "accuracy": round(acc, 4),
        "correct": correct,
        "valid": len(valid),
        "total": len(results),
        "note": "Baseline (no UGAA). chance=0.50",
    }, indent=2))
    print(f"\nPredictions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
