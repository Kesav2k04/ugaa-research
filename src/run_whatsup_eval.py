import json
import random
import os
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

# =========================
# CONFIG
# =========================

MODEL_PATH = "D:/models/llava-1.5-7b"
DATA_PATH = "datasets/whatsup/coco_qa_two_obj.json"
OUTPUT_PATH = "experiments/whatsup_predictions.json"

MAX_SAMPLES = 100
MAX_NEW_TOKENS = 10
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# MODEL LOADING
# =========================


def load_model():
    """
    Load quantized LLaVA model + processor.
    """
    print("Loading quantized LLaVA model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("Model loaded successfully.\n")
    return model, processor


# =========================
# IMAGE LOADING
# =========================


def load_image(img_src):
    """
    Load an image from a web URL or a local path.
    """
    if img_src.startswith(("http://", "https://")):
        response = requests.get(img_src, timeout=15)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(img_src).convert("RGB")
    return image


# =========================
# PROMPT + INFERENCE
# =========================


def build_prompt(option_a, option_b):
    """
    Construct the precise visual prompt formatting for LLaVA.
    """
    question = (
        "Which statement accurately describes the image?\n\n"
        f"Option A: {option_a}\n"
        f"Option B: {option_b}\n\n"
        "Answer ONLY with 'Option A' or 'Option B'."
    )
    return f"USER: <image>\n{question}\nASSISTANT:"


def parse_prediction(answer_text):
    """
    Parse model text generation tokens into evaluation labels.
    """
    cleaned = answer_text.lower().strip()
    if "option a" in cleaned or cleaned == "a":
        return "A"
    if "option b" in cleaned or cleaned == "b":
        return "B"
    return "INVALID"


@torch.no_grad()
def run_inference(model, processor, image, option_a, option_b):
    """
    Execute a forward pass tensor generation.
    """
    prompt = build_prompt(option_a, option_b)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    decoded = processor.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("ASSISTANT:")[-1].strip()
    return parse_prediction(answer)


# =========================
# DATASET LOADING
# =========================


def load_whatsup_dataset():
    """
    Parse, shuffle, sample, and map structural COCO image ID data.
    """
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    print(f"Loaded raw dataset with {len(raw_data)} entries")
    random.seed(SEED)
    sample_data = random.sample(raw_data, min(MAX_SAMPLES, len(raw_data)))
    samples = []

    for idx, entry in enumerate(sample_data):
        image_id = int(entry[0])
        caption_a = entry[1]
        caption_b = entry[2]
        image_url = f"http://images.cocodataset.org/val2017/{image_id:012d}.jpg"

        sample = {
            "question_id": idx,
            "image_id": image_id,
            "image_url": image_url,
            "caption_a": caption_a,
            "caption_b": caption_b,
            "correct_label": "A",
        }
        samples.append(sample)

    print(f"Prepared {len(samples)} sampled questions.\n")
    return samples


# =========================
# EVALUATION
# =========================


def evaluate(model, processor, samples):
    """
    Run prediction batch evaluations over full dataset target indices.
    """
    results = []
    total = len(samples)

    for idx, item in enumerate(samples, start=1):
        option_a = item["caption_a"]
        option_b = item["caption_b"]
        correct_label = item["correct_label"]

        print(
            f"[{idx}/{total}] Evaluating:\n  A: {option_a[:60]}\n  B: {option_b[:60]}"
        )

        try:
            image = load_image(item["image_url"])
            prediction = run_inference(
                model=model,
                processor=processor,
                image=image,
                option_a=option_a,
                option_b=option_b,
            )
            print(f"  Prediction: {prediction}\n")
        except Exception as error:
            prediction = "ERROR"
            print(f"  ERROR: {error}\n")

        results.append(
            {
                "question_id": item["question_id"],
                "option_a": option_a,
                "option_b": option_b,
                "label": correct_label,
                "prediction": prediction,
            }
        )
    return results


# =========================
# METRICS
# =========================


def compute_metrics(results):
    """
    Calculate mathematical performance benchmarks over parsing targets.
    """
    valid_results = [result for result in results if result["prediction"] in ["A", "B"]]
    if not valid_results:
        print("\nNo valid predictions generated.")
        return

    correct = sum(
        1 for result in valid_results if result["prediction"] == result["label"]
    )
    accuracy = (correct / len(valid_results)) * 100

    metrics = {
        "accuracy": f"{accuracy:.2f}%",
        "correct": correct,
        "valid_evals": len(valid_results),
        "total_samples": len(results),
    }

    print("\n========== RESULTS ==========")
    print(json.dumps(metrics, indent=2))


# =========================
# SAVE RESULTS
# =========================


def save_results(results):
    """
    Export the records into JSON structural output tracking files.
    """
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nPredictions saved to:\n{OUTPUT_PATH}")


# =========================
# MAIN
# =========================


def main():
    print("=" * 50)
    print("What'sUp Spatial Evaluation")
    print("=" * 50)

    samples = load_whatsup_dataset()
    model, processor = load_model()

    results = evaluate(
        model=model,
        processor=processor,
        samples=samples,
    )

    save_results(results)
    compute_metrics(results)


if __name__ == "__main__":
    main()
