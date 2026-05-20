# src/run_cash_eval.py — KESAV
# Runs LLaVA inference on CASH benchmark questions, scores with compute_accuracy

import json, os, torch, requests
from PIL import Image
from io import BytesIO
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

MODEL_PATH       = "D:/models/llava-1.5-7b"
CASH_PATH        = "datasets/cash/cash_v1_partial.json"
MAPPINGS_PATH    = "datasets/cash/cash_image_mappings.json"
OUTPUT_PATH      = "experiments/cash_predictions.json"


def load_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def fetch_image(urls: list) -> Image.Image:
    """Try each URL in order, return first that loads as a valid image."""
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            continue
    raise RuntimeError(f"All URLs failed: {urls}")


def run_inference(model, processor, image: Image.Image, question: str) -> str:
    prompt = f"USER: <image>\n{question} Answer yes or no only.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5)
    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer.split("ASSISTANT:")[-1].strip().lower()


if __name__ == "__main__":
    with open(CASH_PATH) as f:
        cash_data = json.load(f)
    with open(MAPPINGS_PATH) as f:
        mappings = json.load(f)

    print(f"Loaded {len(cash_data)} CASH questions")
    print("Loading model...")
    model, processor = load_model()
    print("Running CASH eval...\n")

    results = []
    errors = 0
    for i, item in enumerate(cash_data):
        qid = item["id"]
        question = item["question"]
        label = item["answer"].strip().lower()

        print(f"[{i+1}/{len(cash_data)}] {question[:70]}...")
        try:
            urls = mappings[qid]["urls"]
            image = fetch_image(urls)
            pred = run_inference(model, processor, image, question)
        except Exception as e:
            pred = "error"
            errors += 1
            print(f"  ERROR: {e}")

        results.append({"id": qid, "question": question, "label": label, "prediction": pred})

    os.makedirs("experiments", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} predictions to {OUTPUT_PATH} ({errors} errors)")

    from evaluate import compute_accuracy
    preds  = [r["prediction"] for r in results]
    labels = [r["label"] for r in results]
    score  = compute_accuracy(preds, labels)
    print("\n=== CASH BASELINE SCORES ===")
    print(json.dumps(score, indent=2))
