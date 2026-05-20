# src/run_vsr_eval.py — KESAV
import json, torch, requests
from PIL import Image
from io import BytesIO
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

MODEL_PATH = "D:/models/llava-1.5-7b"

def load_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor

def run_inference(model, processor, image, question):
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5)
    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer.split("ASSISTANT:")[-1].strip().lower()

if __name__ == "__main__":
    data_path = "datasets/vsr/vsr_sample_100.json"
    with open(data_path) as f:
        sample = json.load(f)
    print(f"Loaded {len(sample)} real VSR questions")

    print("Loading model...")
    model, processor = load_model()
    print("Running VSR eval...\n")

    results = []
    for i, item in enumerate(sample):
        question = f"Is the following statement true about the image: '{item['question']}'? Answer yes or no only."
        print(f"[{i+1}/{len(sample)}] {question[:60]}...")
        try:
            image = Image.open(BytesIO(requests.get(item["image_url"]).content)).convert("RGB")
            pred = run_inference(model, processor, image, question)
        except Exception as e:
            pred = "error"
            print(f"  ERROR: {e}")
        results.append({"question": item["question"], "label": item["label"], "prediction": pred})

    out = "experiments/vsr_predictions.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out}")

    from evaluate import compute_accuracy
    preds  = [r["prediction"] for r in results]
    labels = [r["label"] for r in results]
    score  = compute_accuracy(preds, labels)
    print("\n=== SCORES ===")
    print(json.dumps(score, indent=2))