# src/run_pope_eval.py — KESAV
# Runs LLaVA inference on POPE-format questions, saves predictions to JSON

import json
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

MODEL_PATH = "D:/models/llava-1.5-7b"

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor

def run_inference(model, processor, image, question):
    prompt = f"USER: <image>\n{question} Answer yes or no only.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5)
    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer.split("ASSISTANT:")[-1].strip().lower()

def evaluate_pope(questions: list, model, processor) -> list:
    """
    questions: list of dicts with keys:
        - question_id
        - image_url (or image_path)
        - question
        - label (yes/no)
    """
    results = []
    for i, item in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {item['question']}")
        try:
            if item.get("image_url"):
                image = Image.open(BytesIO(requests.get(item["image_url"]).content)).convert("RGB")
            else:
                image = Image.open(item["image_path"]).convert("RGB")
            pred = run_inference(model, processor, image, item["question"])
        except Exception as e:
            print(f"  ERROR: {e}")
            pred = "error"
        results.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "label": item["label"],
            "prediction": pred
        })
    return results

if __name__ == "__main__":
    # MOCK sample — replace with real POPE data from Karthigeyan later
    sample_questions = [
        {
            "question_id": 1,
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "question": "Is there a cat in the image?",
            "label": "yes"
        },
        {
            "question_id": 2,
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "question": "Is there a dog in the image?",
            "label": "no"
        }
    ]

    print("Loading model...")
    model, processor = load_model()
    print("Model loaded. Running POPE eval...\n")

    results = evaluate_pope(sample_questions, model, processor)

    output_path = "experiments/pope_predictions.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} predictions to {output_path}")
    print("Sample:", results)