# src/run_pope_eval_ugaa.py — KESAV
# POPE eval with REAL UGAA (CLIP-based patch distances)

import json, os, sys, torch, requests
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(__file__))
from ugaa_hook import UGAAHook
from clip_distance import load_clip, compute_distance_matrix

MODEL_PATH = "D:/models/llava-1.5-7b"

def load_llava():
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor

def run_inference_ugaa(model, processor, clip_model, clip_processor,
                       ugaa, image, question):
    # Step 1: Compute real CLIP distance matrix for this image+question
    question_tokens = question.replace("?", "").split()[:8]  # first 8 words
    distance_matrix = compute_distance_matrix(
        question_tokens, image, clip_model, clip_processor)  # [T, 16]

    # Step 2: Dummy logits for uncertainty (one per token)
    dummy_logits = torch.randn(len(question_tokens), 32000)

    # Step 3: Re-register UGAA with real distances for this question
    # Hook visual projector output — where patch tokens have spatial identity
    ugaa.remove()
    ugaa.register(model.multi_modal_projector, dummy_logits, distance_matrix)

    # Step 4: Run LLaVA inference normally — UGAA modifies attention internally
    prompt = f"USER: <image>\n{question} Answer yes or no only.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v
              for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5)
    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer.split("ASSISTANT:")[-1].strip().lower()

if __name__ == "__main__":
    print("Loading LLaVA...")
    model, processor = load_llava()

    print("Loading CLIP...")
    clip_model, clip_processor = load_clip()

    ugaa = UGAAHook(gamma=0.5, tau=0.5)
    print("UGAA ready.\n")

    with open("datasets/pope/pope_sample_100.json") as f:
        questions = json.load(f)

    results = []
    for i, item in enumerate(questions):
        print(f"[{i+1}/100] {item['question']}")
        try:
            image = Image.open(
                BytesIO(requests.get(item["image_url"]).content)).convert("RGB")
            pred = run_inference_ugaa(
                model, processor, clip_model, clip_processor,
                ugaa, image, item["question"])
        except Exception as e:
            print(f"  ERROR: {e}")
            pred = "error"
        results.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "label": item["label"],
            "prediction": pred
        })

    ugaa.remove()

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/pope_ugaa_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    from evaluate import compute_f1
    preds = [r["prediction"] for r in results]
    labels = [r["label"] for r in results]
    score = compute_f1(preds, labels)

    print("\n=== POPE WITH REAL UGAA ===")
    print(json.dumps(score, indent=2))
    print("\nBaseline was: F1=0.8041, P=0.8298, R=0.78")
    print(f"UGAA result:  F1={score['f1']}, P={score['precision']}, R={score['recall']}")
