# src/run_whatsup_eval.py — KESAV
import json
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

MODEL_PATH = "D:/models/llava-1.5-7b"

def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4"
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        quantization_config=bnb, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor

def run_inference(model, processor, image, option_a, option_b):
    # Construct a descriptive multiple choice prompt for LLaVA
    question = (
        f"Which statement accurately describes the image?\n"
        f"Option A: {option_a}\n"
        f"Option B: {option_b}\n"
        f"Answer with 'Option A' or 'Option B' only."
    )
    
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10)
        
    answer = processor.decode(output[0], skip_special_tokens=True)
    cleaned_answer = answer.split("ASSISTANT:")[-1].strip().lower()
    
    # Map the text generation back to a prediction choice
    if "option a" in cleaned_answer or "a" == cleaned_answer:
        return "A"
    elif "option b" in cleaned_answer or "b" == cleaned_answer:
        return "B"
    else:
        return cleaned_answer  # Fallback for unexpected formats

if __name__ == "__main__":
    # What'sUp benchmark usually has descriptive / flipped pairs
    data_path = "datasets/whatsup/whatsup_sample_100.json"
    with open(data_path) as f:
        sample = json.load(f)
    print(f"Loaded {len(sample)} real What'sUp questions")

    print("Loading model...")
    model, processor = load_model()
    print("Running What'sUp eval...\n")

    results = []
    for i, item in enumerate(sample):
        # What'sUp typical schema involves original text vs spatial control text
        # Adjust keys if your local json keys match 'caption_a' / 'caption_b'
        option_a = item.get("caption_a", item.get("text_correct"))
        option_b = item.get("caption_b", item.get("text_swapped"))
        correct_option = item.get("correct_label", "A") # Usually 'A' if option_a is correct
        
        print(f"[{i+1}/{len(sample)}] Testing spatial pair: {option_a[:40]} vs {option_b[:40]}...")
        
        try:
            # Handle both local paths and web URLs dynamically
            img_src = item["image_url"]
            if img_src.startswith(("http://", "https://")):
                image = Image.open(BytesIO(requests.get(img_src).content)).convert("RGB")
            else:
                image = Image.open(img_src).convert("RGB")
                
            pred = run_inference(model, processor, image, option_a, option_b)
        except Exception as e:
            pred = "error"
            print(f"  ERROR: {e}")
            
        results.append({
            "option_a": option_a,
            "option_b": option_b,
            "label": correct_option,
            "prediction": pred
        })

    out = "experiments/whatsup_predictions.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out}")

    # Calculate metrics
    valid_results = [r for r in results if r["prediction"] in ["A", "B"]]
    if valid_results:
        correct_count = sum(1 for r in valid_results if r["prediction"] == r["label"])
        accuracy = (correct_count / len(valid_results)) * 100
        print("\n=== SCORES ===")
        print(json.dumps({"accuracy": f"{accuracy:.2f}%", "valid_evals": len(valid_results), "total": len(results)}, indent=2))
    else:
        print("\n=== SCORES ===")
        print("No valid predictions generated to score.")
