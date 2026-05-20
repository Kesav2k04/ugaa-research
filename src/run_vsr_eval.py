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
    # 3 sample entries in VSR format — matches Karthi's actual output
    sample = [
        {"image_link": "http://images.cocodataset.org/train2017/000000419052.jpg",
         "caption": "The pizza is above the couch.", "label": 0},
        {"image_link": "http://images.cocodataset.org/val2017/000000039769.jpg",
         "caption": "There is a cat on the bed.", "label": 1},
        {"image_link": "http://images.cocodataset.org/val2017/000000039769.jpg",
         "caption": "The remote is under the cat.", "label": 0},
    ]

    print("Loading model...")
    model, processor = load_model()
    print("Running VSR eval...\n")

    results = []
    for i, item in enumerate(sample):
        question = f"Is the following statement true about the image: '{item['caption']}'? Answer yes or no only."
        print(f"[{i+1}/{len(sample)}] {question[:60]}...")
        try:
            image = Image.open(BytesIO(requests.get(item["image_link"]).content)).convert("RGB")
            pred = run_inference(model, processor, image, question)
        except Exception as e:
            pred = "error"
            print(f"  ERROR: {e}")
        label_str = "yes" if item["label"] == 1 else "no"
        results.append({"caption": item["caption"], "label": label_str, "prediction": pred})

    out = "experiments/vsr_predictions.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out}")
    print(results)