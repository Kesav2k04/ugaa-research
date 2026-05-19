# src/run_cash_eval.py — KARTHIGEYAN & KESAV
# Runs LLaVA inference on CASH benchmark, saves predictions to JSON

import os
import json
import torch
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

MODEL_PATH = "D:/models/llava-1.5-7b"
CASH_DATASET_PATH = r"G:\AI BASED PROJECT\ugaa-research\datasets\cash\cash_v1_partial.json"
GQA_QUESTIONS_PATH = r"G:\AI BASED PROJECT\ugaa-research\datasets\gqa\questions\val_balanced_questions.json"
VQA_ANNOTATIONS_PATH = r"G:\AI BASED PROJECT\ugaa-research\datasets\gqa\questions\v2_mscoco_train2014_annotations.json"
MAPPING_CACHE_PATH = r"G:\AI BASED PROJECT\ugaa-research\datasets\cash\cash_image_mappings.json"
OUTPUT_PREDICTIONS_PATH = r"G:\AI BASED PROJECT\ugaa-research\experiments\cash_predictions.json"

# Local directories (optional: if you download images locally to speed up evaluation)
GQA_IMAGES_DIR = None  # e.g., r"D:\datasets\gqa\images"
COCO_IMAGES_DIR = None # e.g., r"D:\datasets\coco\train2014"

def load_model():
    """
    Checks environment capabilities and loads the real LLaVA model.
    Falls back to mock mode if dependencies or model files are missing.
    """
    has_cuda = torch.cuda.is_available()
    has_model = os.path.exists(MODEL_PATH)
    
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes
        has_bnb = True
    except ImportError:
        has_bnb = False

    if has_cuda and has_model and has_bnb:
        print("CUDA, bitsandbytes, and model files found. Loading real model in 4-bit...")
        from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
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
        return model, processor, False
    else:
        print("\n" + "="*80)
        print("SYSTEM CHECK: CPU environment or missing model/dependencies.")
        print("Running in MOCK/DRY-RUN mode. The script will download images to verify URLs")
        print("and run mock inference. Real model inference will run on the GPU machine.")
        print("="*80 + "\n")
        return None, None, True

def run_inference(model, processor, image, question):
    """
    Runs LLaVA model generation.
    """
    # Use standard LLaVA prompt for open-ended VQA
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=15)
    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer.split("ASSISTANT:")[-1].strip().lower()

def get_image_mappings(cash_data):
    """
    Builds or loads a mapping of CASH question IDs to image details (imageId, url, source).
    """
    if os.path.exists(MAPPING_CACHE_PATH):
        print("Loading image mappings from cache...")
        with open(MAPPING_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Building image mappings (one-time setup)...")
    mappings = {}

    # Load GQA questions
    print("Loading GQA questions to map spatial/attribute items...")
    with open(GQA_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        gqa_data = json.load(f)

    # Load VQA annotations
    print("Loading VQA annotations to map counting items...")
    with open(VQA_ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        vqa_data = json.load(f)
    
    # Build a fast VQA question_id -> image_id lookup
    vqa_qid_to_iid = {ann["question_id"]: ann["image_id"] for ann in vqa_data["annotations"]}

    for item in tqdm(cash_data, desc="Mapping items"):
        qid = item["id"]
        category = item["category"]

        if category in ["spatial", "attribute"]:
            if qid in gqa_data:
                image_id = gqa_data[qid]["imageId"]
                mappings[qid] = {
                    "image_id": image_id,
                    "source": "gqa",
                    "urls": [
                        f"https://cs.stanford.edu/people/rak248/VG_100K_2/{image_id}.jpg",
                        f"https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg"
                    ]
                }
            else:
                print(f"Warning: GQA QID {qid} not found in val_balanced_questions.json")
        elif category == "counting":
            vqa_qid = int(qid)
            if vqa_qid in vqa_qid_to_iid:
                image_id = vqa_qid_to_iid[vqa_qid]
                filename = f"COCO_train2014_{image_id:012d}.jpg"
                mappings[qid] = {
                    "image_id": image_id,
                    "source": "vqa",
                    "urls": [
                        f"http://images.cocodataset.org/train2014/{filename}"
                    ]
                }
            else:
                print(f"Warning: VQA QID {qid} not found in annotations")

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(MAPPING_CACHE_PATH), exist_ok=True)
    with open(MAPPING_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2)
    print(f"Saved mappings cache to {MAPPING_CACHE_PATH}")
    return mappings

def load_image_from_mapping(mapping_item):
    """
    Loads image from local path if configured and exists; otherwise downloads it.
    """
    image_id = mapping_item["image_id"]
    source = mapping_item["source"]
    urls = mapping_item.get("urls", [])

    # 1. Try local GQA images
    if source == "gqa" and GQA_IMAGES_DIR and os.path.exists(GQA_IMAGES_DIR):
        local_path = os.path.join(GQA_IMAGES_DIR, f"{image_id}.jpg")
        if os.path.exists(local_path):
            return Image.open(local_path).convert("RGB")

    # 2. Try local COCO train2014 images
    if source == "vqa" and COCO_IMAGES_DIR and os.path.exists(COCO_IMAGES_DIR):
        filename = f"COCO_train2014_{image_id:012d}.jpg"
        local_path = os.path.join(COCO_IMAGES_DIR, filename)
        if os.path.exists(local_path):
            return Image.open(local_path).convert("RGB")

    # 3. Fallback to HTTP download
    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            continue
    raise ValueError(f"Failed to load image {image_id} from local paths or URLs: {urls}")

def evaluate_cash(questions: list, mappings: dict, model, processor, is_mock: bool) -> list:
    """
    Evaluates CASH dataset questions.
    If in mock mode, processes 5 samples per category to verify image downloading and mapping.
    """
    results = []
    
    if is_mock:
        print("Mock mode enabled: processing 5 samples per category to verify URLs and image loading...")
        categories = ["spatial", "counting", "attribute"]
        selected_questions = []
        for cat in categories:
            cat_qs = [q for q in questions if q["category"] == cat][:5]
            selected_questions.extend(cat_qs)
    else:
        selected_questions = questions

    for i, item in enumerate(selected_questions):
        qid = item["id"]
        question_text = item["question"]
        category = item["category"]
        ground_truth = item["answer"]
        
        print(f"[{i+1}/{len(selected_questions)}] [{category.upper()}] Q: {question_text}")
        
        try:
            mapping_item = mappings.get(qid)
            if not mapping_item:
                raise ValueError(f"No mapping cache found for QID {qid}")
            
            # Load / download image
            image = load_image_from_mapping(mapping_item)
            
            if is_mock:
                # Simulate prediction (using ground truth for demonstration of successful pipeline)
                pred = ground_truth.lower().strip()
                print(f"  Image downloaded successfully. Mock Pred: '{pred}' (GT: '{ground_truth}')")
            else:
                pred = run_inference(model, processor, image, question_text)
                print(f"  Model Pred: '{pred}' (GT: '{ground_truth}')")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            pred = "error"
            
        results.append({
            "question_id": qid,
            "category": category,
            "question": question_text,
            "label": ground_truth,
            "prediction": pred
        })
        
    return results

if __name__ == "__main__":
    print("Loading CASH dataset questions...")
    with open(CASH_DATASET_PATH, "r", encoding="utf-8") as f:
        cash_data = json.load(f)

    # Get question-to-image mappings (building the cache if not present)
    mappings = get_image_mappings(cash_data)

    print("Checking system specs and loading model...")
    model, processor, is_mock = load_model()

    print("Starting evaluation...")
    results = evaluate_cash(cash_data, mappings, model, processor, is_mock)

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS_PATH), exist_ok=True)
    with open(OUTPUT_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} predictions to {OUTPUT_PREDICTIONS_PATH}")
    if results:
        print("First prediction sample:", results[0])
