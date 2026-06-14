"""
Quick diagnostic: measure CLIP-L per-patch max_sim for TP vs FP on 100-sample POPE.
Expected: gap ≥ 0.02 (vs CLIP-B/32's 0.009).
Run time: ~5 minutes.
Usage: python src/diag_clip_l.py
"""

import json, sys, os, requests, torch
from PIL import Image
from io import BytesIO
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(__file__))
from clip_l_grounding import clip_l_certainty, load_clip_l_components, get_llava_patch_features, pope_noun
from ugaa_hook import YES_TOKEN_IDS, NO_TOKEN_IDS

MODEL_PATH = "D:/models/llava-1.5-7b"


def main():
    print("Loading LLaVA (4-bit)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    device = str(model.device)

    with open("datasets/pope/pope_sample_100.json") as f:
        samples = json.load(f)

    # Pre-fetch images
    print("Fetching images...")
    images = []
    for s in samples:
        try:
            img = Image.open(BytesIO(requests.get(s["image_url"], timeout=10).content)).convert("RGB")
        except:
            img = None
        images.append(img)

    # Load CLIP-L components
    load_clip_l_components(device)

    # Get LLaVA predictions first (to classify TP/FP/TN/FN)
    from ugaa_hook import _get_yes_no_logits
    print("\nClassifying 100 questions by TP/FP/TN/FN using LLaVA logits...")
    cats = {"TP": [], "FP": [], "TN": [], "FN": []}
    for i, (s, img) in enumerate(zip(samples, images)):
        if img is None: continue
        q = s["question"] + " Answer yes or no only."
        yes, no = _get_yes_no_logits(model, processor, img, q, model.device)
        pred = "yes" if yes > no else "no"
        cat = ("TP" if pred=="yes" and s["label"]=="yes" else
               "FP" if pred=="yes" and s["label"]=="no" else
               "TN" if pred=="no" and s["label"]=="no" else "FN")
        cats[cat].append(i)
        if (i+1) % 25 == 0:
            print(f"  {i+1}/100 classified")

    print(f"  TP={len(cats['TP'])} FP={len(cats['FP'])} TN={len(cats['TN'])} FN={len(cats['FN'])}")

    # Compute CLIP-L max_sim for each category
    print("\nComputing CLIP-L max_sim per category (this is the key diagnostic)...")
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    from clip_l_grounding import _CLIP_L_CACHE, CLIP_L_SIG_CENTER
    text_model, tokenizer, visual_proj = _CLIP_L_CACHE[device]

    def get_max_sim(idx):
        s, img = samples[idx], images[idx]
        if img is None: return None
        noun = pope_noun(s["question"])
        px = processor.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device)
        pf = get_llava_patch_features(model, px)
        pf = visual_proj(pf.half()).float()
        pf = pf / pf.norm(dim=-1, keepdim=True)
        ti = tokenizer([noun], padding=True, truncation=True, max_length=77, return_tensors="pt")
        ti = {k: v.to(device) for k, v in ti.items()}
        tf = text_model(**ti).text_embeds.float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
        return float((pf @ tf.T).squeeze(-1).max()), noun

    results = {}
    for cat in ["TP", "FP", "TN", "FN"]:
        sims = []
        print(f"\n  --- {cat} (n={len(cats[cat])}) ---")
        for idx in cats[cat]:
            r = get_max_sim(idx)
            if r:
                sim, noun = r
                sims.append(sim)
                print(f"    {noun!r}: {sim:.4f}")
        if sims:
            results[cat] = sum(sims)/len(sims)
            print(f"  {cat} MEAN: {results[cat]:.4f}")

    print("\n" + "="*60)
    print("CLIP-L DIAGNOSTIC SUMMARY")
    print("="*60)
    for cat, mean_sim in results.items():
        print(f"  {cat}: mean max_sim = {mean_sim:.4f}")

    if "TP" in results and "FP" in results:
        gap = results["TP"] - results["FP"]
        print(f"\n  TP-FP gap: {gap:+.4f}")
        print(f"  CLIP-B/32 baseline gap: +0.009")
        print(f"  Improvement factor: {gap/0.009:.1f}x")
        if gap > 0.02:
            print("\n  ✓ CLIP-L IS DISCRIMINATIVE. Run full eval:")
            print("  D:\\UGAA-MASTER\\ugaa_env\\python.exe src\\run_pope_eval_full.py --split adversarial --variant clip_l --beta 1.0")
        else:
            print("\n  ✗ CLIP-L not discriminative enough. Adjust SIG_CENTER.")
            avg_tp = results.get("TP", 0.27)
            avg_fp = results.get("FP", 0.25)
            suggested_center = (avg_tp + avg_fp) / 2
            print(f"  Suggested CLIP_L_SIG_CENTER = {suggested_center:.3f}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
