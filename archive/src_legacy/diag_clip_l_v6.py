"""
Diagnostic: compute CLIP-L max_sim for the same 100q used by run_ugaa_v6_100q.py
and check whether it discriminates TP from FP and FN from TN.

If max_sim(TP) > max_sim(FP) by a clear margin AND max_sim(FN) > max_sim(TN),
then CLIP-L-based selective flipping is viable as the v6 mechanism.
"""

import json
import os
import sys
import statistics as st

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from clip_l_grounding import load_clip_l_components, get_llava_patch_features, pope_noun
from ugaa_gate_v6 import load_llava_with_gate


DATA_PATH = "datasets/pope/pope_adversarial_full.json"
VAL_PATH = "experiments/ugaa_v6_100q_validation.json"
N = 100


@torch.no_grad()
def clip_l_max_sim(llava_model, processor, image, question, device, text_model, tokenizer, visual_proj):
    noun = pope_noun(question)
    pixel_values = processor.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    patch_feats = get_llava_patch_features(llava_model, pixel_values)
    proj_feats = visual_proj(patch_feats.half()).float()
    proj_feats = proj_feats / proj_feats.norm(dim=-1, keepdim=True)
    enc = tokenizer([noun], padding=True, truncation=True, max_length=77, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    txt = text_model(**enc).text_embeds.float()
    txt = txt / txt.norm(dim=-1, keepdim=True)
    sims = (proj_feats @ txt.T).squeeze(-1)
    return float(sims.max().item()), noun


def main():
    with open(VAL_PATH) as f:
        val = json.load(f)
    val_by_qid = {r["question_id"]: r for r in val["results"]}

    with open(DATA_PATH) as f:
        items = json.load(f)[:N]

    print("Loading LLaVA + CLIP-L...")
    model, processor = load_llava_with_gate()
    device = str(model.device)
    text_model, tokenizer, visual_proj = load_clip_l_components(device)
    print("Ready.\n")

    cats = {"TP": [], "FP": [], "TN": [], "FN": []}
    rows = []
    for i, item in enumerate(items):
        qid = item["question_id"]
        if qid not in val_by_qid:
            continue
        vr = val_by_qid[qid]
        p, l = vr["prediction"], vr["label"]
        cat = ("TP" if p == "yes" and l == "yes" else
               "FP" if p == "yes" and l == "no" else
               "TN" if p == "no" and l == "no" else "FN")
        try:
            img = Image.open(item["local_path"]).convert("RGB")
            sim, noun = clip_l_max_sim(model, processor, img, item["question"], device,
                                        text_model, tokenizer, visual_proj)
        except Exception as e:
            print(f"  [{i+1}] failed: {e}")
            continue
        cats[cat].append(sim)
        rows.append({"qid": qid, "cat": cat, "max_sim": sim, "noun": noun, "gap_raw": vr["gap_raw"]})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{N}] processed")

    print()
    print("CLIP-L max_sim discrimination by category (POPE adversarial, 100q):")
    print(f"  {'Cat':4} {'N':>3} {'min':>6} {'max':>6} {'mean':>6} {'median':>6}")
    for k in ["TP", "FP", "TN", "FN"]:
        vs = cats[k]
        if not vs:
            print(f"  {k:4} (no samples)")
            continue
        print(f"  {k:4} {len(vs):>3} {min(vs):>6.3f} {max(vs):>6.3f} {st.mean(vs):>6.3f} {st.median(vs):>6.3f}")
    print()

    if cats["TP"] and cats["FP"]:
        gap_pos = st.mean(cats["TP"]) - st.mean(cats["FP"])
        print(f"  TP - FP gap: {gap_pos:+.4f}  (need > 0 for YES-side discrimination)")
    if cats["FN"] and cats["TN"]:
        gap_neg = st.mean(cats["FN"]) - st.mean(cats["TN"])
        print(f"  FN - TN gap: {gap_neg:+.4f}  (need > 0 for NO-side recovery)")

    # Per-row dump for downstream tuning
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/clip_l_diag_100q.json", "w") as f:
        json.dump({"rows": rows, "by_cat": {k: cats[k] for k in cats}}, f, indent=2)
    print("\nSaved: experiments/clip_l_diag_100q.json")


if __name__ == "__main__":
    main()
