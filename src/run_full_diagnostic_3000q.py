"""
run_full_diagnostic_3000q.py
============================

Per-question diagnostic for a full POPE split: image-attention to
patches (mean, layers 14-20), CLIP-L max patch-noun similarity, and
absolute first-token logit gap |l_yes - l_no|. The script is the
3,000-question generalisation of the 100-question diagnostic in
experiments/ugaa_v6_100q_validation.json, which Section 5.3 of the
paper relies on.

Output JSON has the same per-question schema as the 100q version
(question_id, label, baseline_prediction, grounding, gap_raw,
yes_raw, no_raw, visual_span) plus an additional clip_l_max_sim
field, so analysis/diagnostic_stats.py works on either artifact
without modification.

Usage (LLaVA-1.5-7B baseline, full 3,000 adversarial questions on
the local RTX 3070 Ti):

    python src/run_full_diagnostic_3000q.py \\
        --split adversarial \\
        --model-path llava-hf/llava-1.5-7b-hf \\
        --data-dir datasets/pope \\
        --output-dir experiments \\
        --device cuda --cache-dir D:/models/hf_cache

Runtime: roughly 1.5x the baseline run because the probe forward
pass with output_attentions=True is the dominant cost. On the
RTX 3070 Ti laptop in 4-bit, expect ~50-70 minutes per split.

The default values match the paper's configuration: layers 14-20
for image attention, CLIP-L/14-336 for patch-noun similarity, the
LLaVA-1.5 ASSISTANT: prompt template, eight-token logit readout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor, AutoTokenizer, BitsAndBytesConfig,
    CLIPModel, CLIPTextModelWithProjection,
    LlavaForConditionalGeneration,
)

sys.path.insert(0, os.path.dirname(__file__))
from ugaa_hook import YES_TOKEN_IDS, NO_TOKEN_IDS

DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
DEFAULT_DATA_DIR = "datasets/pope"
DEFAULT_OUTPUT_DIR = "experiments"

CLIP_L_NAME = "openai/clip-vit-large-patch14-336"
LAYER_START = 14
LAYER_END = 20
VISUAL_START = 1
VISUAL_END = 577

_NOUN_RE = re.compile(r"[Ii]s there an?\s+(.+?)\s+in the image", re.I)


def pope_noun(q: str) -> str:
    m = _NOUN_RE.search(q)
    return m.group(1).strip() if m else q.rstrip("? ").split()[-1]


def load_llava(model_path: str, cache_dir, device: str):
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    if device == "cuda":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, quantization_config=bnb, device_map="auto", **kwargs,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float32, **kwargs,
        )
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    return model, processor


def load_clip_l(device, cache_dir):
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    text_model = CLIPTextModelWithProjection.from_pretrained(
        CLIP_L_NAME, torch_dtype=torch.float16, **kwargs,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(CLIP_L_NAME, **kwargs)
    full = CLIPModel.from_pretrained(CLIP_L_NAME, torch_dtype=torch.float16, **kwargs)
    proj_weight = full.visual_projection.weight.data.clone()
    del full
    import gc; gc.collect()
    visual_proj = torch.nn.Linear(1024, 768, bias=False, dtype=torch.float16)
    visual_proj.weight = torch.nn.Parameter(proj_weight)
    visual_proj = visual_proj.to(device).eval()
    return text_model, tokenizer, visual_proj


@torch.no_grad()
def llava_patch_features(model, pixel_values):
    vt = model.vision_tower
    try:
        out = vt(pixel_values.half(), output_hidden_states=False)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[0, 1:].float()
    except Exception:
        pass
    try:
        out = vt(pixel_values.half(), output_hidden_states=True)
        return out.hidden_states[-1][0, 1:].float()
    except Exception:
        pass
    raise RuntimeError("Could not extract LLaVA patch features.")


@torch.no_grad()
def diagnostic_one(model, processor, clip_text_model, clip_tok, clip_proj,
                   image, question, device):
    """One-question instrumentation. Returns dict with grounding, gap_raw,
    clip_l_max_sim, baseline_prediction, etc."""
    prompt = f"USER: <image>\n{question} Answer yes or no only.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v
              for k, v in inputs.items()}

    # --- forward 1: generate one token + get attentions ---
    try:
        model.language_model.config._attn_implementation = "eager"
        out = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=True,
        )
    finally:
        model.language_model.config._attn_implementation = "sdpa"

    logits = out.scores[0][0].float().cpu()
    yes_raw = max(float(logits[i]) for i in YES_TOKEN_IDS)
    no_raw = max(float(logits[i]) for i in NO_TOKEN_IDS)
    gap_raw = yes_raw - no_raw
    baseline = "yes" if gap_raw > 0 else "no"

    # mean attention layers LAYER_START:LAYER_END from last text token to image patches
    grounding = float("nan")
    try:
        per_layer = []
        for layer_attn in out.attentions[0]:
            a = layer_attn[0].float().cpu()  # [heads, 1, seq]
            vis = a[:, 0, VISUAL_START:VISUAL_END]
            per_layer.append(vis.mean(dim=0))
        if per_layer:
            layers = per_layer[LAYER_START:LAYER_END]
            mean_attn = torch.stack(layers).mean(dim=0).mean().item()
            grounding = float(mean_attn)
    except Exception:
        pass

    # --- CLIP-L max patch-noun similarity ---
    clip_l_max_sim = float("nan")
    try:
        noun = pope_noun(question)
        pix = processor.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        pf = llava_patch_features(model, pix)
        pf_proj = clip_proj(pf.half()).float()
        pf_proj = pf_proj / pf_proj.norm(dim=-1, keepdim=True)
        te = clip_tok([noun], padding=True, truncation=True, max_length=77,
                      return_tensors="pt")
        te = {k: v.to(device) for k, v in te.items()}
        tf = clip_text_model(**te).text_embeds.float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
        clip_l_max_sim = float((pf_proj @ tf.T).squeeze(-1).max())
    except Exception:
        pass

    return {
        "label": None,
        "prediction": baseline,
        "baseline_prediction": baseline,
        "yes_raw": yes_raw,
        "no_raw": no_raw,
        "gap_raw": gap_raw,
        "grounding": grounding,
        "clip_l_max_sim": clip_l_max_sim,
        "visual_span": [VISUAL_START, VISUAL_END],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="adversarial",
                        choices=["adversarial", "popular", "random"])
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit run to first N questions.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    data_path = Path(args.data_dir) / f"pope_{args.split}_full.json"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run scripts/download_pope_full.py first.")
        sys.exit(2)
    with open(data_path) as f:
        items = json.load(f)
    if args.samples:
        items = items[: args.samples]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ugaa_full_{args.split}_{len(items)}q_diagnostic.json"

    print(f"=== Full diagnostic | split={args.split} | n={len(items)} ===")
    print(f"Loading LLaVA from {args.model_path}...")
    model, processor = load_llava(args.model_path, args.cache_dir, args.device)
    print(f"Loading CLIP-L text encoder + visual_projection...")
    clip_text_model, clip_tok, clip_proj = load_clip_l(args.device, args.cache_dir)
    print("Ready.\n")

    results = []
    t0 = time.time()
    for i, item in enumerate(items):
        try:
            image = Image.open(item["local_path"]).convert("RGB")
            d = diagnostic_one(model, processor, clip_text_model, clip_tok, clip_proj,
                               image, item["question"], args.device)
            d["question_id"] = item.get("question_id", i + 1)
            d["label"] = item["label"]
            results.append(d)
        except Exception as exc:
            results.append({
                "question_id": item.get("question_id", i + 1),
                "label": item.get("label"),
                "error": str(exc),
            })
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            valid = [r for r in results if "baseline_prediction" in r]
            correct = sum(1 for r in valid if r["baseline_prediction"] == r["label"])
            acc = correct / max(len(valid), 1)
            print(f"  [{i+1}/{len(items)}] {elapsed/60:.1f}m  acc={acc:.4f}")

    # confusion summary
    valid = [r for r in results if "baseline_prediction" in r]
    tp = sum(1 for r in valid if r["baseline_prediction"]=="yes" and r["label"]=="yes")
    tn = sum(1 for r in valid if r["baseline_prediction"]=="no"  and r["label"]=="no")
    fp = sum(1 for r in valid if r["baseline_prediction"]=="yes" and r["label"]=="no")
    fn = sum(1 for r in valid if r["baseline_prediction"]=="no"  and r["label"]=="yes")
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    summary = {
        "split": args.split,
        "n_total": len(results),
        "n_valid": len(valid),
        "n_errors": len(results) - len(valid),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "model_path": args.model_path,
        "layers_for_grounding": [LAYER_START, LAYER_END],
        "runtime_minutes": round((time.time()-t0)/60.0, 1),
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nDone. f1={f1:.4f} p={p:.4f} r={r:.4f}  ({len(valid)}/{len(results)} valid)")
    print(f"Saved {out_path}")
    print()
    print("Next: run inferential stats on this artifact:")
    print(f"  python analysis/diagnostic_stats.py --diagnostic {out_path}")


if __name__ == "__main__":
    main()
