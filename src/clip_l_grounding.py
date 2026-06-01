"""
clip_l_grounding.py - CLIP-L/14-336 per-patch certainty for POPE hallucination.

WHY THIS IS BETTER THAN CLIP-B/32:
  CLIP-B/32 gave TP sim=0.252 vs FP sim=0.243 → gap=0.009 (not discriminative).
  Root causes:
    1. Wrong model: CLIP-B/32 ≠ LLaVA's vision encoder (CLIP-L/14-336).
       Feature spaces don't align → text-patch similarity is noisy.
    2. Coarse crops: 17 manual crops of the image vs 576 native ViT patches.
       A small object (snowboard in corner) may not appear in any of the 17 crops.

  CLIP-L/14-336 approach:
    - Reuse LLaVA's already-loaded vision tower to get 576 patch features (free).
    - Load only CLIP-L text encoder + visual_projection head (~250MB fp16 total).
    - Features are in the EXACT same CLIP-L embedding space → maximum alignment.
    - 576 patches × 14px = covers entire 336px image at full resolution.
    - Expected: TP max_sim ≈ 0.30-0.45, FP max_sim ≈ 0.15-0.25 → gap ≥ 0.05.

MEMORY:
  LLaVA 4-bit:             ~6.50 GB
  CLIP-L text encoder:     ~0.25 GB fp16
  CLIP-L visual_proj:      ~0.002 GB fp16
  Total:                   ~6.75 GB  (fits in 8 GB with margin)

AUTHORS: Kesav Kumar J, Karthigeyan T
"""

import gc
import re

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPModel, CLIPTextModelWithProjection

CLIP_L_NAME = "openai/clip-vit-large-patch14-336"
# Sigmoid normalization: map CLIP-L per-patch max_sim → certainty [0, 1].
# Expected range: matching=0.30-0.45, non-matching=0.15-0.25 → center=0.27.
# (Calibrated per diagnostic; adjust if max_sim values differ.)
CLIP_L_SIG_CENTER = 0.27
CLIP_L_SIG_SCALE = 25.0

_NOUN_RE = re.compile(r"[Ii]s there an?\s+(.+?)\s+in the image", re.I)
_CLIP_L_CACHE: dict = {}


def pope_noun(question: str) -> str:
    m = _NOUN_RE.search(question)
    return m.group(1).strip() if m else question.rstrip("? ").split()[-1]


def load_clip_l_components(device) -> tuple:
    """
    Load the minimal CLIP-L components needed for per-patch grounding:
      - CLIPTextModelWithProjection (text encoder + text_projection, ~250MB fp16)
      - visual_projection (Linear 1024→768, ~1.5MB fp16)

    The CLIP-L vision encoder is NOT loaded separately - we reuse LLaVA's
    already-loaded model.vision_tower to get patch features at zero extra cost.
    visual_projection is extracted by loading the full CLIPModel on CPU,
    copying the projection weights, then immediately deleting the full model.
    """
    key = str(device)
    if key in _CLIP_L_CACHE:
        return _CLIP_L_CACHE[key]

    print(f"[CLIP-L] Loading text encoder from {CLIP_L_NAME} (~250MB fp16)...")
    text_model = CLIPTextModelWithProjection.from_pretrained(
        CLIP_L_NAME, torch_dtype=torch.float16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(CLIP_L_NAME)

    print("[CLIP-L] Extracting visual_projection head (~1.5MB)...")
    # Load full CLIPModel on CPU only to extract visual_projection weights.
    # Never moves to GPU to avoid OOM.
    full_clip_cpu = CLIPModel.from_pretrained(CLIP_L_NAME, torch_dtype=torch.float16)
    proj_weight = full_clip_cpu.visual_projection.weight.data.clone()  # [768, 1024]
    del full_clip_cpu
    gc.collect()

    visual_proj = nn.Linear(1024, 768, bias=False, dtype=torch.float16)
    visual_proj.weight = nn.Parameter(proj_weight)
    visual_proj = visual_proj.to(device).eval()

    _CLIP_L_CACHE[key] = (text_model, tokenizer, visual_proj)
    print("[CLIP-L] Ready. Additional VRAM used: ~252MB")
    return _CLIP_L_CACHE[key]


@torch.no_grad()
def get_llava_patch_features(llava_model, pixel_values) -> torch.Tensor:
    """
    Extract FINAL-layer patch features from LLaVA's CLIP-L vision tower.

    LLaVA internally uses hidden_states[-2] (penultimate layer) for visual tokens.
    For CLIP-grounded similarity we need the FINAL layer output (last_hidden_state)
    because that is what CLIP's visual_projection is calibrated against.

    Returns: float32 tensor [576, 1024]
    """
    vt = llava_model.vision_tower

    # Attempt 1: vt is a raw CLIPVisionModel - call with output_hidden_states=False
    # to get last_hidden_state.
    try:
        out = vt(pixel_values.half(), output_hidden_states=False)
        if hasattr(out, "last_hidden_state"):
            # Shape: [1, 577, 1024] → drop CLS → [576, 1024]
            return out.last_hidden_state[0, 1:].float()
    except Exception:
        pass

    # Attempt 2: vt is a LLaVA wrapper with a nested .vision_tower attribute.
    try:
        raw_vt = vt.vision_tower
        out = raw_vt(pixel_values.half(), output_hidden_states=False)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[0, 1:].float()
    except Exception:
        pass

    # Attempt 3: call with output_hidden_states=True and take the last element.
    try:
        out = vt(pixel_values.half(), output_hidden_states=True)
        if hasattr(out, "hidden_states") and out.hidden_states:
            return out.hidden_states[-1][0, 1:].float()
    except Exception:
        pass

    raise RuntimeError("Could not extract final-layer patch features from LLaVA vision tower.")


@torch.no_grad()
def clip_l_certainty(llava_model, processor, image, question: str, device: str) -> float:
    """
    CLIP-L grounded certainty: max cosine similarity between the object noun
    and LLaVA's 576 native CLIP-L patch features, mapped through a sigmoid.

    High certainty (≈1) → object likely present → apply minimal NO-bias → TP preserved.
    Low certainty (≈0) → object likely absent → apply full NO-bias → FP flipped.

    Args:
        llava_model: loaded LlavaForConditionalGeneration
        processor:   AutoProcessor for LLaVA
        image:       PIL Image (336×336 will be used)
        question:    POPE question "Is there a [NOUN] in the image?"
        device:      "cuda" or "cpu"

    Returns:
        certainty ∈ (0, 1)
    """
    text_model, tokenizer, visual_proj = load_clip_l_components(device)
    noun = pope_noun(question)

    # Use the image-only sub-processor - LlavaProcessor.image_processor is
    # the CLIPImageProcessor and does not require a text argument.
    pixel_values = processor.image_processor(
        images=image, return_tensors="pt"
    )["pixel_values"].to(device)

    # Vision: get 576 patch features from LLaVA's CLIP-L tower [576, 1024]
    patch_feats = get_llava_patch_features(llava_model, pixel_values)  # float32

    # Project to CLIP-L embedding space [576, 768]
    proj_feats = visual_proj(patch_feats.half()).float()
    proj_feats = proj_feats / proj_feats.norm(dim=-1, keepdim=True)

    # Text: encode noun with CLIP-L text encoder [1, 768]
    txt_enc = tokenizer(
        [noun], padding=True, truncation=True, max_length=77, return_tensors="pt"
    )
    txt_enc = {k: v.to(device) for k, v in txt_enc.items()}
    txt_feat = text_model(**txt_enc).text_embeds.float()  # [1, 768]
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    # Max patch-text cosine similarity [576]
    sims = (proj_feats @ txt_feat.T).squeeze(-1)
    max_sim = float(sims.max())
    top3_sim = float(sims.topk(min(3, sims.numel())).values.mean())

    certainty = float(
        torch.sigmoid(torch.tensor(CLIP_L_SIG_SCALE * (max_sim - CLIP_L_SIG_CENTER)))
    )

    print(
        f"  [CLIP-L] noun={noun!r} | "
        f"max_sim={max_sim:.3f} top3={top3_sim:.3f} | "
        f"certainty={certainty:.3f}"
    )
    return certainty


@torch.no_grad()
def run_diagnostic(llava_model, processor, samples: list, images: list, n: int = 15):
    """
    Quick diagnostic: compare CLIP-L max_sim for TP vs FP questions.
    Run this before the full eval to verify the feature space is discriminative.
    Returns dict with mean sim per category.
    """
    from evaluate import compute_f1
    from ugaa_hook import YES_TOKEN_IDS, NO_TOKEN_IDS, _get_yes_no_logits

    device = str(llava_model.device)
    text_model, tokenizer, visual_proj = load_clip_l_components(device)

    # First pass: determine category (TP/FP/TN/FN) from model predictions
    cats = {"TP": [], "FP": [], "TN": [], "FN": []}
    for i, (s, img) in enumerate(zip(samples, images)):
        if img is None:
            continue
        q = s["question"] + " Answer yes or no only."
        real_yes, real_no = _get_yes_no_logits(llava_model, processor, img, q, llava_model.device)
        pred = "yes" if (real_yes - real_no) > 0 else "no"
        label = s["label"]
        cat = ("TP" if pred == "yes" and label == "yes" else
               "FP" if pred == "yes" and label == "no" else
               "TN" if pred == "no" and label == "no" else "FN")
        cats[cat].append(i)

    results = {}
    for cat, idxs in cats.items():
        sims = []
        for idx in idxs[:n]:
            s, img = samples[idx], images[idx]
            if img is None:
                continue
            noun = pope_noun(s["question"])
            px = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
            pf = get_llava_patch_features(llava_model, px)
            pf_proj = visual_proj(pf.half()).float()
            pf_proj = pf_proj / pf_proj.norm(dim=-1, keepdim=True)
            te = tokenizer([noun], padding=True, truncation=True, max_length=77, return_tensors="pt")
            te = {k: v.to(device) for k, v in te.items()}
            tf = text_model(**te).text_embeds.float()
            tf = tf / tf.norm(dim=-1, keepdim=True)
            max_s = float((pf_proj @ tf.T).squeeze(-1).max())
            sims.append(max_s)

        if sims:
            mean_sim = sum(sims) / len(sims)
            results[cat] = {"n": len(sims), "mean_sim": mean_sim, "sims": sims}
            print(f"  {cat} (n={len(sims)}): mean_sim={mean_sim:.4f}")

    if "TP" in results and "FP" in results:
        gap = results["TP"]["mean_sim"] - results["FP"]["mean_sim"]
        print(f"\n  TP-FP sim gap: {gap:+.4f}  (CLIP-B/32 was +0.009)")
        print(f"  {'DISCRIMINATIVE ✓' if gap > 0.02 else 'NOT discriminative ✗'} "
              f"(threshold: >0.02)")

    return results
