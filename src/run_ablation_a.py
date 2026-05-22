# src/run_ablation_a.py — KESAV + KARTHIGEYAN
# Ablation A: signal-swap on POPE at fixed beta (default 1.0).
#
# Five certainty variants, all plugged into the same downstream pipeline:
#   (i)   entropy    — UGAA: normalized entropy of spatial-word→patch attention
#   (ii)  magnitude  — mean L1 norm of spatial-word→patch attention (no normalization)
#   (iii) cls        — CLS-token→patch attention (DAMRO-style, no spatial words)
#   (iv)  uniform    — constant 0.5 (ablates the modulation; tests pure NO-bias)
#   (v)   object     — normalized entropy of OBJECT-NOUN→patch attention
#                      (uses the noun being asked about, not generic spatial words)
#                      Hypothesis: object-noun attention concentration reflects whether
#                      the object is visually grounded, directly predicting hallucination.
#
# All variants use the same 8-token yes/no logit extraction.
# Purpose: isolate contribution of the UGAA signal vs. any uniform NO-bias.
# Critical finding from runs 1–4: entropy == uniform (both F1=0.8211).
# The object variant tests whether a more localizable probe token helps.
#
# Usage:
#   python src/run_ablation_a.py [--variant entropy|magnitude|cls|uniform|object|all]
#   python src/run_ablation_a.py --variant object --beta 1.0

import argparse
import json
import os
import re
import sys

import requests
import torch
from PIL import Image
from io import BytesIO
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    CLIPModel,
    CLIPProcessor,
    LlavaForConditionalGeneration,
)

sys.path.insert(0, os.path.dirname(__file__))
from ugaa_hook import (
    SPATIAL_WORDS,
    YES_TOKEN_IDS as YES_TOKEN_IDS_LOCAL,
    NO_TOKEN_IDS as NO_TOKEN_IDS_LOCAL,
    _get_yes_no_logits,
)
from evaluate import compute_f1

MODEL_PATH = "D:/models/llava-1.5-7b"
DATASET = "datasets/pope/pope_sample_100.json"
DEFAULT_BETA = 1.0
VISUAL_START = 1
VISUAL_END = 577
VARIANTS = ["entropy", "magnitude", "cls", "uniform", "object"]

# Stopwords for extracting object nouns from POPE questions.
# "Is there a [OBJECT] in the image?" → remove these, keep the object noun(s).
POPE_STOPWORDS = {"is", "there", "a", "an", "in", "the", "image"}

# ---------------------------------------------------------------------------
# CLIP certainty — object presence signal
# ---------------------------------------------------------------------------
# Uses openai/clip-vit-base-patch32 (~290MB fp16) loaded once alongside LLaVA.
# Signal: max cosine similarity between the object noun and a 4×4 crop grid
# of the image, mapped through a sigmoid to [0, 1]:
#   sim ≈ 0.30 (object present)  → certainty ≈ 0.83 → bias ≈ 0.17  (TP preserved)
#   sim ≈ 0.15 (object absent)   → certainty ≈ 0.11 → bias ≈ 0.89  (FP pushed to NO)
# Contrast with constant beta=1.0 (certainty=0.5 for ALL → bias=0.5 for ALL),
# which broke recall on 3000 questions by treating TPs and FPs identically.

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# Sigmoid center: ~0.22 is CLIP-B/32's zero-shot object-presence threshold.
# Scale 20 gives steep transition: sim=0.30→certainty=0.83, sim=0.15→certainty=0.11.
CLIP_SIG_CENTER = 0.22
CLIP_SIG_SCALE = 20.0
_clip_cache: dict = {}

_NOUN_RE = re.compile(r"[Ii]s there an?\s+(.+?)\s+in the image", re.I)


def _pope_noun(question: str) -> str:
    """Extract the object noun from a POPE question."""
    m = _NOUN_RE.search(question)
    if m:
        return m.group(1).strip()
    # Fallback: strip stopwords
    words = question.lower().rstrip("?").split()
    result = " ".join(w for w in words if w not in POPE_STOPWORDS)
    return result if result else "object"


def _get_clip(device):
    """Lazy-load CLIP-B/32 in fp16 alongside LLaVA. Cached after first call."""
    key = str(device)
    if key not in _clip_cache:
        print(f"[CLIP] Loading {CLIP_MODEL_NAME} (fp16)...")
        model = CLIPModel.from_pretrained(
            CLIP_MODEL_NAME, torch_dtype=torch.float16
        ).to(device).eval()
        proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_cache[key] = (model, proc)
        print("[CLIP] Ready.")
    return _clip_cache[key]


@torch.no_grad()
def clip_certainty_for_question(image: Image.Image, question: str, device) -> float:
    """CLIP-grounded object certainty.

    Crops the image into a 4×4 grid plus the whole image (17 crops total),
    computes CLIP cosine similarity between each crop and the object noun,
    and maps the max similarity through a sigmoid to get certainty ∈ [0, 1].
    """
    clip_model, clip_proc = _get_clip(device)
    noun = _pope_noun(question)

    # 4×4 crops + whole image
    w, h = image.size
    pw, ph = max(w // 4, 1), max(h // 4, 1)
    crops = [
        image.crop((c * pw, r * ph, min((c + 1) * pw, w), min((r + 1) * ph, h)))
        for r in range(4) for c in range(4)
    ]
    crops.append(image)  # global similarity as additional signal

    img_inputs = clip_proc(images=crops, return_tensors="pt", padding=True)
    img_inputs = {
        k: v.to(device).half() for k, v in img_inputs.items()
        if isinstance(v, torch.Tensor)
    }
    img_feats = clip_model.get_image_features(**img_inputs).float()
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

    txt_inputs = clip_proc(text=[noun], return_tensors="pt", padding=True)
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items() if isinstance(v, torch.Tensor)}
    txt_feat = clip_model.get_text_features(**txt_inputs).float()
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    sims = (img_feats @ txt_feat.T).squeeze(-1).cpu()
    max_sim = float(sims.max())

    # Sigmoid: center=0.22, scale=20 → steep transition around CLIP's match threshold
    certainty = float(torch.sigmoid(torch.tensor(CLIP_SIG_SCALE * (max_sim - CLIP_SIG_CENTER))))
    print(f"  [CLIP] noun={noun!r} | max_sim={max_sim:.3f} | certainty={certainty:.3f}")
    return certainty


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_llava():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


# ---------------------------------------------------------------------------
# Certainty signal implementations
# ---------------------------------------------------------------------------

@torch.no_grad()
def _probe_attentions(model, processor, image, question):
    """Run a probe forward pass; return (attentions, tokens, spatial_positions).

    tokens: list of token strings from the tokenizer (for variant-specific position lookup)
    spatial_positions: pre-computed SPATIAL_WORDS positions (used by entropy/magnitude)
    Returns (None, [], []) on failure.
    """
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    token_ids = inputs["input_ids"][0]
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids.tolist())
    spatial_positions = [
        i for i, t in enumerate(tokens)
        if t.lstrip("▁").lower() in SPATIAL_WORDS
    ]

    try:
        model.language_model.config._attn_implementation = "eager"
        outputs = model(**inputs, output_attentions=True, return_dict=True)
    except Exception as e:
        print(f"  [probe] failed: {e}")
        return None, [], []
    finally:
        model.language_model.config._attn_implementation = "sdpa"

    return outputs.attentions, tokens, spatial_positions


def _object_positions(tokens: list, question: str) -> list:
    """Return token positions corresponding to the object noun in the question.

    For POPE: "Is there a [OBJECT] in the image?" — strips POPE_STOPWORDS and
    matches the remaining content words against token strings.
    """
    words = question.lower().rstrip("?").split()
    obj_words = {w.strip(".,!?;:") for w in words} - POPE_STOPWORDS
    return [
        i for i, t in enumerate(tokens)
        if t.lstrip("▁").lower() in obj_words
    ]


def _entropy_from_positions(attentions, positions, visual_start, visual_end):
    """Shared helper: normalized entropy of attention from positions → visual patches."""
    if not positions or attentions is None:
        return 0.0
    pos_tensor = torch.tensor(positions, dtype=torch.long)
    per_patch_layers = []
    for layer_attn in attentions:
        if layer_attn is None:
            continue
        a = layer_attn[0].float().cpu()
        tok_to_vis = a[:, pos_tensor, visual_start:visual_end]
        per_patch_layers.append(tok_to_vis.mean(dim=(0, 1)))
    if not per_patch_layers:
        return 0.5
    per_patch = torch.stack(per_patch_layers).mean(dim=0)
    p = per_patch / (per_patch.sum() + 1e-9)
    H = -(p * (p + 1e-9).log()).sum()
    H_max = torch.log(torch.tensor(float(p.numel())))
    return float((1.0 - H / H_max).clamp(0.0, 1.0))


# All certainty functions share signature: (attentions, tokens, spatial_positions, vs, ve, question)

def certainty_entropy(attentions, tokens, spatial_positions, visual_start, visual_end, question):
    """UGAA v5: normalized entropy of spatial-word→patch attention."""
    return _entropy_from_positions(attentions, spatial_positions, visual_start, visual_end)


def certainty_magnitude(attentions, tokens, spatial_positions, visual_start, visual_end, question):
    """Baseline: mean L1 magnitude of spatial-word→patch attention (no normalization)."""
    if not spatial_positions or attentions is None:
        return 0.0
    pos_tensor = torch.tensor(spatial_positions, dtype=torch.long)
    mags = []
    for layer_attn in attentions:
        if layer_attn is None:
            continue
        a = layer_attn[0].float().cpu()
        sw_to_vis = a[:, pos_tensor, visual_start:visual_end]
        mags.append(sw_to_vis.mean().item())
    if not mags:
        return 0.5
    raw = sum(mags) / len(mags)
    uniform_val = 1.0 / (visual_end - visual_start)
    return float(min(raw / (10 * uniform_val), 1.0))


def certainty_cls(attentions, tokens, spatial_positions, visual_start, visual_end, question):
    """DAMRO-style: CLS (position 0) → visual patch attention entropy."""
    if attentions is None:
        return 0.5
    per_patch_layers = []
    for layer_attn in attentions:
        if layer_attn is None:
            continue
        a = layer_attn[0].float().cpu()
        per_patch_layers.append(a[:, 0, visual_start:visual_end].mean(dim=0))
    if not per_patch_layers:
        return 0.5
    per_patch = torch.stack(per_patch_layers).mean(dim=0)
    p = per_patch / (per_patch.sum() + 1e-9)
    H = -(p * (p + 1e-9).log()).sum()
    H_max = torch.log(torch.tensor(float(p.numel())))
    return float((1.0 - H / H_max).clamp(0.0, 1.0))


def certainty_uniform(attentions, tokens, spatial_positions, visual_start, visual_end, question):
    """Control: constant 0.5 — tests if any result is from modulation vs pure bias."""
    return 0.5


def certainty_object(attentions, tokens, spatial_positions, visual_start, visual_end, question):
    """UGAA object variant: normalized entropy of OBJECT-NOUN→patch attention.

    Uses the noun being asked about ("Is there a snowboard?") instead of
    generic spatial words like "in"/"left". Object words are more localizable:
    when the object is present, its patches attract higher attention
    (lower entropy → higher certainty → less NO-bias applied).
    """
    obj_positions = _object_positions(tokens, question)
    if not obj_positions:
        return 0.5
    return _entropy_from_positions(attentions, obj_positions, visual_start, visual_end)


CERTAINTY_FNS = {
    "entropy":   certainty_entropy,
    "magnitude": certainty_magnitude,
    "cls":       certainty_cls,
    "uniform":   certainty_uniform,
    "object":    certainty_object,
}


# ---------------------------------------------------------------------------
# Generation-time attention variant (single-pass)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_logits_and_gen_attn(model, processor, image, question, device):
    """Single generate() call returning both first-token logits and per-layer
    visual attention at the moment the yes/no token is generated.

    output_attentions=True triggers SDPA→eager fallback automatically.
    Shape: attentions[0][layer_i] = [batch, heads, 1, seq_len]
    """
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=True,
    )

    logits = out.scores[0][0].float()
    logit_yes = max(logits[i].item() for i in YES_TOKEN_IDS_LOCAL)
    logit_no = max(logits[i].item() for i in NO_TOKEN_IDS_LOCAL)

    per_layer = []
    if out.attentions is not None:
        for layer_attn in out.attentions[0]:
            a = layer_attn[0].float().cpu()  # [heads, 1, seq]
            vis = a[:, 0, VISUAL_START:VISUAL_END]  # [heads, n_visual]
            per_layer.append(vis.mean(dim=0))  # [n_visual]

    return logit_yes, logit_no, per_layer


def _gen_attn_certainty(per_layer, layer_start=14, layer_end=20):
    """Entropy of mid-layer (14–20) generation-time visual attention.

    Opus diagnosis: layers 14–20 show peak visual grounding in LLaVA-1.5.
    Final layers are dominated by next-token linguistics; averaging all 32 washes signal.
    """
    if not per_layer or len(per_layer) < layer_end:
        return 0.5
    layers = per_layer[layer_start:layer_end]
    per_patch = torch.stack(layers).mean(dim=0)
    p = per_patch / (per_patch.sum() + 1e-9)
    H = -(p * (p + 1e-9).log()).sum()
    H_max = torch.log(torch.tensor(float(p.numel())))
    return float((1.0 - H / H_max).clamp(0.0, 1.0))


# ---------------------------------------------------------------------------
# VCD noise variant (2-pass: real + gaussian noise image)
# ---------------------------------------------------------------------------

import numpy as np

def _make_noise_image():
    """Uniform random RGB noise image — zero systematic yes/no bias.
    Gray (v4) had slight NO-bias (blank_no > blank_yes), amplifying YES.
    Random noise has zero mean bias, giving a clean language-prior estimate.
    """
    arr = np.random.randint(0, 256, (336, 336, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def infer_vcd_noise(model, processor, image, question, beta, device):
    """VCD with Gaussian noise: score = (real_gap) - alpha*(noise_gap).
    noise_gap estimates the language-prior contribution.
    alpha = beta (reuse same hyperparameter for sweep comparability).
    """
    real_yes, real_no = _get_yes_no_logits(model, processor, image, question, device)
    noise_img = _make_noise_image()
    noise_yes, noise_no = _get_yes_no_logits(model, processor, noise_img, question, device)
    score = (real_yes - real_no) - beta * (noise_yes - noise_no)
    return "yes" if score > 0 else "no"


# ---------------------------------------------------------------------------
# Per-question inference — dispatches by variant name
# ---------------------------------------------------------------------------

def infer_with_variant(model, processor, image, question, variant, beta, device):
    """Dispatch inference to the right path for each variant."""
    if variant == "generation_mid":
        logit_yes, logit_no, per_layer = _get_logits_and_gen_attn(
            model, processor, image, question, device
        )
        certainty = _gen_attn_certainty(per_layer)
        score = (logit_yes - logit_no) - beta * (1.0 - certainty)
        return "yes" if score > 0 else "no"

    if variant == "vcd_noise":
        return infer_vcd_noise(model, processor, image, question, beta, device)

    if variant == "clip_certainty":
        certainty = clip_certainty_for_question(image, question, device)
        real_yes, real_no = _get_yes_no_logits(model, processor, image, question, device)
        score = (real_yes - real_no) - beta * (1.0 - certainty)
        return "yes" if score > 0 else "no"

    # Probe-based variants (entropy, magnitude, cls, uniform, object)
    attentions, tokens, spatial_positions = _probe_attentions(
        model, processor, image, question
    )
    fn = CERTAINTY_FNS[variant]
    certainty = fn(attentions, tokens, spatial_positions, VISUAL_START, VISUAL_END, question)
    real_yes, real_no = _get_yes_no_logits(model, processor, image, question, device)
    score = (real_yes - real_no) - beta * (1.0 - certainty)
    return "yes" if score > 0 else "no"


SPECIAL_VARIANTS = {"generation_mid", "vcd_noise", "clip_certainty"}
ALL_VARIANTS = list(CERTAINTY_FNS.keys()) + sorted(SPECIAL_VARIANTS)


# ---------------------------------------------------------------------------
# Run one variant over all 100 POPE questions
# ---------------------------------------------------------------------------

def run_variant(variant: str, samples, images, model, processor, beta: float = DEFAULT_BETA):
    device = model.device
    print(f"\n{'='*50}\nVariant: {variant} | beta={beta}\n{'='*50}\n")

    results = []
    for i, item in enumerate(samples):
        question = item["question"] + " Answer yes or no only."
        print(f"[{i+1}/{len(samples)}] {item['question']}")
        image = images[i]
        if image is None:
            pred = "error"
        else:
            try:
                pred = infer_with_variant(
                    model, processor, image, question, variant, beta, device
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                pred = "error"
        results.append({
            "question_id": item.get("question_id", i + 1),
            "question": item["question"],
            "label": item["label"],
            "prediction": pred,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=ALL_VARIANTS + ["all"],
        default="all",
        help="Which certainty signal to test (default: all).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help=f"NO-bias strength (default: {DEFAULT_BETA}).",
    )
    args = parser.parse_args()
    variants_to_run = ALL_VARIANTS if args.variant == "all" else [args.variant]

    print("Loading LLaVA...")
    model, processor = load_llava()
    print("Model loaded.\n")

    with open(DATASET) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} POPE questions.")

    print("Pre-fetching images...")
    images = []
    for i, item in enumerate(samples):
        try:
            img_data = requests.get(item["image_url"], timeout=10).content
            images.append(Image.open(BytesIO(img_data)).convert("RGB"))
        except Exception as e:
            print(f"  [{i+1}] fetch failed: {e}")
            images.append(None)
    print(f"Cached {sum(1 for x in images if x is not None)}/{len(samples)} images.\n")

    os.makedirs("experiments", exist_ok=True)
    summary = {}

    for variant in variants_to_run:
        results = run_variant(variant, samples, images, model, processor, beta=args.beta)

        path = f"experiments/ablation_a_{variant}_b{args.beta}_predictions.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        preds = [r["prediction"] for r in results]
        labels = [r["label"] for r in results]
        metrics = compute_f1(preds, labels)
        summary[variant] = metrics
        print(f"\n[{variant}] F1={metrics['f1']} P={metrics['precision']} R={metrics['recall']}")

    print(f"\n========== ABLATION A SUMMARY (POPE, beta={args.beta}) ==========")
    print(f"Baseline:  F1=0.8041  P=0.8298  R=0.7800")
    print(f"UGAA v5:   F1=0.8211  P=0.8667  R=0.7800  (entropy, beta=1.0, reference)")
    for variant, m in summary.items():
        delta = m["f1"] - 0.8041
        print(
            f"{variant:12s}: F1={m['f1']:.4f}  P={m['precision']:.4f}  "
            f"R={m['recall']:.4f}  delta_F1={delta:+.4f}"
        )


if __name__ == "__main__":
    main()
