"""
UGAA v5 - Spatial-Attention Certainty-Modulated NO-bias
========================================================

Diagnosis carried over from v1-v4:
  Baseline VSR: TP=48, TN=18, FP=31, FN=3.  79/100 predictions are "yes"
  against a true split of 51/49 - a language-prior YES-bias the model
  applies to linguistically plausible spatial statements.

Why v4 (VCD with blank image) failed:
  Blank image logits show yes≈10.9, no≈11.0 - the blank carries slight
  NO-bias, so `score - alpha * (blank_yes - blank_no)` *amplifies* the
  YES side instead of cancelling it.  The blank image is not a clean
  estimate of the language prior.

v5 (this file) - keep the working parts, drop the broken parts:
  Pass 1: probe LLaVA's own attention; measure spatial-word→patch
          attention concentration as normalized entropy (peaky → certain).
  Pass 2: run model with real image → get yes/no logits.
  Decision: score = (real_yes - real_no) - beta * (1 - certainty)
            predict "yes" if score > 0 else "no".

  Low certainty (diffuse attention, image does not localize the relation)
  applies a NO-push.  High certainty leaves the logits alone.

Differentiation vs. close papers:
  AdaptVis (ICML 2025): output-token confidence -> softmax temperature on
                        image attention.  Signal: text-side. UGAA: image-
                        side attention concentration.  Operator: logit
                        bias, not temperature scaling.
  VCD (Leng et al.):    fixed alpha, noise/blank image at logit level.
                        UGAA: no second image; per-question modulation
                        from the probe pass.
  Seo NeurIPS 2025:     CLIP-ViT deviation -> binary mask in VE.
                        UGAA: no perturbation, no gradient, LLM-side
                        attention signal, continuous gate.

Authors: Kesav Kumar J, Karthigeyan T
"""

import torch
import torch.nn as nn


SPATIAL_WORDS = {
    "left", "right", "above", "below", "behind", "front", "under",
    "over", "near", "far", "beside", "between", "inside", "outside",
    "top", "bottom", "side", "next", "ahead", "facing", "opposite",
    "beneath", "atop", "along", "across", "around", "through", "past",
    "back", "middle", "center", "edge", "corner", "adjacent", "away",
    "parallel", "within", "surrounding", "among", "beyond",
    "touching", "contains", "part", "connected", "on", "in", "at",
    "overlapping", "attached", "hanging",
}

# LLaMA tokenizer IDs for yes/no in LLaVA-1.5 (verified via AutoProcessor).
# After "ASSISTANT:" the model generates a space-prefixed token, so both
# the space-prefixed (" yes"=4874, " no"=694) and bare ("yes"=3582,
# "no"=1217) variants can be the top token depending on context.
# Capitalised forms ("Yes"=8241, "No"=3782, " Yes"=3869, " No"=1939) also
# appear.  We take max over all 4 yes-variants and all 4 no-variants so
# that capital or space-leading tokens are never misclassified.
YES_TOKEN_IDS = [3582, 8241, 4874, 3869]  # yes, Yes, ▁yes, ▁Yes
NO_TOKEN_IDS  = [1217, 3782,  694, 1939]  # no,  No,  ▁no,  ▁No


def _get_yes_no_logits(
    model: nn.Module,
    processor,
    image,
    question: str,
    device,
) -> tuple:
    """
    Single forward pass; returns (logit_yes, logit_no) as float scalars.
    Uses output_scores via generate(max_new_tokens=1) so we get the
    first-token distribution - most reliable for yes/no questions.

    Both lowercase and capitalised variants are included so that a model
    response of "No" (token 3782) does not register as positive.
    """
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # scores[0]: [batch, vocab_size] logits at position 0
    logits = out.scores[0][0].float()  # [vocab_size]
    # Take the maximum logit across all yes-variants and all no-variants.
    yes_vals = {i: logits[i].item() for i in YES_TOKEN_IDS}
    no_vals  = {i: logits[i].item() for i in NO_TOKEN_IDS}
    logit_yes = max(yes_vals.values())
    logit_no  = max(no_vals.values())
    return logit_yes, logit_no


class UGAAHook:
    """
    UGAA v5: Spatial-Attention Certainty-Modulated NO-bias.

    Two forward passes (probe + real). No blank image, no contrastive
    subtraction. The probe pass measures attention concentration from
    spatial words to visual patches; low concentration -> diffuse signal
    -> the image does not support a confident YES -> add a NO-bias to the
    (yes - no) logit gap, scaled by (1 - certainty).

        score = (real_yes - real_no) - beta * (1 - certainty)
        pred  = "yes" if score > 0 else "no"

    Usage:
        ugaa = UGAAHook(beta=0.5)
        pred = ugaa.infer(model, processor, image, question)
    """

    def __init__(self, beta: float = 0.5):
        """
        Args:
            beta: NO-bias strength when the image probe is uninformative.
                  Effective bias applied to (yes - no) is `beta * (1 - certainty)`.
                  Sweep range: {0.3, 0.5, 0.8, 1.0, 1.5}.
        """
        self.beta = beta

    # ------------------------------------------------------------------
    # Pass 1: Spatial certainty from LLaVA's own attention
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_spatial_certainty(
        self,
        model: nn.Module,
        processor,
        image,
        question: str,
        visual_start: int = 1,
        visual_end: int = 577,
    ) -> float:
        """
        Returns a scalar certainty in [0, 1].
          1.0 = model's spatial-word attention concentrates on specific patches
                (high certainty - leave logits alone)
          0.0 = attention is diffuse / no spatial words found
                (low certainty - apply maximum NO-bias)
        """
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        token_ids = inputs["input_ids"][0]
        tokens = processor.tokenizer.convert_ids_to_tokens(token_ids.tolist())
        spatial_positions = [
            i for i, t in enumerate(tokens)
            if t.lstrip("▁").lower() in SPATIAL_WORDS
        ]

        if not spatial_positions:
            print("[UGAA] No spatial words → certainty=0.0 (max NO-bias)")
            return 0.0

        try:
            model.language_model.config._attn_implementation = "eager"
            outputs = model(**inputs, output_attentions=True, return_dict=True)
        except Exception as e:
            print(f"[UGAA] Probe failed ({e}) → certainty=0.5")
            return 0.5
        finally:
            model.language_model.config._attn_implementation = "sdpa"

        attentions = outputs.attentions
        if attentions is None:
            return 0.5

        spatial_pos_tensor = torch.tensor(spatial_positions, dtype=torch.long)
        per_patch_layers = []

        for layer_attn in attentions:
            if layer_attn is None:
                continue
            a = layer_attn[0].float().cpu()  # [heads, seq, seq]
            # attention from spatial words to visual patches
            sw_to_vis = a[:, spatial_pos_tensor, visual_start:visual_end]
            # [heads, n_spatial, n_visual] → mean over heads and spatial words
            per_patch_layers.append(sw_to_vis.mean(dim=(0, 1)))  # [n_visual]

        if not per_patch_layers:
            return 0.5

        # Average per-patch attention across all 32 LLM decoder layers.
        # Top-k/mean ratio was constant (~20.6) across all VSR questions
        # because it's dominated by patch-distribution shape, not by content.
        # Normalized entropy varies per question and has a principled meaning:
        # peaky distribution → low entropy → high certainty.
        per_patch = torch.stack(per_patch_layers, dim=0).mean(dim=0)  # [n_visual]
        p = per_patch / (per_patch.sum() + 1e-9)
        H = -(p * (p + 1e-9).log()).sum()
        H_max = torch.log(torch.tensor(float(p.numel())))
        certainty = float((1.0 - H / H_max).clamp(0.0, 1.0))

        print(
            f"[UGAA] spatial_words={len(spatial_positions)} | "
            f"H={H.item():.3f}/{H_max.item():.3f} | certainty={certainty:.3f}"
        )
        return certainty

    # ------------------------------------------------------------------
    # Main inference: probe + real image + certainty-modulated NO-bias
    # ------------------------------------------------------------------

    def infer(
        self,
        model: nn.Module,
        processor,
        image,
        question: str,
        visual_start: int = 1,
        visual_end: int = 577,
    ) -> str:
        """
        Full UGAA v5 inference for a yes/no question.

        Returns "yes" or "no".
        """
        device = model.device

        # Pass 1: spatial certainty from the probe forward.
        certainty = self.compute_spatial_certainty(
            model, processor, image, question, visual_start, visual_end
        )

        # Pass 2: real image yes/no logits.
        real_yes, real_no = _get_yes_no_logits(
            model, processor, image, question, device
        )

        # Decision: certainty-modulated NO-bias on the (yes - no) gap.
        raw_gap = real_yes - real_no
        no_bias = self.beta * (1.0 - certainty)
        score = raw_gap - no_bias

        pred = "yes" if score > 0 else "no"

        print(
            f"[UGAA] real=({real_yes:.2f},{real_no:.2f}) | "
            f"certainty={certainty:.3f} | "
            f"β(1-c)={no_bias:.3f} | score={score:.3f} → {pred}"
        )
        return pred

    def remove(self) -> None:
        """No-op for API compatibility with eval scripts."""
        pass
