"""
UGAA v6 — Uncertainty-Gated Asymmetric Attention (Inverted Operator)
=====================================================================

Empirical diagnosis from v1–v5 (Kesav + Karthigeyan, May 17–23):
    Every NO-bias mechanism — entropy, magnitude, CLS, uniform, object-noun,
    CLIP-B/32, CLIP-L/14-336 — HURT POPE adversarial F1:
        baseline           F1=0.8221  P=0.8658  R=0.7827
        v5 entropy β=1.0   F1=0.8164  (-0.0057)   P↑ but R↓↓
        CLIP-B β=1.0       F1=0.8171  (-0.0050)
        CLIP-B β=1.5       F1=0.8104  (-0.0117)
    Precision rose, recall collapsed. The operator was the bottleneck — not
    the signal. Suppressing YES uniformly flips legitimate TPs to FNs faster
    than it catches FPs.

v6 design:
    1. INVERT the operator. Boost YES when the model is grounded
       (high decoder-attention on image tokens + diverse VE patches).
       Weakly suppress YES only when the model is inattentive AND scene
       is uniform — those are the bona fide hallucination prerequisites.
    2. CONFIDENCE GATE. Never modify logits when |yes - no| is large.
       Confident TPs and TNs stay untouched → cannot be flipped to errors.
       Only uncertain decisions (small gap) are intervened on.
    3. CORRECT visual-token positions. v5 used positions 1-577; the
       <image> token in LLaVA-1.5 expands at position 4 (after BOS, USER,
       :, space) → image tokens occupy 4-579. Dynamically detected from
       input_ids each call.
    4. PENULTIMATE VE layer. LLaVA-1.5 routes vision_tower's
       hidden_states[-2] (not last_hidden_state) to the LLM. Hook
       encoder.layers[-2]'s output.
    5. MID-LAYER decoder attention (14-20). run_ablation_a.py:348
       documents these as peak visual-grounding layers; layer 22 reflects
       next-token linguistics, not grounding.

Token IDs are the 8-token standard verified by run_pope_2tok_baseline.py
on the full 3000-question POPE — the SAME standard used by the published
baselines (F1=0.8221 adversarial / 0.8498 popular / 0.8713 random).

Authors: Kesav Kumar J, Karthigeyan T
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration


YES_TOKEN_IDS: List[int] = [3582, 8241, 4874, 3869]  # yes, Yes, ▁yes, ▁Yes
NO_TOKEN_IDS: List[int] = [1217, 3782, 694, 1939]    # no,  No,  ▁no,  ▁No
LLAVA_IMAGE_TOKEN_ID: int = 32000                    # LLaVA-1.5 placeholder id
LLAVA_N_IMAGE_TOKENS: int = 576                      # 24×24 ViT-L/14-336

# Decoder layers averaged for the grounding signal. Empirically validated
# (see run_ablation_a.py:348): layers 14–20 carry the peak text→image
# attention in LLaVA-1.5; final layers (22+) collapse onto next-token
# linguistics. Configurable but DO NOT change without re-tuning.
DEFAULT_GROUNDING_LAYERS: Tuple[int, int] = (14, 20)


def load_llava_with_gate(model_path: str = "D:/models/llava-1.5-7b"):
    """Load LLaVA-1.5 in 4-bit + eager attention. Eager is mandatory for
    output_attentions to return tensors (SDPA returns None and the gate
    falls back to neutral, silently breaking)."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def locate_image_tokens(input_ids: torch.Tensor) -> Tuple[int, int]:
    """Find image-token span in the EXPANDED sequence.

    Returns (visual_start, visual_end) such that positions [visual_start,
    visual_end) in the post-expansion hidden_states / attention are the
    576 image tokens. Falls back to (4, 580) for the standard LLaVA-1.5
    "USER: <image>" template.
    """
    ids = input_ids[0].tolist() if input_ids.dim() == 2 else input_ids.tolist()
    if LLAVA_IMAGE_TOKEN_ID in ids:
        start = ids.index(LLAVA_IMAGE_TOKEN_ID)
        return start, start + LLAVA_N_IMAGE_TOKENS
    # Fallback for the canonical "USER: <image>" prompt
    return 4, 4 + LLAVA_N_IMAGE_TOKENS


class UGAAGateV6:
    """Confidence-gated asymmetric YES-boost / weak NO-tilt.

    Inputs (per question):
        ve_patches:       (576, 1024) from encoder.layers[-2] forward hook
        attn_layers_14_20: tensor stack (n_layers, n_heads, seq, seq)
        logits:           (vocab_size,) raw decoder logits for next token
        input_ids:        (1, seq_len) — used to find image-token span

    Operator:
        gap         = max(logits[YES]) - max(logits[NO])
        |gap| ≥ confidence_margin →   NO intervention (zone='CONFIDENT')
        |gap| <  confidence_margin →   intervene per grounding zone:
            grounding ≥ high_thresh AND complexity ≥ complexity_min:
                logits[YES] += alpha · strength       zone='HIGH'
            grounding ≤ low_thresh AND complexity ≤ complexity_max:
                logits[YES] -= beta  · strength       zone='LOW'
            else:                                     zone='NEUTRAL'
    """

    def __init__(
        self,
        model,
        alpha: float = 1.5,
        beta: float = 0.5,
        high_thresh: float = 0.015,
        low_thresh: float = 0.005,
        confidence_margin: float = 1.5,
        complexity_min: float = 0.20,
        complexity_max: float = 0.80,
        layers: Tuple[int, int] = DEFAULT_GROUNDING_LAYERS,
    ):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.confidence_margin = confidence_margin
        self.complexity_min = complexity_min
        self.complexity_max = complexity_max
        self.layer_lo, self.layer_hi = layers
        self._ve_patches: Optional[torch.Tensor] = None
        self._hooks: list = []

    # ------------------------------------------------------------------ hooks
    def register_hooks(self) -> None:
        """Hook the second-to-last vision-encoder layer to capture the
        hidden_states LLaVA actually routes to the LLM (vision_feature_layer=-2).
        Decoder attention is read from the forward outputs in `process()` —
        no decoder hook needed (avoids per-layer hook overhead)."""
        ve_layers = self.model.vision_tower.vision_model.encoder.layers
        target = ve_layers[-2]

        def ve_hook(_module, _inp, output):
            # output is a tuple; output[0] is hidden_states (1, 577, 1024).
            # Drop CLS at index 0 to leave 576 patches.
            hs = output[0]
            self._ve_patches = hs[:, 1:, :].detach()

        self._hooks.append(target.register_forward_hook(ve_hook))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ------------------------------------------------------------------ signals
    @staticmethod
    def _scene_complexity(patches: torch.Tensor) -> float:
        """Pairwise cosine distance across a 64-patch subsample of the
        576 ViT patches. High = diverse scene; low = uniform.

        Subsample is deterministic (stride 9) so per-question complexity
        is reproducible. Full 576x576 would cost 331K entries ≈ 0.66 MB
        fp16; 64x64 = 4 KB. The mean over the upper triangle is the
        scalar 'how different are random pairs of patches'."""
        if patches is None:
            return 0.5
        patches = patches[0] if patches.dim() == 3 else patches
        idx = torch.arange(0, patches.shape[0], 9, device=patches.device)
        sampled = F.normalize(patches[idx].float(), dim=-1)
        sim = sampled @ sampled.T
        dist = 1.0 - sim
        n = dist.shape[0]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dist.device), diagonal=1)
        raw = dist[mask].mean().item()
        # Normalize: empirically COCO patches → raw ≈ 0.05–0.45
        return float(min(max((raw - 0.05) / 0.40, 0.0), 1.0))

    @staticmethod
    def _grounding_score(
        attentions: Tuple[torch.Tensor, ...],
        visual_start: int,
        visual_end: int,
        layer_lo: int,
        layer_hi: int,
    ) -> float:
        """Mean attention from the LAST token (the position that predicts
        yes/no) to the image-token span, averaged across heads and the
        mid-layer range (layer_lo..layer_hi inclusive).

        The last token's outgoing attention to image patches is the
        per-question scalar 'how much is the model looking at the image
        when deciding'. Higher = more visually grounded prediction.
        """
        if attentions is None or len(attentions) <= layer_hi:
            return 0.0
        # Slice the layer range; each element is (1, n_heads, seq, seq)
        per_layer = []
        for li in range(layer_lo, layer_hi + 1):
            a = attentions[li]
            if a is None:
                continue
            seq = a.shape[-1]
            last = seq - 1
            img_s = min(visual_start, seq - 1)
            img_e = min(visual_end, seq)
            if img_e <= img_s:
                continue
            # mean over heads of attention from `last` token to image span
            v = a[0, :, last, img_s:img_e].float().mean().item()
            per_layer.append(v)
        if not per_layer:
            return 0.0
        return sum(per_layer) / len(per_layer)

    # ------------------------------------------------------------------ gate
    @staticmethod
    def _yesno_max(logits: torch.Tensor) -> Tuple[float, float]:
        yes = max(logits[i].item() for i in YES_TOKEN_IDS)
        no = max(logits[i].item() for i in NO_TOKEN_IDS)
        return yes, no

    def process(
        self,
        logits: torch.Tensor,
        attentions: Tuple[torch.Tensor, ...],
        input_ids: torch.Tensor,
    ) -> dict:
        """Apply the confidence-gated asymmetric operator.

        Returns a diagnostic dict with raw + modified yes/no maxes,
        grounding_score, complexity, zone, and the final prediction.
        """
        # 1. Locate image span in this question's expanded sequence.
        visual_start, visual_end = locate_image_tokens(input_ids)

        # 2. Pre-gate baseline decision (used for both confidence check
        #    and final return if no intervention).
        logits_f = logits.float().clone()
        yes_raw, no_raw = self._yesno_max(logits_f)
        gap_raw = yes_raw - no_raw
        baseline_pred = "yes" if gap_raw > 0 else "no"

        # 3. Compute signals (only relevant if we might intervene).
        complexity = self._scene_complexity(self._ve_patches)
        grounding = self._grounding_score(
            attentions, visual_start, visual_end, self.layer_lo, self.layer_hi
        )

        # 4. Confidence gate: only intervene on uncertain decisions.
        zone = "CONFIDENT"
        modified = logits_f
        if abs(gap_raw) < self.confidence_margin:
            # 5. Asymmetric grounding-based intervention.
            if grounding >= self.high_thresh and complexity >= self.complexity_min:
                # Grounded prediction → boost YES toward truthful detection.
                strength = (grounding - self.high_thresh) / max(1.0 - self.high_thresh, 1e-6)
                for tok in YES_TOKEN_IDS:
                    modified[tok] = modified[tok] + self.alpha * strength
                zone = "HIGH"
            elif grounding <= self.low_thresh and complexity <= self.complexity_max:
                # Inattentive + uniform scene → mild NO tilt.
                strength = (self.low_thresh - grounding) / max(self.low_thresh, 1e-6)
                for tok in YES_TOKEN_IDS:
                    modified[tok] = modified[tok] - self.beta * strength
                zone = "LOW"
            else:
                zone = "NEUTRAL"

        # 6. Final decision after potential modification.
        yes_mod, no_mod = self._yesno_max(modified)
        pred = "yes" if yes_mod > no_mod else "no"

        # Clear VE patches so the next question gets fresh values
        # (and any rare hook miss is detectable as None).
        self._ve_patches = None

        return {
            "prediction": pred,
            "baseline_prediction": baseline_pred,
            "yes_raw": yes_raw,
            "no_raw": no_raw,
            "yes_mod": yes_mod,
            "no_mod": no_mod,
            "gap_raw": gap_raw,
            "grounding": grounding,
            "complexity": complexity,
            "zone": zone,
            "visual_span": (visual_start, visual_end),
        }
