"""
UGAA (Uncertainty-Guided Attention Asymmetry) mechanism.

Modulates transformer attention using token-level uncertainty and spatial
distance priors. Uncertain tokens have their attention redistributed toward
visually proximate patches via a learned asymmetric mask.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class UGAAHook:
    """Implements the UGAA mechanism as a forward hook on a transformer layer."""

    def __init__(self, gamma: float = 0.5, tau: float = 0.5):
        self.gamma = gamma
        self.tau = tau
        self._hook_handle: Optional[torch.utils.hooks.RemovableHook] = None

    def compute_uncertainty(self, token_logits: torch.Tensor) -> torch.Tensor:
        """
        Computes per-token entropy from logit distributions.

        Args:
            token_logits: [seq_len, vocab_size] or [batch, seq_len, vocab_size]

        Returns:
            entropy: [seq_len] or [batch, seq_len]
        """
        probs = torch.softmax(token_logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def compute_gate(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Computes gating values from uncertainty scores.

        Args:
            uncertainty: [seq_len] entropy values

        Returns:
            gate: [seq_len] values in (0, 1)
        """
        return torch.sigmoid(uncertainty - self.tau)

    def apply_masym(
        self,
        attention: torch.Tensor,
        gate: torch.Tensor,
        distance_matrix: np.ndarray,
    ) -> torch.Tensor:
        """
        Applies asymmetric attention mask weighted by gate and spatial distance.

        Args:
            attention:        [heads, seq_len, seq_len]
            gate:             [seq_len] gating values per token
            distance_matrix:  [num_text_tokens, 16] distance from each token to each patch

        Returns:
            modified attention of the same shape as input
        """
        heads, seq, _ = attention.shape
        dist_tensor = torch.tensor(distance_matrix, dtype=attention.dtype, device=attention.device)
        num_tokens = dist_tensor.shape[0]

        # proximity weight: 1 - distance, clipped to [0, 1]
        proximity = (1.0 - dist_tensor).clamp(0.0, 1.0)  # [num_tokens, 16]

        modified = attention.clone()
        for t in range(min(num_tokens, seq)):
            g = gate[t]  # scalar
            # Scale attention toward proximity-weighted distribution
            patch_cols = min(16, seq)
            mask = self.gamma * g * proximity[t, :patch_cols]  # [patch_cols]
            modified[:, t, :patch_cols] = modified[:, t, :patch_cols] + mask.unsqueeze(0)

        # Re-normalize across the key dimension
        modified = modified / (modified.sum(dim=-1, keepdim=True) + 1e-9)
        return modified

    def register(self, layer: nn.Module, dummy_logits: torch.Tensor, distance_matrix: np.ndarray) -> None:
        """
        Hooks multi_modal_projector.linear_2 output — the correct intervention point.

        Visual patch tokens (576 tokens, 4096-dim each) exit linear_2 before being
        concatenated with text tokens. We scale each patch token by its CLIP proximity
        to the question, weighted by uncertainty. This change is irreversible by LayerNorm
        because it's multiplicative across the full token, not a hidden-dim bias.

        Args:
            layer:            model.multi_modal_projector
            dummy_logits:     [T, vocab_size] — proxy for question token uncertainty
            distance_matrix:  [T, 16] CLIP distances from question tokens to 4x4 patches
        """
        uncertainty = self.compute_uncertainty(dummy_logits)
        gate = self.compute_gate(uncertainty)  # [T]

        dist_tensor = torch.tensor(distance_matrix, dtype=torch.float32)
        proximity = (1.0 - dist_tensor).clamp(0.0, 1.0)  # [T, 16]

        # Aggregate uncertainty across question tokens: [16] patch relevance scores
        # Each patch gets a score = mean over tokens of (gate * proximity)
        patch_weights = (gate.unsqueeze(1) * proximity).mean(dim=0)  # [16]
        # Normalize to [1-gamma, 1+gamma] range so patches stay meaningful
        w_min, w_max = patch_weights.min(), patch_weights.max()
        if w_max > w_min:
            patch_weights = (patch_weights - w_min) / (w_max - w_min)  # [0, 1]
        patch_scale = 1.0 + self.gamma * patch_weights  # [16], values in [1.0, 1+gamma]

        device = next(layer.parameters()).device
        patch_scale = patch_scale.to(device=device, dtype=torch.float16)

        print(f"UGAA visual hook | gate_mean={gate.mean():.4f} | "
              f"patch_scale min={patch_scale.min():.4f} max={patch_scale.max():.4f}")

        def _hook(module, input, output):
            # output: [batch, num_visual_tokens, 4096]
            # LLaVA-1.5 uses 336x336 / 14px patches = 24x24 = 576 visual tokens
            # We have 16 patch weights (4x4 grid) — each covers 576/16 = 36 tokens
            if isinstance(output, tuple):
                feat = output[0]
            else:
                feat = output

            batch, num_tokens, hidden = feat.shape
            tokens_per_patch = max(1, num_tokens // 16)

            scale_expanded = patch_scale.repeat_interleave(tokens_per_patch)  # [num_tokens]
            # Handle rounding: trim or pad to exact num_tokens
            if scale_expanded.shape[0] > num_tokens:
                scale_expanded = scale_expanded[:num_tokens]
            elif scale_expanded.shape[0] < num_tokens:
                pad = torch.ones(num_tokens - scale_expanded.shape[0],
                                 device=device, dtype=torch.float16)
                scale_expanded = torch.cat([scale_expanded, pad])

            scaled = feat * scale_expanded.unsqueeze(0).unsqueeze(-1)  # [B, T, H]

            if isinstance(output, tuple):
                return (scaled,) + output[1:]
            return scaled

        self._hook_handle = layer.linear_2.register_forward_hook(_hook)

    def remove(self) -> None:
        """Removes the registered hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
