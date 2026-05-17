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

    def register(self, layer: nn.Module) -> None:
        """Registers a forward hook on the given module."""
        def _hook(module, input, output):
            # Expects output to be (attn_output, attn_weights) or just attn_output
            if isinstance(output, tuple) and len(output) >= 2:
                return output  # hook site; user should call apply_masym separately
            return output

        self._hook_handle = layer.register_forward_hook(_hook)

    def remove(self) -> None:
        """Removes the registered hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
