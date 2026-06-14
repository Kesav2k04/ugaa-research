"""
pope-audit: Token-Set Choice Audit for POPE VLM Hallucination Evaluation.

Core utilities for auditing yes/no token extraction in Visual Language Model
hallucination benchmarks, as described in:

    "TOKEN-SET CHOICE CONFOUNDS POPE: A Systematic Audit of Yes/No
     Extraction in VLM Hallucination Evaluation"
    Kesav Kumar J, Karthigeyan T (2026)

Submodules
----------
evaluate
    F1 / accuracy metrics for binary yes/no predictions.
pope_loader
    POPE benchmark dataset loader (JSONL → structured dicts).
ugaa_hook
    UGAA v5 inference hook and yes/no token-ID constants for
    LLaMA-family models.  Requires ``torch``.
clip_l_grounding
    CLIP-L/14-336 per-patch certainty computation for
    object-presence grounding.  Requires ``torch`` and ``transformers``.

Usage
-----
Lightweight (no torch needed)::

    from pope_audit.evaluate import compute_f1
    from pope_audit.pope_loader import load_pope_split

Full (requires torch)::

    from pope_audit.ugaa_hook import YES_TOKEN_IDS, NO_TOKEN_IDS
"""

__version__ = "0.1.0"

# --------------------------------------------------------------------------
# Lightweight re-exports: these modules use only stdlib (json, os).
# They are always importable regardless of whether torch is installed.
# --------------------------------------------------------------------------
from pope_audit.evaluate import compute_f1, compute_accuracy, load_pope
from pope_audit.pope_loader import load_pope_split, get_ground_truths

# --------------------------------------------------------------------------
# Heavy re-exports: these modules require torch (and transformers).
# We use lazy importing so that `import pope_audit` succeeds even in
# environments without torch (e.g. documentation builds, CI linting,
# or users who only need the evaluation functions).
# --------------------------------------------------------------------------

# LLaMA-family yes/no token ID constants (replicated here so lightweight
# code can access them without importing torch).
YES_TOKEN_IDS = [3582, 8241, 4874, 3869]
NO_TOKEN_IDS = [1217, 3782, 694, 1939]


def __getattr__(name):
    """Lazy import for torch-dependent symbols."""
    _torch_symbols = {
        "UGAAHook": ("pope_audit.ugaa_hook", "UGAAHook"),
        "SPATIAL_WORDS": ("pope_audit.ugaa_hook", "SPATIAL_WORDS"),
        "clip_l_certainty": ("pope_audit.clip_l_grounding", "clip_l_certainty"),
        "load_clip_l_components": ("pope_audit.clip_l_grounding", "load_clip_l_components"),
    }
    if name in _torch_symbols:
        module_path, attr = _torch_symbols[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'pope_audit' has no attribute {name!r}")


__all__ = [
    # Always available (stdlib-only)
    "compute_f1",
    "compute_accuracy",
    "load_pope",
    "load_pope_split",
    "get_ground_truths",
    "YES_TOKEN_IDS",
    "NO_TOKEN_IDS",
    # Lazy-loaded (require torch)
    "UGAAHook",
    "SPATIAL_WORDS",
    "clip_l_certainty",
    "load_clip_l_components",
]
