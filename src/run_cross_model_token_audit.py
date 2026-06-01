"""
run_cross_model_token_audit.py
==============================

Token-readout audit on a second LLaMA-family VLM. The main paper's
eight-token set is LLaVA-1.5 / LLaMA-2 specific. For any new model
(LLaVA-1.6 with Mistral or Vicuna backbone, InstructBLIP, etc.) the
correct read-out token IDs must be derived from the model's own
tokenizer; this script does that derivation at startup and reports
four parallel readouts plus a string-parse evaluation.

Readouts reported per question:

  legacy_llava15_2tok        compares logits at IDs 3582 vs 1217
                             (LLaVA-1.5 paper baseline; NOT valid
                             outside LLaMA-2-vocabulary models).
  legacy_llava15_8tok        max-logit over the LLaVA-1.5 eight-token
                             set (3582, 8241, 4874, 3869) vs
                             (1217, 3782, 694, 1939). Same caveat.
  dynamic_single_token       max-logit over the *single-token* IDs
                             obtained by encoding "yes", "Yes", " yes",
                             " Yes" (and the corresponding no-forms)
                             with the model's own tokenizer. Forms
                             that tokenize to more than one token are
                             excluded (a first-token logit cannot
                             represent a multi-token surface form)
                             and recorded under multitoken_forms in
                             the summary.
  string_parse               runs greedy generation for max_new_tokens
                             (default 6), takes the decoded text, and
                             parses the first "yes"/"no" substring
                             (case-insensitive). This bypasses the
                             vocabulary-ID question entirely and is
                             the protocol the paper recommends.

Prompt modes (`--prompt-mode`):

  paper_template             USER: <image>\\n{question} Answer yes or no only.\\nASSISTANT:
                             The LLaVA-1.5 template the paper uses.
  native_chat_template       processor.apply_chat_template(...) if the
                             processor provides one. Used when the
                             second model expects an INST-style or
                             other model-specific template.

Pass `--prompt-mode both` to run the audit twice (once per template).
The two artifacts are written to separate JSONs so the comparison is
explicit.

Output:

    experiments/cross_model/<slug>_pope_<split>_<n>q_<prompt_mode>_token_audit.json

Run:

    python src/run_cross_model_token_audit.py \\
        --model-path llava-hf/llava-v1.6-mistral-7b-hf \\
        --data-dir datasets/pope --output-dir experiments/cross_model \\
        --samples 500 --quantize 4bit --device cuda \\
        --cache-dir D:/models/hf_cache \\
        --prompt-mode both
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


# -----------------------------------------------------------------------------
# Legacy LLaVA-1.5 / LLaMA-2 read-out IDs (used only as comparison; NOT a
# generic read-out for non-LLaMA-2 models).
# -----------------------------------------------------------------------------
LEGACY_LLAVA15_YES_2TOK = [3582]
LEGACY_LLAVA15_NO_2TOK = [1217]
LEGACY_LLAVA15_YES_8TOK = [3582, 8241, 4874, 3869]
LEGACY_LLAVA15_NO_8TOK = [1217, 3782, 694, 1939]

# Surface forms whose single-token id we attempt to discover dynamically
# from the model's own tokenizer.
YES_FORMS = ["yes", "Yes", " yes", " Yes"]
NO_FORMS = ["no", "No", " no", " No"]


def _slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


# -----------------------------------------------------------------------------
# Tokenizer-derived dynamic readout
# -----------------------------------------------------------------------------

def derive_dynamic_token_sets(tokenizer):
    """Encode each yes/no surface form and split into single-token vs
    multi-token forms.

    For each form, we call tokenizer.encode(form, add_special_tokens=False).
    A single-token form contributes its id to dynamic_yes_ids or
    dynamic_no_ids. A multi-token form is recorded under
    multitoken_forms so the operator knows it cannot be used in a
    first-token logit comparison.
    """
    info = {
        "dynamic_yes_ids": [],
        "dynamic_no_ids": [],
        "yes_form_to_ids": {},
        "no_form_to_ids": {},
        "multitoken_forms": [],
    }
    for form in YES_FORMS:
        try:
            ids = tokenizer.encode(form, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(form)
        ids = list(int(i) for i in ids)
        info["yes_form_to_ids"][form] = ids
        if len(ids) == 1:
            info["dynamic_yes_ids"].append(ids[0])
        else:
            info["multitoken_forms"].append(
                {"polarity": "yes", "form": form, "ids": ids}
            )
    for form in NO_FORMS:
        try:
            ids = tokenizer.encode(form, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(form)
        ids = list(int(i) for i in ids)
        info["no_form_to_ids"][form] = ids
        if len(ids) == 1:
            info["dynamic_no_ids"].append(ids[0])
        else:
            info["multitoken_forms"].append(
                {"polarity": "no", "form": form, "ids": ids}
            )
    # Deduplicate while preserving order.
    info["dynamic_yes_ids"] = sorted(set(info["dynamic_yes_ids"]))
    info["dynamic_no_ids"] = sorted(set(info["dynamic_no_ids"]))
    return info


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def _load_model(model_path: str, quantize: str, device: str, cache_dir):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, quantization_config=bnb, device_map="auto", **kwargs,
        )
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, quantization_config=bnb, device_map="auto", **kwargs,
        )
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=dtype, **kwargs,
        )
        if device == "cuda":
            model = model.to("cuda")

    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    return model, processor


# -----------------------------------------------------------------------------
# Prompt rendering
# -----------------------------------------------------------------------------

PAPER_TEMPLATE = "USER: <image>\n{question} Answer yes or no only.\nASSISTANT:"


def build_prompt(processor, question: str, mode: str) -> str:
    """Render a prompt for one of the two supported templates.

    paper_template          identical to the paper run.
    native_chat_template    processor.apply_chat_template(...) when
                            available; falls back to paper_template
                            if the processor does not provide one.
    """
    text = f"{question} Answer yes or no only."
    if mode == "paper_template":
        return PAPER_TEMPLATE.format(question=question)
    # native_chat_template
    tok = getattr(processor, "tokenizer", None)
    try:
        if tok is not None and getattr(tok, "chat_template", None):
            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            return processor.apply_chat_template(conv, add_generation_prompt=True)
    except Exception:
        pass
    return PAPER_TEMPLATE.format(question=question)


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

_YES_RE = re.compile(r"\byes\b", re.I)
_NO_RE = re.compile(r"\bno\b", re.I)


def _string_parse(decoded: str) -> str:
    """Parse the model's decoded response into yes / no / unknown.

    We look for the first yes/no word boundary in the decoded text.
    Word boundaries avoid matching "now" as "no" or "yes-no-maybe" as
    something other than "yes".
    """
    if not decoded:
        return "unknown"
    head = decoded.strip().lower()[:64]
    m_yes = _YES_RE.search(head)
    m_no = _NO_RE.search(head)
    if m_yes and not m_no:
        return "yes"
    if m_no and not m_yes:
        return "no"
    if m_yes and m_no:
        return "yes" if m_yes.start() < m_no.start() else "no"
    return "unknown"


def _predict_one(model, processor, image, question, prompt_mode, dynamic,
                 device, parse_max_new_tokens):
    import torch
    prompt = build_prompt(processor, question, prompt_mode)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}

    # Single forward pass with output_scores=True, then continue
    # generating up to parse_max_new_tokens for the string-parse readout.
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=parse_max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )

    # First-token logits
    logits = out.scores[0][0].float().cpu()
    top = int(logits.argmax())

    def _max_or_inf_neg(ids):
        if not ids:
            return float("-inf")
        return max(float(logits[i]) for i in ids)

    # Legacy LLaVA-1.5 readouts (informational only for non-LLaMA-2 vocab)
    pred_legacy_2 = (
        "yes" if logits[LEGACY_LLAVA15_YES_2TOK[0]]
                  > logits[LEGACY_LLAVA15_NO_2TOK[0]] else "no"
    )
    yes_l8 = _max_or_inf_neg(LEGACY_LLAVA15_YES_8TOK)
    no_l8 = _max_or_inf_neg(LEGACY_LLAVA15_NO_8TOK)
    pred_legacy_8 = "yes" if yes_l8 > no_l8 else "no"

    # Dynamic single-token readout from this model's own tokenizer
    yes_d = _max_or_inf_neg(dynamic["dynamic_yes_ids"])
    no_d = _max_or_inf_neg(dynamic["dynamic_no_ids"])
    if dynamic["dynamic_yes_ids"] and dynamic["dynamic_no_ids"]:
        pred_dynamic = "yes" if yes_d > no_d else "no"
    else:
        pred_dynamic = "unknown"

    # String-parse readout (greedy generation, parse decoded text)
    seq = out.sequences[0]
    # Slice off the prompt prefix when it is present in the sequence.
    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    gen_ids = seq[input_len:]
    try:
        decoded = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    except Exception:
        decoded = ""
    pred_string = _string_parse(decoded)

    return {
        "top_token_id": top,
        "yes_logit_legacy_8tok": yes_l8 if yes_l8 != float("-inf") else None,
        "no_logit_legacy_8tok": no_l8 if no_l8 != float("-inf") else None,
        "yes_logit_dynamic": yes_d if yes_d != float("-inf") else None,
        "no_logit_dynamic": no_d if no_d != float("-inf") else None,
        "pred_legacy_2tok": pred_legacy_2,
        "pred_legacy_8tok": pred_legacy_8,
        "pred_dynamic_single": pred_dynamic,
        "pred_string_parse": pred_string,
        "decoded_response": decoded,
    }


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def _confusion(records, key):
    tp = sum(1 for r in records if r.get(key) == "yes" and r["label"] == "yes")
    tn = sum(1 for r in records if r.get(key) == "no" and r["label"] == "no")
    fp = sum(1 for r in records if r.get(key) == "yes" and r["label"] == "no")
    fn = sum(1 for r in records if r.get(key) == "no" and r["label"] == "yes")
    unk = sum(1 for r in records if r.get(key) not in ("yes", "no"))
    return tp, tn, fp, fn, unk


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return f1, p, r


def _eval_block(records, key, n_valid):
    tp, tn, fp, fn, unk = _confusion(records, key)
    f1, p, r = _f1(tp, fp, fn)
    yes = sum(1 for x in records if x.get(key) == "yes")
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "unknown": unk,
        "f1": round(f1, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "yes_predictions": yes,
        "yes_rate": round(yes / max(n_valid, 1), 4),
    }


# -----------------------------------------------------------------------------
# Run one prompt mode end-to-end
# -----------------------------------------------------------------------------

def run_one_mode(args, model, processor, dynamic, items, prompt_mode, device):
    import torch
    import transformers
    from PIL import Image

    n = len(items)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"{_slug(args.model_path)}_pope_{args.split}_{n}q_"
        f"{prompt_mode}_token_audit.json"
    )

    try:
        tokenizer_class = type(processor.tokenizer).__name__
    except Exception:
        tokenizer_class = type(processor).__name__

    device_name = (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu")

    print(f"\n=== prompt_mode={prompt_mode} | n={n} | model={args.model_path} ===")
    print(f"  prompt sample: {build_prompt(processor, 'Is there a cat in the image?', prompt_mode)!r}")
    print(f"  dynamic_yes_ids={dynamic['dynamic_yes_ids']}  "
          f"dynamic_no_ids={dynamic['dynamic_no_ids']}")
    if dynamic["multitoken_forms"]:
        for mt in dynamic["multitoken_forms"]:
            print(f"  multitoken: polarity={mt['polarity']} form={mt['form']!r} ids={mt['ids']}")

    t0 = time.time()
    records = []
    top_counts = {}
    errors = 0

    for i, item in enumerate(items):
        try:
            image = Image.open(item["local_path"]).convert("RGB")
            preds = _predict_one(
                model, processor, image, item["question"], prompt_mode,
                dynamic, device, args.parse_max_new_tokens,
            )
            try:
                top_str = processor.tokenizer.decode([preds["top_token_id"]])
            except Exception:
                top_str = ""
            top_counts[preds["top_token_id"]] = (
                top_counts.get(preds["top_token_id"], 0) + 1
            )
            records.append({
                "question_id": item.get("question_id", i + 1),
                "label": item["label"],
                "top_token_id": int(preds["top_token_id"]),
                "top_token_str": top_str,
                "pred_legacy_2tok": preds["pred_legacy_2tok"],
                "pred_legacy_8tok": preds["pred_legacy_8tok"],
                "pred_dynamic_single": preds["pred_dynamic_single"],
                "pred_string_parse": preds["pred_string_parse"],
                "decoded_response": preds["decoded_response"],
            })
        except Exception as exc:
            errors += 1
            records.append({
                "question_id": item.get("question_id", i + 1),
                "label": item.get("label"),
                "error": str(exc),
            })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] elapsed={time.time()-t0:.1f}s errors={errors}")

    elapsed = time.time() - t0
    valid = [r for r in records if "pred_string_parse" in r]
    n_valid = len(valid)

    sorted_top = sorted(top_counts.items(), key=lambda kv: -kv[1])[:12]
    decoded_top = []
    for tok_id, cnt in sorted_top:
        try:
            s = processor.tokenizer.decode([tok_id])
        except Exception:
            s = ""
        decoded_top.append({"token_id": int(tok_id), "count": int(cnt),
                            "decoded": s})

    summary = {
        "model_path": args.model_path,
        "prompt_mode": prompt_mode,
        "paper_template_string": PAPER_TEMPLATE,
        "tokenizer_class": tokenizer_class,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device_name": device_name,
        "split": args.split,
        "n_total": len(records),
        "n_valid": n_valid,
        "n_errors": errors,
        "quantize": args.quantize,
        "parse_max_new_tokens": args.parse_max_new_tokens,
        # Token sets
        "legacy_llava15_yes_2tok": LEGACY_LLAVA15_YES_2TOK,
        "legacy_llava15_no_2tok":  LEGACY_LLAVA15_NO_2TOK,
        "legacy_llava15_yes_8tok": LEGACY_LLAVA15_YES_8TOK,
        "legacy_llava15_no_8tok":  LEGACY_LLAVA15_NO_8TOK,
        "dynamic_yes_ids":         dynamic["dynamic_yes_ids"],
        "dynamic_no_ids":          dynamic["dynamic_no_ids"],
        "yes_form_to_ids":         dynamic["yes_form_to_ids"],
        "no_form_to_ids":          dynamic["no_form_to_ids"],
        "multitoken_forms":        dynamic["multitoken_forms"],
        # Readouts
        "eval_legacy_2tok":        _eval_block(valid, "pred_legacy_2tok", n_valid),
        "eval_legacy_8tok":        _eval_block(valid, "pred_legacy_8tok", n_valid),
        "eval_dynamic_single":     _eval_block(valid, "pred_dynamic_single", n_valid),
        "eval_string_parse":       _eval_block(valid, "pred_string_parse", n_valid),
        "top_token_counts":        decoded_top,
        "runtime_seconds":         round(elapsed, 2),
        "sec_per_question":        round(elapsed / max(n_valid, 1), 3),
        "note":                    (
            "Legacy LLaVA-1.5 readouts are reported only for comparison; "
            "they are valid only when the model uses the LLaMA-2 vocabulary "
            "and the LLaVA-1.5 ASSISTANT: prompt suffix. Dynamic single-token "
            "and string-parse readouts are the ones to trust for this model."
        ),
    }

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": records}, f, indent=2)

    print(f"\n  legacy_2tok:    F1={summary['eval_legacy_2tok']['f1']}  "
          f"P={summary['eval_legacy_2tok']['precision']}  "
          f"R={summary['eval_legacy_2tok']['recall']}  "
          f"yr={summary['eval_legacy_2tok']['yes_rate']}")
    print(f"  legacy_8tok:    F1={summary['eval_legacy_8tok']['f1']}  "
          f"P={summary['eval_legacy_8tok']['precision']}  "
          f"R={summary['eval_legacy_8tok']['recall']}  "
          f"yr={summary['eval_legacy_8tok']['yes_rate']}")
    print(f"  dynamic_single: F1={summary['eval_dynamic_single']['f1']}  "
          f"P={summary['eval_dynamic_single']['precision']}  "
          f"R={summary['eval_dynamic_single']['recall']}  "
          f"yr={summary['eval_dynamic_single']['yes_rate']}  "
          f"unk={summary['eval_dynamic_single']['unknown']}")
    print(f"  string_parse:   F1={summary['eval_string_parse']['f1']}  "
          f"P={summary['eval_string_parse']['precision']}  "
          f"R={summary['eval_string_parse']['recall']}  "
          f"yr={summary['eval_string_parse']['yes_rate']}  "
          f"unk={summary['eval_string_parse']['unknown']}")
    print(f"  saved to {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", required=True,
                        help="HF repo id or local path of the second model.")
    parser.add_argument("--data-dir", default="datasets/pope",
                        help="Directory containing pope_{split}_full.json.")
    parser.add_argument("--output-dir", default="experiments/cross_model",
                        help="Where to write the audit JSON(s).")
    parser.add_argument("--split", default="adversarial",
                        choices=["adversarial", "popular", "random"])
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of questions to run.")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--quantize", default="4bit",
                        choices=["4bit", "8bit", "none"])
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace cache directory.")
    parser.add_argument(
        "--prompt-mode", default="paper_template",
        choices=["paper_template", "native_chat_template", "both"],
        help="Which prompt template(s) to evaluate.",
    )
    parser.add_argument(
        "--parse-max-new-tokens", type=int, default=6,
        help="Max new tokens to generate for the string-parse readout.",
    )
    parser.add_argument(
        "--dry-run-tokenizer", action="store_true",
        help="Load only the tokenizer (via AutoTokenizer) and report the "
             "dynamic single-token sets. Skips model load and inference.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir) / f"pope_{args.split}_full.json"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. "
              "Run scripts/download_pope_full.py first.")
        sys.exit(2)

    with open(data_path) as f:
        items = json.load(f)
    items = items[: args.samples]

    if args.dry_run_tokenizer:
        # Useful for the QA-of-the-QA: confirm dynamic IDs without
        # touching the GPU.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            args.model_path,
            **({"cache_dir": args.cache_dir} if args.cache_dir else {}),
        )
        info = derive_dynamic_token_sets(tok)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{_slug(args.model_path)}_tokenizer_dryrun.json"
        info["tokenizer_class"] = type(tok).__name__
        info["model_path"] = args.model_path
        info["samples_requested"] = args.samples
        info["status"] = "tokenizer_only_dry_run"
        with open(out_path, "w") as f:
            json.dump(info, f, indent=2)
        print(json.dumps(info, indent=2))
        print(f"\nDry-run artifact written to {out_path}")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failure_path = out_dir / f"{_slug(args.model_path)}_model_load_failed.json"

    try:
        model, processor = _load_model(
            args.model_path, args.quantize, args.device, args.cache_dir,
        )
    except Exception as exc:
        import traceback
        record = {
            "model_path": args.model_path,
            "status": "model_load_failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "split": args.split,
            "samples_requested": len(items),
        }
        with open(failure_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Model load failed; wrote diagnostic to {failure_path}")
        sys.exit(3)

    try:
        dynamic = derive_dynamic_token_sets(processor.tokenizer)
    except Exception as exc:
        print(f"WARNING: dynamic tokenizer derivation failed ({exc}); "
              "dynamic readout will be marked unknown for every question.")
        dynamic = {
            "dynamic_yes_ids": [],
            "dynamic_no_ids": [],
            "yes_form_to_ids": {},
            "no_form_to_ids": {},
            "multitoken_forms": [],
        }

    modes = (["paper_template", "native_chat_template"]
             if args.prompt_mode == "both" else [args.prompt_mode])
    for m in modes:
        run_one_mode(args, model, processor, dynamic, items, m, args.device)


if __name__ == "__main__":
    main()
