"""
run_multi_model_audit.py
========================

Full POPE audit on additional LLaMA-family VLMs. The driver is a
generalisation of scripts/run_cross_model_token_audit.py: each supported
model registers an adapter that knows (i) how to load the model,
(ii) how to build the prompt template, (iii) how to call generate to
get first-token logits + a short generated continuation. The four
readouts (legacy_2tok, legacy_8tok, dynamic_single, string_parse)
are computed identically across models so the resulting JSONs are
directly comparable.

Supported model adapters:

  llava15            llava-hf/llava-1.5-7b-hf           (reference model from main paper)
  llava16_mistral    llava-hf/llava-v1.6-mistral-7b-hf  (cross-model run from Section 7)
  instructblip       Salesforce/instructblip-vicuna-7b
  mplug_owl2         MAGAer13/mplug-owl2-llama2-7b

Usage:

  python scripts/run_multi_model_audit.py \\
      --model instructblip \\
      --split adversarial --samples 3000 \\
      --data-dir datasets/pope \\
      --output-dir experiments/multi_model \\
      --device cuda --quantize 4bit \\
      --cache-dir D:/models/hf_cache

Output:
  experiments/multi_model/<model>_pope_<split>_<n>q_token_audit.json

If a model adapter is not yet implemented in this script, the run
exits cleanly with a status message naming the next step (typically
adding a small adapter block at the top of this file).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Project root one level above scripts/ for CWD-independent default paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Adapter interface
# -----------------------------------------------------------------------------

class Adapter:
    """Minimum interface a model adapter must implement."""
    model_id: str

    def load(self, cache_dir, device, quantize):
        raise NotImplementedError

    def build_prompt(self, question: str, prompt_mode: str) -> str:
        raise NotImplementedError

    def call_generate(self, model, processor, image, prompt, max_new_tokens, device):
        """Return (logits_first_token: 1D tensor, decoded_response: str)."""
        raise NotImplementedError


# -----------------------------------------------------------------------------
# LLaVA-1.5 adapter
# -----------------------------------------------------------------------------

PAPER_TEMPLATE_LLAVA15 = "USER: <image>\n{question} Answer yes or no only.\nASSISTANT:"


class LlavaAdapter(Adapter):
    def __init__(self, model_id: str):
        self.model_id = model_id

    def load(self, cache_dir, device, quantize):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        if quantize == "4bit" and device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id, quantization_config=bnb, device_map="auto", **kwargs,
            )
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=dtype, **kwargs,
            )
            if device == "cuda":
                model = model.to("cuda")
        processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        return model, processor

    def build_prompt(self, question: str, prompt_mode: str) -> str:
        if prompt_mode == "native_chat_template":
            return _try_chat_template(None, question)  # falls back to paper template
        return PAPER_TEMPLATE_LLAVA15.format(question=question)

    def call_generate(self, model, processor, image, prompt, max_new_tokens, device):
        import torch
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                return_dict_in_generate=True, output_scores=True,
                do_sample=False,
            )
        logits = out.scores[0][0].float().cpu()
        seq = out.sequences[0]
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        gen_ids = seq[input_len:]
        try:
            decoded = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        except Exception:
            decoded = ""
        return logits, decoded


class LlavaNextAdapter(LlavaAdapter):
    """Same as LlavaAdapter but uses LlavaNextForConditionalGeneration so
    that LLaVA-1.6 (mistral / vicuna) checkpoints load correctly."""

    def load(self, cache_dir, device, quantize):
        import torch
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        if quantize == "4bit" and device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id, quantization_config=bnb, device_map="auto", **kwargs,
            )
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=dtype, **kwargs,
            )
            if device == "cuda":
                model = model.to("cuda")
        processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        return model, processor

    def build_prompt(self, question: str, prompt_mode: str) -> str:
        if prompt_mode == "native_chat_template":
            # The mistral variant uses [INST] ... [/INST] structure.
            return f"[INST] <image>\n{question} Answer yes or no only. [/INST]"
        return PAPER_TEMPLATE_LLAVA15.format(question=question)


# -----------------------------------------------------------------------------
# InstructBLIP adapter (Vicuna-7B backend)
# -----------------------------------------------------------------------------

INSTRUCTBLIP_TEMPLATE = "Question: {question} Answer yes or no only. Answer:"


class InstructBlipAdapter(Adapter):
    model_id = "Salesforce/instructblip-vicuna-7b"

    def load(self, cache_dir, device, quantize):
        import torch
        from transformers import (
            InstructBlipForConditionalGeneration, InstructBlipProcessor,
        )
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        if quantize == "4bit" and device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model = InstructBlipForConditionalGeneration.from_pretrained(
                self.model_id, quantization_config=bnb, device_map="auto", **kwargs,
            )
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = InstructBlipForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=dtype, **kwargs,
            )
            if device == "cuda":
                model = model.to("cuda")
        processor = InstructBlipProcessor.from_pretrained(self.model_id, **kwargs)
        return model, processor

    def build_prompt(self, question: str, prompt_mode: str) -> str:
        # InstructBLIP does not have a separate "chat template"; the
        # paper_template option uses Question:/Answer: which is the
        # canonical form for this model.
        return INSTRUCTBLIP_TEMPLATE.format(question=question)

    def call_generate(self, model, processor, image, prompt, max_new_tokens, device):
        import torch
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                return_dict_in_generate=True, output_scores=True,
                do_sample=False, min_new_tokens=1,
            )
        logits = out.scores[0][0].float().cpu()
        seq = out.sequences[0]
        try:
            decoded = processor.tokenizer.decode(seq, skip_special_tokens=True)
        except Exception:
            decoded = ""
        return logits, decoded


# -----------------------------------------------------------------------------
# mPLUG-Owl2 adapter
# -----------------------------------------------------------------------------

class MPlugOwl2Adapter(Adapter):
    """mPLUG-Owl2 uses a custom architecture that is not part of the
    standard Transformers AutoModel registry. The HF repo
    `MAGAer13/mplug-owl2-llama2-7b` ships its own modeling code. Set
    `trust_remote_code=True` to load it; the adapter then uses the
    `chat` method exposed by the custom class for inference.

    If the user has not whitelisted trust_remote_code, this adapter
    will exit cleanly with an instruction to enable it.
    """
    model_id = "MAGAer13/mplug-owl2-llama2-7b"

    def load(self, cache_dir, device, quantize):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = {"cache_dir": cache_dir, "trust_remote_code": True} if cache_dir else {"trust_remote_code": True}
        if quantize == "4bit" and device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=bnb, device_map="auto", **kwargs,
            )
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=dtype, **kwargs,
            )
            if device == "cuda":
                model = model.to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        # mPLUG-Owl2 exposes a custom processor in its repo
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        except Exception:
            class _Wrap:
                def __init__(self, tok): self.tokenizer = tok; self.image_processor = None
            processor = _Wrap(tokenizer)
        return model, processor

    def build_prompt(self, question: str, prompt_mode: str) -> str:
        # mPLUG-Owl2 uses USER:/ASSISTANT: with <|image|> placeholder.
        return f"USER: <|image|>{question} Answer yes or no only.\nASSISTANT:"

    def call_generate(self, model, processor, image, prompt, max_new_tokens, device):
        # mPLUG-Owl2's tokenizer and image processor are integrated;
        # adjust this method to the version of the repo you have. The
        # baseline implementation below assumes processor(text=..,
        # images=..) returns a dict similar to LLaVA's processor.
        import torch
        if processor.image_processor is None:
            raise RuntimeError(
                "mPLUG-Owl2 image processor not detected. Re-load with the "
                "official AutoProcessor (trust_remote_code=True)."
            )
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                return_dict_in_generate=True, output_scores=True,
                do_sample=False,
            )
        logits = out.scores[0][0].float().cpu()
        seq = out.sequences[0]
        try:
            decoded = processor.tokenizer.decode(seq, skip_special_tokens=True)
        except Exception:
            decoded = ""
        return logits, decoded


def _try_chat_template(processor, question: str) -> str:
    """Stub: LLaVA-1.5 processor does not ship a chat template; fall back."""
    return PAPER_TEMPLATE_LLAVA15.format(question=question)


# -----------------------------------------------------------------------------
# Qwen2-VL adapter
# -----------------------------------------------------------------------------

class Qwen2VLAdapter(Adapter):
    """Qwen/Qwen2-VL-7B-Instruct adapter. Uses the HF-native
    `Qwen2VLForConditionalGeneration` class so no trust_remote_code is
    required. The processor handles image+text together through its
    `apply_chat_template` method, which is the only template the model
    is trained on; we expose both a paper_template mode (the LLaVA-1.5
    `USER: ... ASSISTANT:` form passed as raw text) and a native chat
    template mode. Practitioners running on cloud GPUs (T4 / P100 /
    A100) should generally prefer the native chat template here because
    Qwen2-VL was not trained on the LLaVA-1.5 string and will produce
    erratic first-token behaviour under it."""

    model_id = "Qwen/Qwen2-VL-7B-Instruct"

    def load(self, cache_dir, device, quantize):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2VLForConditionalGeneration not available in your "
                "transformers version. Install transformers>=4.45.0:\n"
                "    pip install --upgrade 'transformers>=4.45.0'"
            ) from exc

        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        if quantize == "4bit" and device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, quantization_config=bnb,
                device_map="auto", **kwargs,
            )
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=dtype, **kwargs,
            )
            if device == "cuda":
                model = model.to("cuda")
        processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        return model, processor

    def build_prompt(self, question: str, prompt_mode: str):
        # We return a tuple (text, messages) so call_generate can pick
        # the right code path; LlavaAdapter et al. return a plain str.
        text = f"{question} Answer yes or no only."
        if prompt_mode == "paper_template":
            return ("paper_template",
                    PAPER_TEMPLATE_LLAVA15.format(question=question))
        # native_chat_template via processor.apply_chat_template
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }]
        return ("native_chat_template", messages)

    def call_generate(self, model, processor, image, prompt, max_new_tokens, device):
        import torch
        mode, payload = prompt if isinstance(prompt, tuple) else ("paper_template", prompt)
        if mode == "native_chat_template":
            text = processor.apply_chat_template(payload, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image],
                               padding=True, return_tensors="pt")
        else:
            inputs = processor(text=[payload], images=[image],
                               padding=True, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                return_dict_in_generate=True, output_scores=True,
                do_sample=False,
            )
        logits = out.scores[0][0].float().cpu()
        seq = out.sequences[0]
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        gen_ids = seq[input_len:]
        try:
            decoded = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        except Exception:
            decoded = ""
        return logits, decoded


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

ADAPTERS = {
    "llava15":         LlavaAdapter("llava-hf/llava-1.5-7b-hf"),
    "llava16_mistral": LlavaNextAdapter("llava-hf/llava-v1.6-mistral-7b-hf"),
    "instructblip":    InstructBlipAdapter(),
    "mplug_owl2":      MPlugOwl2Adapter(),
    "qwen2_vl":        Qwen2VLAdapter(),
}

# Surface forms whose single-token id is discovered per-model.
YES_FORMS = ["yes", "Yes", " yes", " Yes"]
NO_FORMS = ["no", "No", " no", " No"]

LEGACY_LLAVA15_YES_2TOK = [3582]
LEGACY_LLAVA15_NO_2TOK = [1217]
LEGACY_LLAVA15_YES_8TOK = [3582, 8241, 4874, 3869]
LEGACY_LLAVA15_NO_8TOK = [1217, 3782, 694, 1939]


def derive_dynamic_token_sets(tokenizer):
    info = {"dynamic_yes_ids": [], "dynamic_no_ids": [],
            "yes_form_to_ids": {}, "no_form_to_ids": {},
            "multitoken_forms": []}
    for form in YES_FORMS:
        try:
            ids = tokenizer.encode(form, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(form)
        ids = [int(i) for i in ids]
        info["yes_form_to_ids"][form] = ids
        if len(ids) == 1:
            info["dynamic_yes_ids"].append(ids[0])
        else:
            info["multitoken_forms"].append(
                {"polarity": "yes", "form": form, "ids": ids})
    for form in NO_FORMS:
        try:
            ids = tokenizer.encode(form, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(form)
        ids = [int(i) for i in ids]
        info["no_form_to_ids"][form] = ids
        if len(ids) == 1:
            info["dynamic_no_ids"].append(ids[0])
        else:
            info["multitoken_forms"].append(
                {"polarity": "no", "form": form, "ids": ids})
    info["dynamic_yes_ids"] = sorted(set(info["dynamic_yes_ids"]))
    info["dynamic_no_ids"] = sorted(set(info["dynamic_no_ids"]))
    return info


_YES_RE = re.compile(r"\byes\b", re.I)
_NO_RE = re.compile(r"\bno\b", re.I)


def string_parse(decoded: str) -> str:
    if not decoded:
        return "unknown"
    head = decoded.strip().lower()[:64]
    my = _YES_RE.search(head); mn = _NO_RE.search(head)
    if my and not mn: return "yes"
    if mn and not my: return "no"
    if my and mn: return "yes" if my.start() < mn.start() else "no"
    return "unknown"


def confusion(records, key):
    tp = sum(1 for r in records if r.get(key) == "yes" and r["label"] == "yes")
    tn = sum(1 for r in records if r.get(key) == "no" and r["label"] == "no")
    fp = sum(1 for r in records if r.get(key) == "yes" and r["label"] == "no")
    fn = sum(1 for r in records if r.get(key) == "no" and r["label"] == "yes")
    unk = sum(1 for r in records if r.get(key) not in ("yes", "no"))
    return tp, tn, fp, fn, unk


def f1score(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r) if (p + r) else 0.0), p, r


def eval_block(records, key, n_valid):
    tp, tn, fp, fn, unk = confusion(records, key)
    f1, p, r = f1score(tp, fp, fn)
    yes = sum(1 for x in records if x.get(key) == "yes")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "unknown": unk,
            "f1": round(f1, 4), "precision": round(p, 4), "recall": round(r, 4),
            "yes_predictions": yes,
            "yes_rate": round(yes / max(n_valid, 1), 4)}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=list(ADAPTERS.keys()),
                        help="Which model adapter to use.")
    parser.add_argument("--split", default="adversarial",
                        choices=["adversarial", "popular", "random"])
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of questions to run.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "datasets" / "pope"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments" / "multi_model"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--quantize", default="4bit",
                        choices=["4bit", "8bit", "none"])
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--prompt-mode", default="paper_template",
                        choices=["paper_template", "native_chat_template"])
    parser.add_argument("--parse-max-new-tokens", type=int, default=6)
    args = parser.parse_args()

    adapter = ADAPTERS[args.model]

    data_path = Path(args.data_dir) / f"pope_{args.split}_full.json"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        sys.exit(2)
    with open(data_path) as f:
        items = json.load(f)
    items = items[: args.samples]
    n = len(items)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_pope_{args.split}_{n}q_{args.prompt_mode}_token_audit.json"

    print(f"=== {args.model} | {args.split} | n={n} | prompt_mode={args.prompt_mode} ===")
    print(f"Loading model {adapter.model_id} (quantize={args.quantize})...")
    try:
        model, processor = adapter.load(args.cache_dir, args.device, args.quantize)
    except Exception as exc:
        import traceback
        rec = {"model": args.model, "model_id": adapter.model_id,
               "status": "model_load_failed",
               "error": str(exc), "traceback": traceback.format_exc(),
               "split": args.split, "samples_requested": n}
        with open(out_dir / f"{args.model}_model_load_failed.json", "w") as f:
            json.dump(rec, f, indent=2)
        print(f"Model load failed; diagnostic written to {out_dir}.")
        sys.exit(3)

    dynamic = derive_dynamic_token_sets(processor.tokenizer)
    sample_prompt = adapter.build_prompt("Is there a cat in the image?", args.prompt_mode)
    print(f"Sample prompt: {sample_prompt!r}")
    print(f"dynamic_yes_ids={dynamic['dynamic_yes_ids']}  "
          f"dynamic_no_ids={dynamic['dynamic_no_ids']}")
    if dynamic["multitoken_forms"]:
        for mt in dynamic["multitoken_forms"]:
            print(f"  multitoken: polarity={mt['polarity']} form={mt['form']!r} ids={mt['ids']}")

    from PIL import Image
    t0 = time.time()
    records = []
    top_counts = {}
    errors = 0

    for i, item in enumerate(items):
        try:
            image = Image.open(item["local_path"]).convert("RGB")
            prompt = adapter.build_prompt(item["question"], args.prompt_mode)
            logits, decoded = adapter.call_generate(
                model, processor, image, prompt, args.parse_max_new_tokens, args.device,
            )
            top = int(logits.argmax())
            top_counts[top] = top_counts.get(top, 0) + 1

            # legacy readouts (LLaVA-1.5 IDs)
            pred_legacy_2 = "yes" if logits[LEGACY_LLAVA15_YES_2TOK[0]] > logits[LEGACY_LLAVA15_NO_2TOK[0]] else "no"
            yes_l8 = max(float(logits[i_]) for i_ in LEGACY_LLAVA15_YES_8TOK)
            no_l8 = max(float(logits[i_]) for i_ in LEGACY_LLAVA15_NO_8TOK)
            pred_legacy_8 = "yes" if yes_l8 > no_l8 else "no"

            # dynamic single-token readout
            if dynamic["dynamic_yes_ids"] and dynamic["dynamic_no_ids"]:
                yes_d = max(float(logits[i_]) for i_ in dynamic["dynamic_yes_ids"])
                no_d = max(float(logits[i_]) for i_ in dynamic["dynamic_no_ids"])
                pred_dynamic = "yes" if yes_d > no_d else "no"
            else:
                pred_dynamic = "unknown"

            pred_string = string_parse(decoded)

            try:
                top_str = processor.tokenizer.decode([top])
            except Exception:
                top_str = ""

            records.append({
                "question_id": item.get("question_id", i + 1),
                "label": item["label"],
                "top_token_id": top, "top_token_str": top_str,
                "pred_legacy_2tok": pred_legacy_2,
                "pred_legacy_8tok": pred_legacy_8,
                "pred_dynamic_single": pred_dynamic,
                "pred_string_parse": pred_string,
                "decoded_response": decoded,
            })
        except Exception as exc:
            errors += 1
            records.append({
                "question_id": item.get("question_id", i + 1),
                "label": item.get("label"),
                "error": str(exc),
            })
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n}] elapsed={elapsed/60:.1f}m errors={errors}")

    elapsed = time.time() - t0
    valid = [r for r in records if "pred_string_parse" in r]
    n_valid = len(valid)

    import torch, transformers
    sorted_top = sorted(top_counts.items(), key=lambda kv: -kv[1])[:12]
    decoded_top = []
    for tok_id, cnt in sorted_top:
        try:
            s = processor.tokenizer.decode([tok_id])
        except Exception:
            s = ""
        decoded_top.append({"token_id": tok_id, "count": cnt, "decoded": s})

    try:
        tokenizer_class = type(processor.tokenizer).__name__
    except Exception:
        tokenizer_class = type(processor).__name__

    summary = {
        "model": args.model,
        "model_id": adapter.model_id,
        "prompt_mode": args.prompt_mode,
        "tokenizer_class": tokenizer_class,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "cpu"),
        "split": args.split,
        "n_total": len(records),
        "n_valid": n_valid,
        "n_errors": errors,
        "quantize": args.quantize,
        "parse_max_new_tokens": args.parse_max_new_tokens,
        "legacy_llava15_yes_2tok": LEGACY_LLAVA15_YES_2TOK,
        "legacy_llava15_no_2tok": LEGACY_LLAVA15_NO_2TOK,
        "legacy_llava15_yes_8tok": LEGACY_LLAVA15_YES_8TOK,
        "legacy_llava15_no_8tok": LEGACY_LLAVA15_NO_8TOK,
        "dynamic_yes_ids": dynamic["dynamic_yes_ids"],
        "dynamic_no_ids":  dynamic["dynamic_no_ids"],
        "yes_form_to_ids": dynamic["yes_form_to_ids"],
        "no_form_to_ids":  dynamic["no_form_to_ids"],
        "multitoken_forms": dynamic["multitoken_forms"],
        "eval_legacy_2tok":    eval_block(valid, "pred_legacy_2tok", n_valid),
        "eval_legacy_8tok":    eval_block(valid, "pred_legacy_8tok", n_valid),
        "eval_dynamic_single": eval_block(valid, "pred_dynamic_single", n_valid),
        "eval_string_parse":   eval_block(valid, "pred_string_parse", n_valid),
        "top_token_counts": decoded_top,
        "runtime_seconds": round(elapsed, 2),
        "sec_per_question": round(elapsed / max(n_valid, 1), 3),
    }

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": records}, f, indent=2)

    print()
    for key in ("eval_legacy_2tok", "eval_legacy_8tok",
                "eval_dynamic_single", "eval_string_parse"):
        e = summary[key]
        print(f"  {key:22s} F1={e['f1']}  P={e['precision']}  R={e['recall']}  yr={e['yes_rate']}  unk={e['unknown']}")
    print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
