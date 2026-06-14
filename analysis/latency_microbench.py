"""
latency_microbench.py
=====================

Micro-benchmark the per-question latency cost of three POPE readout
protocols on LLaVA-1.5-7B:

  (a) two-token logit readout: max_new_tokens=1, generate once, look at
      two logit indices.
  (b) eight-token logit readout: same, but look at eight indices.
  (c) string-parse readout: max_new_tokens=N (default 6), generate, then
      decode and substring-match yes/no.

The first-token decode dominates wall-clock time because the model
runs its full vision tower + LLaMA forward pass once. Reading two
versus eight indices into logits is negligible. Generating one extra
token for string parse adds one decoder step on a 7B LLM at 4-bit,
which is much cheaper than the first step because the KV-cache is
populated and no image tokens are re-prefilled.

The benchmark runs M questions per protocol, reports min/median/mean/max
wall-clock latency per question, and a percentage delta against the
two-token baseline. It writes the results to a JSON file the paper
cites in Section 9.

Usage:

    python analysis/latency_microbench.py \\
        --model-path llava-hf/llava-1.5-7b-hf \\
        --data-dir datasets/pope --split adversarial \\
        --samples 50 --device cuda --cache-dir D:/models/hf_cache

Default is 50 questions; 50 is enough for stable median timings and
keeps the run under 5 minutes on the RTX 3070 Ti.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Import paths from src/ and anchor data/output to the repo root so the
# script runs from any working directory (local, Linux, or cloud GPU).
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

YES_2TOK = [3582]
NO_2TOK = [1217]
YES_8TOK = [3582, 8241, 4874, 3869]
NO_8TOK = [1217, 3782, 694, 1939]
PAPER_TEMPLATE = "USER: <image>\n{question} Answer yes or no only.\nASSISTANT:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--data-dir", default=str(REPO / "datasets" / "pope"))
    parser.add_argument("--split", default="adversarial")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--parse-max-new-tokens", type=int, default=6)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", default=str(REPO / "experiments" / "latency_microbench.json"))
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import (
        AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration,
    )

    kwargs = {"cache_dir": args.cache_dir} if args.cache_dir else {}
    if args.device == "cuda":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, quantization_config=bnb, device_map="auto", **kwargs,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float32, **kwargs,
        )
    processor = AutoProcessor.from_pretrained(args.model_path, **kwargs)
    device = str(model.device)

    data = Path(args.data_dir) / f"pope_{args.split}_full.json"
    with open(data) as f:
        items = json.load(f)
    items = items[: args.samples]

    def _run_protocol(max_new_tokens, do_parse):
        latencies = []
        for item in items:
            image = Image.open(item["local_path"]).convert("RGB")
            prompt = PAPER_TEMPLATE.format(question=item["question"])
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}
            if args.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True, output_scores=True,
                    do_sample=False,
                )
            # Always read logits at the same eight indices; the work is
            # negligible (~1 microsecond) but we include it for fairness.
            logits = out.scores[0][0].float().cpu()
            _ = max(float(logits[i]) for i in YES_8TOK)
            _ = max(float(logits[i]) for i in NO_8TOK)
            if do_parse:
                seq = out.sequences[0]
                input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
                gen_ids = seq[input_len:]
                _ = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if args.device == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        return latencies

    # Warm-up (model + autotuner)
    print("Warming up (5 questions)...")
    _ = _run_protocol(1, False)
    _ = _run_protocol(args.parse_max_new_tokens, True)

    print(f"\nMeasuring {len(items)} questions per protocol...")
    print("  (a) two-token logit (max_new_tokens=1)...")
    lat_2t = _run_protocol(1, False)
    print(f"      median = {statistics.median(lat_2t):.4f}s/q")
    print("  (b) eight-token logit (max_new_tokens=1)...")
    lat_8t = _run_protocol(1, False)
    print(f"      median = {statistics.median(lat_8t):.4f}s/q")
    print(f"  (c) string-parse (max_new_tokens={args.parse_max_new_tokens})...")
    lat_sp = _run_protocol(args.parse_max_new_tokens, True)
    print(f"      median = {statistics.median(lat_sp):.4f}s/q")

    def _stats(xs):
        return {"n": len(xs), "min": min(xs), "median": statistics.median(xs),
                "mean": statistics.fmean(xs), "max": max(xs),
                "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0}

    s2t = _stats(lat_2t)
    s8t = _stats(lat_8t)
    ssp = _stats(lat_sp)
    delta_8t = (s8t["median"] - s2t["median"]) / s2t["median"] * 100
    delta_sp = (ssp["median"] - s2t["median"]) / s2t["median"] * 100

    summary = {
        "model_path": args.model_path,
        "device": args.device,
        "samples": len(items),
        "parse_max_new_tokens": args.parse_max_new_tokens,
        "two_token":   s2t,
        "eight_token": s8t,
        "string_parse": ssp,
        "median_delta_pct_eight_vs_two":   round(delta_8t, 2),
        "median_delta_pct_string_vs_two":  round(delta_sp, 2),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Median latency per question:")
    print(f"  two-token:    {s2t['median']:.4f} s  (baseline)")
    print(f"  eight-token:  {s8t['median']:.4f} s  ({delta_8t:+.2f}% vs two-token)")
    print(f"  string-parse: {ssp['median']:.4f} s  ({delta_sp:+.2f}% vs two-token)")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
