# Token-Set Choice Confounds POPE

> A systematic audit of yes/no token extraction in VLM hallucination
> evaluation. Companion code, predictions, and figures for the paper
> *Token-Set Choice Confounds POPE: A Systematic Audit of Yes/No
> Extraction in VLM Hallucination Evaluation* (preprint, 2026).

[![PyPI version](https://img.shields.io/pypi/v/pope-audit.svg)](https://pypi.org/project/pope-audit/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/pope-audit.svg)](https://pypi.org/project/pope-audit/)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-pope--audit--records-ffce1c.svg)](https://huggingface.co/datasets/kesav2k04/pope-audit-records)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Reproducible](https://img.shields.io/badge/9%2C000%20questions-deterministic-brightgreen.svg)](experiments/)

## Install

The audit utilities ship as a lightweight package on PyPI:

```bash
pip install pope-audit
```

```python
from pope_audit import YES_TOKEN_IDS, NO_TOKEN_IDS, compute_f1

# The eight-token rule for LLaMA-family VLMs on POPE (see below):
YES_TOKEN_IDS  # [3582, 8241, 4874, 3869]  -> yes, Yes, ' yes', ' Yes'
NO_TOKEN_IDS   # [1217, 3782,  694, 1939]  -> no,  No,  ' no',  ' No'
```

`pope-audit` itself depends only on `transformers`, `sentencepiece`, and
`Pillow`. The GPU stack (`torch`, `bitsandbytes`, ...) is intentionally
**not** a hard dependency, because the right torch wheel is
platform/CUDA-specific; install it from
[pytorch.org](https://pytorch.org/get-started/locally/) first if you want
to run the model-loading paths. For the full research stack used to
reproduce the paper, install the extra and clone this repo (see
[Reproducing the headline numbers](#reproducing-the-headline-numbers)):

```bash
pip install "pope-audit[research]"
```

The 9,000 per-question prediction records, diagnostics, and POPE question
manifests that back every number in the paper are published as a
companion dataset on the Hugging Face Hub:
[🤗 `kesav2k04/pope-audit-records`](https://huggingface.co/datasets/kesav2k04/pope-audit-records).

```python
from datasets import load_dataset

pope = load_dataset("kesav2k04/pope-audit-records", "pope_questions", split="adversarial")
```

## TL;DR

POPE's reference evaluator reads the model's first generated token by
comparing two specific vocabulary IDs: `3582` for `yes` and `1217` for
`no`. In 9,000 greedy-decode runs of LLaVA-1.5-7B across the three
POPE splits, those two IDs appeared **zero times**. The model produces
`' Yes'` (token 3869) and `' No'` (token 1939) instead, because
SentencePiece prepends a whitespace byte after the prompt suffix
`ASSISTANT:`. Reading the wrong IDs shifts the LLaVA-1.5-7B adversarial
F1 by 6.13 points (0.7608 vs. 0.8221), which is larger than the
headline gain claimed by several recent inference-time methods.

This repository contains the evaluation pipeline, the 9,000 prediction
records, the diagnostic script, and the LaTeX / HTML versions of the
paper.

## Headline numbers (all from `experiments/`)

| Split | Two-token F1 | Eight-token F1 | Gap | Yes-rate (8-tok) | Recall (8-tok) |
|---|---:|---:|---:|---:|---:|
| Adversarial | 0.7608 | **0.8221** | +0.0613 | 0.452 | 0.7827 |
| Popular     | 0.7961 | **0.8498** | +0.0537 | 0.434 | 0.7940 |
| Random      | 0.8397 | **0.8713** | +0.0316 | 0.411 | 0.7940 |

The corrected eight-token baseline of 0.8221 on the adversarial split
is stable: in the full 3,000-question run it converges to 0.8219 by
question 1,000 and stays within 0.820 to 0.825 through question 3,000.

## The eight-token rule

For LLaMA-family VLMs evaluated on POPE under the prompt template
`USER: <image>\n{question}\nASSISTANT:`, we recommend:

```python
YES_TOKEN_IDS = [3582, 8241, 4874, 3869]   # yes, Yes, ' yes', ' Yes'
NO_TOKEN_IDS  = [1217, 3782,  694, 1939]   # no,  No,  ' no',  ' No'

# Decision rule (matches greedy decoding by construction)
yes_score = max(logits[i] for i in YES_TOKEN_IDS)
no_score  = max(logits[i] for i in NO_TOKEN_IDS)
prediction = "yes" if yes_score > no_score else "no"
```

Token-set choice is context-dependent. For any new model-template
combination we recommend running the diagnostic in
`scripts/run_pope_2tok_baseline.py` on a 100-question sample first; if the
argmax token of any question is not in `YES_TOKEN_IDS` or
`NO_TOKEN_IDS`, add it before trusting the rule.

## Repository layout

```
.
|-- README.md                  this file
|-- CITATION.cff               machine-readable citation metadata
|-- LICENSE                    MIT
|-- pyproject.toml             installable package metadata (pip install -e .)
|-- MANIFEST.in                sdist contents control
|-- requirements.txt           pinned Python dependencies
|-- environment.yml            conda environment
|-- .gitignore                 excludes model weights, caches, build, secrets
|
|-- src/pope_audit/            the installable library (pip package `pope-audit`)
|   |-- __init__.py            public API (lazy torch import)
|   |-- evaluate.py            compute_f1 / compute_accuracy / load_pope
|   |-- pope_loader.py         load POPE splits
|   |-- ugaa_hook.py           UGAA v5 + _get_yes_no_logits + token-ID constants
|   `-- clip_l_grounding.py    CLIP-L per-patch similarity module
|
|-- scripts/                   runnable drivers + data downloaders
|   |-- run_pope_2tok_baseline.py   diagnostic + 2-tok vs 8-tok comparison
|   |-- run_pope_eval_full.py       full 3,000q POPE eval driver
|   |-- run_ablation_a.py           nine inference-time correction methods
|   |-- run_full_diagnostic_3000q.py  3,000q per-question diagnostic
|   |-- run_cross_model_token_audit.py  second-model token audit
|   |-- run_multi_model_audit.py    four-readout audit across VLMs
|   `-- download_pope_full.py       fetch POPE questions + images
|
|-- analysis/                  paper-claim audit, stats, and figure scripts
|   |-- fact_audit.py
|   |-- diagnostic_stats.py
|   |-- string_parse_equivalence.py
|   `-- latency_microbench.py
|
|-- experiments/               JSON outputs from every run
|   |-- pope_adversarial_2tok_vs_8tok.json    main result, adv. split
|   |-- pope_popular_2tok_vs_8tok.json
|   |-- pope_random_2tok_vs_8tok.json
|   |-- ugaa_full_adversarial_3000q_diagnostic.json   3,000q diagnostic (sec. 5.3)
|   |-- cross_model/           LLaVA-1.6-Mistral per-question logs
|   `-- multi_model/           recorded metrics for the four extra VLMs
|
|-- datasets/                  POPE questions and image paths (images gitignored)
|-- docs/                      local + cloud setup notes
|-- references/                other-paper bibtex + survey CSV
|-- paper/                     paper-side audit docs (LaTeX sources gitignored)
|
`-- archive/                   initial UGAA research, not needed for reproduction
    |-- src_legacy/            older / experimental scripts (preserved verbatim)
    |-- experiments_legacy/    superseded prediction logs
    |-- notebooks/             cloud runners (run_on_cloud.ipynb reproduces Table 6)
    `-- misc/                  one-off utilities
```

## Reproducing the headline numbers

The full pipeline runs on an 8 GB GPU in roughly 80 minutes for all
9,000 questions.

### 0. Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

# (optional) install the audit utilities as an editable package so the
# scripts can `import pope_audit` from any working directory:
pip install -e .
```

Or, with conda:

```bash
conda env create -f environment.yml
conda activate ugaa
```

Notes on the two development machines we used are in
[docs/local_setup_notes.md](docs/local_setup_notes.md).

### 1. Download LLaVA-1.5-7B weights

We used the HuggingFace checkpoint `llava-hf/llava-1.5-7b-hf`,
quantized to 4-bit NF4 through BitsAndBytes. The default
`--model-path llava-hf/llava-1.5-7b-hf` flag downloads it to the
HuggingFace cache on first use. Use `--cache-dir <dir>` to control
where it lives.

### 2. Reproduce the protocol comparison (Table 2 in the paper)

```bash
python scripts/run_pope_2tok_baseline.py --split adversarial \
    --model-path llava-hf/llava-1.5-7b-hf \
    --data-dir datasets/pope --output-dir experiments

python scripts/run_pope_2tok_baseline.py --split popular  --output-dir experiments
python scripts/run_pope_2tok_baseline.py --split random   --output-dir experiments
```

Expected output files and metrics (eight-token rule, full 3,000-question splits):

| File | F1 | Precision | Recall | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `experiments/pope_adversarial_2tok_vs_8tok.json` | 0.8221 | 0.8658 | 0.7827 | 1174 | 1318 | 182 | 326 |
| `experiments/pope_popular_2tok_vs_8tok.json`     | 0.8498 | 0.9140 | 0.7940 | 1191 | 1388 | 112 | 309 |
| `experiments/pope_random_2tok_vs_8tok.json`      | 0.8713 | 0.9652 | 0.7940 | 1191 | 1457 |  43 | 309 |

### 3. Reproduce the nine inference-time corrections (Table 4)

```bash
python scripts/run_ablation_a.py --variant all    --beta 1.0 \
    --model-path llava-hf/llava-1.5-7b-hf \
    --dataset datasets/pope/pope_sample_100.json \
    --output-dir experiments

# Full 3,000-question runs for the headline correction methods:
python scripts/run_pope_eval_full.py --split adversarial --baseline \
    --output-dir experiments
python scripts/run_pope_eval_full.py --split adversarial \
    --variant clip_certainty --beta 1.0 --output-dir experiments
python scripts/run_pope_eval_full.py --split adversarial \
    --variant clip_certainty --beta 1.5 --output-dir experiments
```

Expected per-method summary files in `experiments/`:

| File | F1 | Precision | Recall |
|---|---:|---:|---:|
| `pope_full_adversarial_baseline_summary.json`   | 0.8221 | 0.8658 | 0.7827 |
| `pope_full_adversarial_beta1.0_summary.json`    | 0.8164 | 0.9023 | 0.7453 |
| `pope_full_adversarial_clip_b1.0_summary.json`  | 0.8171 | 0.9072 | 0.7433 |
| `pope_full_adversarial_clip_b1.5_summary.json`  | 0.8104 | 0.9278 | 0.7193 |

### 4. Multi-model validation (four additional VLMs)

To test whether the token-set confound is specific to LLaVA-1.5 or a
property of the readout protocol itself, run the same four-readout audit
(`legacy_2tok`, `legacy_8tok`, `dynamic_single`, `string_parse`) on
other VLMs with `scripts/run_multi_model_audit.py`. Each model is run on all
three POPE splits at the full 3,000 questions per split.

```bash
# LLaVA-1.6-Mistral (local, 4-bit) -- one split shown; repeat for popular, random
python scripts/run_multi_model_audit.py --model llava16_mistral \
    --split adversarial --samples 3000 \
    --data-dir datasets/pope --output-dir experiments/cross_model \
    --device cuda --quantize 4bit

# InstructBLIP / mPLUG-Owl2 / Qwen2-VL run the same way on a T4 (Kaggle/Colab);
# see archive/notebooks/run_on_cloud.ipynb for the turnkey cloud runner.
python scripts/run_multi_model_audit.py --model instructblip --split adversarial --samples 3000 --output-dir experiments/multi_model --device cuda
python scripts/run_multi_model_audit.py --model mplug_owl2   --split adversarial --samples 3000 --output-dir experiments/multi_model --device cuda
python scripts/run_multi_model_audit.py --model qwen2_vl     --split adversarial --samples 3000 --output-dir experiments/multi_model --device cuda --prompt-mode native_chat_template
```

Measured POPE F1 (3,000 questions per split; LLaVA-1.6 local RTX 3070 Ti,
the other three on Kaggle T4). Bold marks the F1 a careful practitioner
would report for each model:

| Model (tokenizer) | Split | legacy_2tok | legacy_8tok | dynamic_single | string_parse |
|---|---|---:|---:|---:|---:|
| LLaVA-1.5-7B (LLaMA-2) | adversarial | 0.7608 | **0.8221** | 0.8221 | 0.8221 |
| LLaVA-1.5-7B (LLaMA-2) | popular | 0.7961 | **0.8498** | 0.8498 | 0.8498 |
| LLaVA-1.5-7B (LLaMA-2) | random | 0.8397 | **0.8713** | 0.8713 | 0.8713 |
| LLaVA-1.6-Mistral-7B (Mistral) | adversarial | 0.6667 | 0.0000 | **0.8521** | 0.8521 |
| LLaVA-1.6-Mistral-7B (Mistral) | popular | 0.6667 | 0.0000 | **0.8953** | 0.8953 |
| LLaVA-1.6-Mistral-7B (Mistral) | random | 0.6667 | 0.0000 | **0.9163** | 0.9163 |
| InstructBLIP-Vicuna-7B (LLaMA) | adversarial | 0.7074 | 0.8183 | **0.8183** | 0.6667 |
| InstructBLIP-Vicuna-7B (LLaMA) | popular | 0.7277 | 0.8553 | **0.8553** | 0.6667 |
| InstructBLIP-Vicuna-7B (LLaMA) | random | 0.7613 | 0.8815 | **0.8815** | 0.6667 |
| mPLUG-Owl2-LLaMA2-7B (LLaMA-2) | adversarial | 0.7754 | 0.8001 | **0.8001** | 0.8001 |
| mPLUG-Owl2-LLaMA2-7B (LLaMA-2) | popular | 0.8060 | 0.8312 | **0.8312** | 0.8312 |
| mPLUG-Owl2-LLaMA2-7B (LLaMA-2) | random | 0.8466 | 0.8707 | **0.8707** | 0.8707 |
| Qwen2-VL-7B-Instruct (Qwen) | adversarial | 0.6674 | 0.0000 | **0.8457** | 0.8457 |
| Qwen2-VL-7B-Instruct (Qwen) | popular | 0.6671 | 0.0000 | **0.8578** | 0.8578 |
| Qwen2-VL-7B-Instruct (Qwen) | random | 0.6667 | 0.0000 | **0.8671** | 0.8671 |

Two patterns hold across every model and split. The legacy LLaVA-1.5
eight-token readout collapses to F1 0.00 on the models with disjoint
vocabularies (LLaVA-1.6-Mistral, Qwen2-VL), while the tokenizer-derived
`dynamic_single` readout holds between 0.80 and 0.92 everywhere.
`string_parse` matches `dynamic_single` on every model except
InstructBLIP, where the free-text parse did not isolate the answer token
and collapsed to an all-yes prediction (0.6667); the single-token logit
readout is the robust default. Corresponding artifacts:

- `experiments/cross_model/llava16_mistral_pope_{adversarial,popular,random}_3000q_paper_template_token_audit.json` (full per-question records)
- `experiments/multi_model/multi_model_results.json` (recorded metrics for the four additional models; the three cloud runs are reproducible with `archive/notebooks/run_on_cloud.ipynb`)
- the original 500-question two-prompt-template comparison remains in `experiments/cross_model/llava-hf_llava-v1.6-mistral-7b-hf_pope_adversarial_500q_*_token_audit.json`

For a CPU-only sanity check that just verifies the dynamic
single-token IDs derived from the second model's tokenizer:

```bash
python scripts/run_cross_model_token_audit.py \
    --model-path llava-hf/llava-v1.6-mistral-7b-hf \
    --dry-run-tokenizer --cache-dir ./hf_cache
```

See [experiments/cross_model/README.md](experiments/cross_model/README.md)
for the full description of every JSON field and what to look at.

### 5. Verify the paper's numeric claims and submission gate

```bash
python analysis/fact_audit.py        # phase 1+2 paper-claim audit, phase 3 strict gate
python analysis/diagnostic_stats.py  # Mann-Whitney U on the 100q diagnostic
```

`fact_audit.py` exits with code 0 if every claim matches the JSON
ground truth and the paper sources contain no placeholders, anonymous-
author text (in the arXiv version), TODO/FIXME, em dashes, corrupted
characters, or off-by-one Table 7 entries.

## Determinism

All runs use:
- `torch.manual_seed(42)`
- `transformers==4.40.1`
- `bitsandbytes` 4-bit NF4 quantization, fp16 compute dtype
- `model.generate(max_new_tokens=1, output_scores=True)` with greedy
  decoding

Rerunning the same script on the same hardware produces identical
predictions. Different GPUs may differ in the third decimal of
individual logits but not in the argmax token, so the eight-token F1
is stable across hardware.

## Hardware footprint

| Hardware | Throughput | Total time for 9,000 questions |
|---|---|---|
| NVIDIA RTX 3070 Ti, 8 GB VRAM | 0.508 s/q | 76.3 min |
| NVIDIA RTX 3050, 4 GB VRAM | (verification only) | partial |

### Cloud Hardware
- Kaggle Tesla T4x2 GPU (mPLUG-Owl2, InstructBLIP, Qwen2-VL)
- Approximately 27,000 POPE evaluations across three models and three splits

### Cloud Accounts
Experiments were distributed across multiple Kaggle sessions to accommodate GPU quota limits.

## Citing this work

If you find the audit useful, please cite via `CITATION.cff` or the
BibTeX entry below.

```bibtex
@misc{xxxx2026tokenset,
  title  = {Token-Set Choice Confounds {POPE}:
            A Systematic Audit of Yes/No Extraction
            in {VLM} Hallucination Evaluation},
  author = {Jayakumar, Kesav Kumar and Thilak, Karthigeyan},
  year   = {2026},
  note   = {Preprint, arXiv identifier to be added on submission.}
}
```

## License

MIT, see [LICENSE](LICENSE). POPE images are used under the COCO Terms
of Use. LLaVA-1.5 weights are used under their respective license
(Apache 2.0). CLIP weights are used under MIT.

## Acknowledgments

We used Claude (Anthropic) as a coding assistant for experimental
script development and figure rendering. All experimental design,
analysis, and writing are our own. This work received no external
funding and was conducted independently during the authors'
undergraduate studies.
  
