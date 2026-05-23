# Token-Set Choice Confounds POPE

> A systematic audit of yes/no token extraction in VLM hallucination
> evaluation. Companion code, predictions, and figures for the paper
> *Token-Set Choice Confounds POPE: A Systematic Audit of Yes/No
> Extraction in VLM Hallucination Evaluation* (preprint, 2026).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Reproducible](https://img.shields.io/badge/9%2C000%20questions-deterministic-brightgreen.svg)](experiments/)

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
`src/run_pope_2tok_baseline.py` on a 100-question sample first; if the
argmax token of any question is not in `YES_TOKEN_IDS` or
`NO_TOKEN_IDS`, add it before trusting the rule.

## Repository layout

```
.
|-- README.md                  this file
|-- CITATION.cff               machine-readable citation metadata
|-- LICENSE                    MIT
|-- requirements.txt           pinned Python dependencies
|-- .gitignore                 excludes model weights, caches, secrets
|
|-- src/                       paper-relevant Python modules and runners
|   |-- evaluate.py            compute_f1 implementation
|   |-- pope_loader.py         load POPE JSONs
|   |-- ugaa_hook.py           ugaa_v5 + _get_yes_no_logits helpers
|   |-- clip_l_grounding.py    CLIP-L per-patch similarity module
|   |-- run_pope_2tok_baseline.py
|   |                          diagnostic + 2-tok vs 8-tok comparison
|   |-- run_pope_eval_full.py  full 3,000q POPE eval driver
|   |-- run_ablation_a.py      nine inference-time correction methods
|   `-- archive/               older / experimental scripts (preserved verbatim)
|
|-- scripts/                   data downloaders and one-shot utilities
|
|-- experiments/               JSON outputs from every run
|   |-- pope_adversarial_2tok_vs_8tok.json    main result, adv. split
|   |-- pope_popular_2tok_vs_8tok.json
|   |-- pope_random_2tok_vs_8tok.json
|   |-- pope_full_adversarial_baseline_summary.json
|   |-- pope_full_adversarial_beta1.0_summary.json
|   |-- pope_full_adversarial_clip_b1.0_summary.json
|   |-- pope_full_adversarial_clip_b1.5_summary.json
|   `-- ugaa_v6_100q_validation.json    100q diagnostic (sec. 5.3)
|
|-- datasets/                  POPE questions and image paths (images gitignored)
|-- analysis/                  exploratory notebooks and plots
|-- notebooks/
|
|-- paper/                     paper sources
|   |-- neurips/               NeurIPS 2026 submission (anonymous, with checklist)
|   |-- arxiv/                 arXiv preprint (authors visible)
|   `-- html/                  arXiv-style HTML mirror with figures and MathJax
|
`-- references/                threat-paper bibtex + reading notes
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
```

### 1. Download LLaVA-1.5-7B weights

We used the HuggingFace checkpoint
`llava-hf/llava-1.5-7b-hf` quantized to 4-bit NF4 through BitsAndBytes.
Set the local path in each script's `MODEL_PATH` constant or override
on the command line.

### 2. Reproduce the protocol comparison

```bash
python src/run_pope_2tok_baseline.py --split adversarial
python src/run_pope_2tok_baseline.py --split popular
python src/run_pope_2tok_baseline.py --split random
```

Each call writes its own JSON to `experiments/`. The summary fields
`eval_2token` and `eval_8token` reproduce Table 2 of the paper to four
decimals.

### 3. Reproduce the nine inference-time corrections

```bash
python src/run_ablation_a.py --variant all   --beta 1.0
python src/run_ablation_a.py --variant clip_l --beta 1.0
# ... see file header for the full menu
```

Outputs go to `experiments/ablation_a_*_predictions.json`. Aggregate
metrics for Table 4 of the paper come from
`experiments/pope_full_adversarial_*_summary.json` (full 3,000-question
runs).

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

## Citing this work

If you find the audit useful, please cite via `CITATION.cff` or the
BibTeX entry below.

```bibtex
@misc{jayakumar2026tokenset,
  title  = {Token-Set Choice Confounds {POPE}:
            A Systematic Audit of Yes/No Extraction
            in {VLM} Hallucination Evaluation},
  author = {Jayakumar, Kesav Kumar and Thilak, Karthikeyan},
  year   = {2026},
  note   = {Preprint. arXiv:XXXX.XXXXX (to appear).}
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
