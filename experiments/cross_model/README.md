# Cross-model token audit

Per-question token-readout artifacts produced by
`scripts/run_cross_model_token_audit.py` against a second LLaMA-family VLM.

## Why the LLaVA-1.5 token IDs are not a generic readout

The eight-token set in the main paper (`[3582, 8241, 4874, 3869]` for
yes, `[1217, 3782, 694, 1939]` for no) is **LLaVA-1.5 specific**. It
was derived from the LLaMA-2 SentencePiece tokenizer under the
`USER: <image>\n{question}\nASSISTANT:` prompt suffix. Two facts make
it unsafe to transfer to other models:

1. **Different vocabulary.** A second model with a different
   tokenizer assigns different integer IDs to the same surface forms.
   On `llava-hf/llava-v1.6-mistral-7b-hf`, "Yes" is token id 5592 and
   "yes" is 5081 (Mistral-style vocabulary). The LLaVA-1.5 IDs do not
   point at those tokens at all.
2. **Different first-token surface forms.** The space-prefixed
   variants " yes" and " Yes" tokenize to two tokens under the
   Mistral vocabulary, so they cannot be represented by a single
   first-token logit. A first-token comparison must therefore use
   only the single-token forms the tokenizer actually produces.

For these reasons, the cross-model audit script derives a
`dynamic_single_token` readout per model from the model's own
tokenizer, keeps the legacy LLaVA-1.5 readouts only as informational
columns, and additionally reports a `string_parse` readout that
bypasses the vocabulary-id question entirely.

## How to run

```bash
python scripts/run_cross_model_token_audit.py \
    --model-path llava-hf/llava-v1.6-mistral-7b-hf \
    --data-dir datasets/pope \
    --output-dir experiments/cross_model \
    --samples 500 --quantize 4bit --device cuda \
    --cache-dir ./hf_cache \
    --prompt-mode both
```

If a 500-question run hits an 8 GB VRAM ceiling, rerun with
`--samples 100`. The script saves a `*_model_load_failed.json`
diagnostic if loading fails before inference.

To verify the dynamic token sets without touching the GPU:

```bash
python scripts/run_cross_model_token_audit.py \
    --model-path llava-hf/llava-v1.6-mistral-7b-hf \
    --dry-run-tokenizer --cache-dir ./hf_cache
```

This writes
`experiments/cross_model/<slug>_tokenizer_dryrun.json` with
`dynamic_yes_ids`, `dynamic_no_ids`, and any `multitoken_forms`.

## Output JSON fields

For each `--prompt-mode` the script writes
`<slug>_pope_<split>_<n>q_<prompt_mode>_token_audit.json`. The
`summary` contains:

- `model_path`, `prompt_mode`, `paper_template_string`,
  `tokenizer_class`, `transformers_version`, `torch_version`,
  `device_name`, `quantize`, `split`, `n_total`, `n_valid`,
  `n_errors`, `parse_max_new_tokens`.
- Legacy LLaVA-1.5 IDs (`legacy_llava15_yes_2tok` etc.) for reference.
- Tokenizer-derived `dynamic_yes_ids`, `dynamic_no_ids`,
  `yes_form_to_ids`, `no_form_to_ids`, `multitoken_forms`.
- Four `eval_*` blocks with `tp/tn/fp/fn/unknown`, F1, precision,
  recall, yes-rate:
  - `eval_legacy_2tok`, `eval_legacy_8tok` (informational only when
    the model is not LLaVA-1.5).
  - `eval_dynamic_single` (the readout to trust for first-token
    logit comparison on this model).
  - `eval_string_parse` (greedy generation, parsed for `yes`/`no`).
- `top_token_counts`: decoded top-12 first tokens across all questions.

Each `results` entry contains: `question_id`, `label`,
`top_token_id`, `top_token_str`, `pred_legacy_2tok`,
`pred_legacy_8tok`, `pred_dynamic_single`, `pred_string_parse`,
`decoded_response`.

## What to look for

- Are `eval_dynamic_single` and `eval_string_parse` close to each
  other? If so, the dynamic single-token readout is a good proxy.
- Is `eval_legacy_8tok` very different from `eval_string_parse`?
  That is the cross-model analogue of the paper's main finding: a
  readout that does not match the model's actual generated tokens
  produces a misleading aggregate.
- Does any `multitoken_forms` entry indicate a tokenizer that splits
  a yes/no surface form into multiple tokens? If yes, that form
  cannot enter the first-token logit comparison and the dynamic set
  is correspondingly smaller.

## Measured results (LLaVA-1.6-Mistral-7B, POPE adversarial, first 500q)

These are the values produced by the rewritten script on
`llava-hf/llava-v1.6-mistral-7b-hf` under 4-bit NF4 on an
NVIDIA RTX 3070 Ti Laptop GPU (`transformers v4.44.2`,
`torch 2.5.1+cu121`). They are the same values verified by
`analysis/fact_audit.py` Phase 4.

```
dynamic_yes_ids: [5081, 5592]    # yes, Yes
dynamic_no_ids:  [708,  1770]    # no,  No
multitoken_forms: ' yes' -> [28705, 5081]   ' Yes' -> [28705, 5592]
                  ' no'  -> [28705, 708]    ' No'  -> [28705, 1770]
top tokens generated:  5592 ("Yes"), 1770 ("No"). All 500 first
                       tokens belong to {5592, 1770} in both
                       prompt-mode runs.
```

| Prompt template | Readout | F1 | Precision | Recall | Yes-rate | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paper_template       | legacy_2tok    | 0.6667 | 0.5000 | 1.0000 | 1.000 | 250 |   0 | 250 |   0 |
| paper_template       | legacy_8tok    | 0.0000 | 0.0000 | 0.0000 | 0.000 |   0 | 250 |   0 | 250 |
| paper_template       | dynamic_single | 0.8719 | 0.8352 | 0.9120 | 0.546 | 228 | 205 |  45 |  22 |
| paper_template       | string_parse   | 0.8719 | 0.8352 | 0.9120 | 0.546 | 228 | 205 |  45 |  22 |
| native_chat_template | legacy_2tok    | 0.6925 | 0.5297 | 1.0000 | 0.944 | 250 |  28 | 222 |   0 |
| native_chat_template | legacy_8tok    | 0.0000 | 0.0000 | 0.0000 | 0.000 |   0 | 250 |   0 | 250 |
| native_chat_template | dynamic_single | 0.8621 | 0.9517 | 0.7880 | 0.414 | 197 | 240 |  10 |  53 |
| native_chat_template | string_parse   | 0.8621 | 0.9517 | 0.7880 | 0.414 | 197 | 240 |  10 |  53 |

Notes:

- Under both prompt templates, `dynamic_single` and `string_parse`
  agree to four decimals. That is the expected behaviour under greedy
  decoding when no multi-token surface form ever reaches first position.
- `legacy_8tok` collapses to F1 = 0 because none of the LLaMA-2 IDs
  the main paper recommends are ever the argmax token; every comparison
  resolves the same way and every prediction is labelled `no`.
- `legacy_2tok` shows a near-all-yes bias because it is comparing two
  never-generated IDs and the comparison floor lands on `yes` for
  almost every question.

## Stale artifact preserved as evidence

`llava-hf_llava-v1.6-mistral-7b-hf_pope_adversarial_1q_DIAGNOSTIC_ONLY_legacy_ids_invalid.json`
is a 1-question artifact from before the dynamic-readout fix. It is
kept on disk as evidence of the QA finding that motivated the rewrite:
the legacy LLaVA-1.5 eight-token check incorrectly labelled a
LLaVA-1.6 generation of token id 5592 ("Yes") as "no" because the
LLaMA-2 ID set does not cover the Mistral vocabulary. Do not cite
this file as a result.
