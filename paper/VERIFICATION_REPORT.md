# Verification Report — "Token-Set Choice Confounds POPE" revision

## JSON files read and confirmed (ground truth, never modified)

| File | Use | Result |
|------|-----|--------|
| `experiments/ugaa_full_adversarial_3000q_diagnostic.json` | UGAA baseline confusion, attention, logit gap, set-max logits | TP1173/TN1317/FP183/FN327; att correct 0.0738 (n2490), wrong 0.0801 (n510); gap means 2.066/1.482/0.700/0.962; yes_raw 24.63, no_raw 24.53 |
| `experiments/pope_adversarial_2tok_vs_8tok.json` | 2-tok & 8-tok matrices, Eq. derivation | 8tok TP1174/FP182/FN326/TN1318; 2tok TP1438/FP842/FN62/TN658; F1 0.8221 / 0.7608; top tokens 1939→1644, 3869→1356 |
| `experiments/pope_popular_2tok_vs_8tok.json` | Table 2 / Table 11 popular | 2tok F1 0.7961, 8tok 0.8498; 8tok TP1191/FP112/FN309/TN1388 |
| `experiments/pope_random_2tok_vs_8tok.json` | Table 2 / Table 11 random | 2tok F1 0.8397, 8tok 0.8713; 8tok TP1191/FP43/FN309/TN1457 |
| `experiments/multi_model/multi_model_results.json` | Table 6 (4 models × 3 splits × 4 readouts) | every cell matches the paper exactly |
| `analysis/diagnostic_stats.py` | S2 bootstrap params | n_boot=10000, `np.random.default_rng(42)`, percentile method (lines 112–125) |
| Regenerated cloud artifacts (D:\RESEARCH DOCS 2026\…) | R1 precision + robustness | see below |

## Regenerated cloud-artifact audit (R1 / verification run)

Reruns on Kaggle Tesla T4 with newer transformers (mPLUG 4.34.1, InstructBLIP 4.57.1, Qwen 5.10.1):

- **Precision recovered**: mPLUG-Owl2 `quantize=none` (full precision), InstructBLIP `quantize=4bit` (run log + summary), Qwen2-VL `quantize=4bit`.
- **Reproduction**: Qwen2-VL all splits exact; mPLUG-Owl2 adversarial+popular exact; InstructBLIP adversarial differs by ≈0.004 F1 (0.7107/0.8222 vs paper 0.7074/0.8183); mPLUG-Owl2 random regeneration was a degenerate all-yes file (unusable).
- **Decision (operator)**: report the original-run metrics in Table 6; describe the rerun as a secondary verification run. No reported number changed.

## Numbers verified vs. assumed

- **Verified (recomputed from JSON)**: all confusion-matrix cells (adv/pop/rand, both protocols), all Table 2 / Table 6 / Table 11 F1s, φ=264/326=0.810, ρ_N=660/1318=0.501, aggregate 924/1644=0.5621, F1_8=2348/2856=0.8221, F1_2=2876/3780=0.7608, φ=0 lower bound 2348/3516=0.6678, attention/logit-gap means + SDs + pooled SDs + Cohen d (−0.45/−0.46, 1.06), set-max logit means 24.63/24.53, bootstrap params.
- **Assumed/operator-supplied**: the R1 secondary-verification-run wording (operator provided verbatim); cloud-model original-run transformers versions taken from `multi_model_results.json` provenance.

## Issues found that were NOT in the original fix list

1. **InstructBLIP-adv reproducibility gap (≈0.004 F1)** between the paper's original run and the regenerated artifact — surfaced to operator; resolved by reporting original metrics + verification-run footnote.
2. **mPLUG-Owl2 random regenerated file is degenerate (all-yes)** — flagged; original console metric retained as the only valid source for that cell.
3. **M1 ρ arithmetic**: the prompt's `FP_2 = 182 + ρ·G_tot = 1106` is internally inconsistent with Fig. 5 (FP_2=842); root cause traced (924 = 660 FP-flips + 264 TP-flips). Resolved with the operator-approved two-rate model.
4. **Liang citation misattribution** on the "Why ρ≈0.56" paragraph (HELM benchmark sensitivity cited for LLaMA-specific calibration) — removed during the M1 rewrite.
5. **`run_metadata.json` at D:\RESEARCH DOCS 2026\ reports transformers 5.10.1** (Qwen env), distinct from the per-folder metadata — reconciled in the R1 wording.

## Clarifications requested and operator answers

| Question | Answer used |
|----------|-------------|
| Multi-model data source (per-question files absent) | Use `multi_model_results.json` summary as ground truth |
| Eq. 2 ρ parametrization | Two-rate closed form (φ, ρ_N) |
| L1 logit data | Repo-wide search first → only set-max logits stored → softened L1 |
| HTML version | Edit `index.html` directly |
| C3 prompt-format claim | Cite the existing 500q two-template measurement |
| §9.3 framing | Keep strict no-infer framing (verifiable-readout count = 0) |
| S4 citation | Citation-free pre-specification wording |
| R1 precision | Use regenerated artifacts as provenance source |
| Table 6 vs regen | Report original metrics + verification-run note (operator wording) |

## Phase 8.5 validation tasks (V1–V4)

- **V1 (official POPE codebase)**: NOT present in the repo (only the authors' own `run_pope_eval*.py`). Per instructions, did not download external code. **Not performed.**
- **V2 (quantization sensitivity, LLaVA-1.5 FP16 vs NF4)**: no FP16/BF16 LLaVA-1.5 run exists in `experiments/`. **Not performed.** (The regenerated mPLUG-Owl2 full-precision run is a different model and is reported under R1.)
- **V3 (full-layer attention)**: the diagnostic stores only the aggregated layers-14–20 grounding scalar; no per-layer attention. Retained the S4 justification wording; the "full-layer gives same result" sentence was **not** added (would be unsupported).
- **V4 (prompt-template sensitivity)**: the 500q two-template comparison exists (Table 7) and was integrated into C3 as a measured value (≈1 F1 pt, cross-model).

## Items NOT applied (with reason)

- **S3 SD table in HTML**: omitted (the HTML is a condensed web version without statistical appendices; adding it introduces no contradiction, and its absence contradicts nothing).
- **M3/M4 in HTML**: the HTML has no theorem proof or corollary statement (it references "the Theorem in Section 6"), so there is no target text.
- **T2/T3 in HTML**: the HTML has no survey-table placeholder and no §9.3 denominator block.

## Page count (compiled with MiKTeX; throwaway builds, not delivered)

Both versions compile cleanly: no LaTeX errors, no undefined references or citations.

- **arXiv**: HEAD baseline = 27 pp; edited = **28 pp**. The required fix-additions (two-rate derivation, S3 appendix table App. D, R1 verification paragraph, +3 survey rows) outweigh the Phase-7 prose cuts. Operator decision: **accept ~26–28 pp** (arXiv has no hard limit); no further content degraded.
- **NeurIPS**: HEAD baseline = body to p9 (Conclusion on p9, within the 9-pp limit); edited = body to **p10** (Conclusion on p10), i.e. **~1 page over** the 9-pp limit. Total file 27 pp (References + NeurIPS checklist occupy ~p11–15; first appendix at p16 — these do not count toward the limit). Cause: the body additions, dominated by the **R1 secondary-verification-run paragraph in §8 (~8 lines)**, the single most movable block. Operator decision: **compile and report only** — no cuts made. To return to 9 pp, moving that paragraph to the appendix (leaving a one-line pointer), or trimming the C3/L1 sentences, would recover the page.

Note: the committed `paper/arxiv/main.pdf`, `paper/neurips/main.pdf`, and the `*.aux/*.bbl` files are now stale relative to the edited `.tex`; recompile before submission.
