# Verification Report — "Token-Set Choice Confounds POPE" revision

## JSON files read and confirmed (ground truth, never modified)

| File | Use | Result |
| --- | --- | --- |
| `experiments/ugaa_full_adversarial_3000q_diagnostic.json` | Baseline confusion, attention, logit gap, set-max logits | TP 1173 / TN 1317 / FP 183 / FN 327; attention correct 0.0738 (n=2490), wrong 0.0801 (n=510); logit-gap means 2.066 / 1.482 / 0.700 / 0.962; yes_raw 24.63, no_raw 24.53 |
| `experiments/pope_adversarial_2tok_vs_8tok.json` | Two-token and eight-token matrices, Eq. derivation | 8tok TP 1174 / FP 182 / FN 326 / TN 1318; 2tok TP 1438 / FP 842 / FN 62 / TN 658; F1 0.8221 / 0.7608; top tokens 1939→1644, 3869→1356 |
| `experiments/pope_popular_2tok_vs_8tok.json` | Table 2 / Table 11 popular split | 2tok F1 0.7961, 8tok F1 0.8498; 8tok TP 1191 / FP 112 / FN 309 / TN 1388 |
| `experiments/pope_random_2tok_vs_8tok.json` | Table 2 / Table 11 random split | 2tok F1 0.8397, 8tok F1 0.8713; 8tok TP 1191 / FP 43 / FN 309 / TN 1457 |
| `experiments/multi_model/multi_model_results.json` | Table 6 (four models × three splits × four readouts) | Every cell matches the paper exactly |
| `analysis/diagnostic_stats.py` | Bootstrap parameters for Section 5.3 | n_boot=10000, `np.random.default_rng(42)`, percentile method (lines 112–125) |
| Regenerated cloud artifacts (D:\RESEARCH DOCS 2026\…) | Secondary verification run provenance | See Regenerated cloud-artifact audit below |

---

## Regenerated cloud-artifact audit (secondary verification run)

Reruns on Kaggle Tesla T4 with newer transformers versions
(mPLUG 4.34.1, InstructBLIP 4.57.1, Qwen 5.10.1):

- **Precision recovered:** mPLUG-Owl2 `quantize=none` (full precision);
  InstructBLIP `quantize=4bit` (run log and summary); Qwen2-VL `quantize=4bit`.
- **Reproduction outcomes:**
  - Qwen2-VL: all splits reproduced exactly.
  - mPLUG-Owl2: adversarial and popular splits reproduced exactly.
  - InstructBLIP (adversarial): differs by approximately 0.004 F1
    (0.7107 / 0.8222 vs. paper values 0.7074 / 0.8183).
  - mPLUG-Owl2 (random): regenerated file was a degenerate all-yes output
    due to a storage error; original console metric retained as the only
    valid source for that cell.
- **Decision:** Report the original-run metrics in Table 6 and describe
  the rerun as a secondary verification run. No reported number was changed.

---

## Numbers verified vs. assumed

**Verified (recomputed directly from JSON artifacts):**
all confusion-matrix cells (adversarial, popular, random; both protocols);
all Table 2, Table 6, and Table 11 F1 values;
φ = 264/326 = 0.810; ρ_N = 660/1318 = 0.501;
aggregate flip rate 924/1644 = 0.5621;
F1_8 = 2348/2856 = 0.8221; F1_2 = 2876/3780 = 0.7608;
φ = 0 lower bound = 2348/3516 = 0.6678;
attention and logit-gap means, standard deviations, pooled SDs,
and Cohen's d values (−0.45/−0.46 and 1.06);
set-max logit means 24.63 and 24.53; bootstrap parameters.

**Assumed / operator-supplied:**
secondary verification run wording (provided verbatim by operator);
cloud-model original-run transformers versions taken from
`multi_model_results.json` provenance notes.

---

## Issues found that were not in the original fix list

1. **InstructBLIP adversarial reproducibility gap (~0.004 F1)** between the
   paper's original run and the regenerated artifact. Surfaced to operator;
   resolved by reporting original metrics with a verification-run note.

2. **mPLUG-Owl2 random regenerated file is degenerate (all-yes output).**
   Flagged; original console metric retained as the only valid source
   for that cell.

3. **Two-rate ρ arithmetic inconsistency.** The single-rate parametrization
   gave FP_2 = 182 + ρ·G_tot = 1106, which is internally inconsistent with
   Figure 5 (FP_2 = 842). Root cause traced: 924 total flips split into
   660 FP-flips and 264 TP-flips. Resolved with the operator-approved
   two-rate closed form (φ, ρ_N).

4. **Liang citation misattribution** in the "Why ρ ≈ 0.56" paragraph.
   HELM benchmark-sensitivity citation was incorrectly applied to a
   LLaMA-specific calibration claim. Removed during the M1 rewrite.

5. **`run_metadata.json` transformers version conflict.** The file at
   D:\RESEARCH DOCS 2026\ reports transformers 5.10.1 (Qwen environment),
   distinct from per-folder metadata. Reconciled in the secondary
   verification run wording.

---

## Clarifications requested and operator decisions

| Question | Decision used |
| --- | --- |
| Multi-model data source (per-question files absent) | Use `multi_model_results.json` summary as ground truth |
| Eq. 2 ρ parametrization | Two-rate closed form (φ, ρ_N) |
| L1 logit data availability | Repo-wide search first → only set-max logits stored → soften L1 claim accordingly |
| HTML version handling | Edit `index.html` directly |
| C3 prompt-format claim | Cite the existing 500-question two-template measurement |
| §9.3 framing | Strict no-infer framing: verifiable-readout count = 0 |
| S4 citation | Citation-free pre-specification wording |
| Secondary verification run precision | Use regenerated artifacts as provenance source |
| Table 6 vs. regenerated metrics | Report original metrics with verification-run note (operator wording) |

---

## Phase 8.5 validation tasks (V1–V4)

- **V1 (official POPE codebase):** Not present in the repo (only the
  authors' own `run_pope_eval*.py` scripts). External code not downloaded
  per instructions. **Not performed.**

- **V2 (quantization sensitivity, LLaVA-1.5 FP16 vs. NF4):** No FP16/BF16
  LLaVA-1.5 run exists in `experiments/`. **Not performed.** The regenerated
  mPLUG-Owl2 full-precision run is a different model and is reported under
  the secondary verification run.

- **V3 (full-layer attention):** The diagnostic stores only the aggregated
  layers-14–20 grounding scalar; no per-layer attention data is available.
  The S4 justification wording was retained; the sentence "full-layer gives
  the same result" was **not added** because it would be unsupported.

- **V4 (prompt-template sensitivity):** The 500-question two-template
  comparison exists (Table 7 / Table 8 in the paper) and was integrated
  into C3 as a measured value (~1 F1 point, cross-model, indicative only).

---

## Items not applied (with reason)

| Item | Reason not applied |
| --- | --- |
| S3 SD table in HTML | The HTML is a condensed web version without statistical appendices; omission introduces no contradiction. |
| M3/M4 in HTML | The HTML references "the Theorem in Section 6" without a proof or corollary block; no target text exists to modify. |
| T2/T3 in HTML | The HTML has no survey-table placeholder and no §9.3 denominator block. |

---

## Page count (compiled with MiKTeX; throwaway builds, not delivered)

Both versions compile cleanly: no LaTeX errors, no undefined references,
no undefined citations.

**arXiv version:**
HEAD baseline = 27 pp; edited = **28 pp**.
The required additions (two-rate derivation, Appendix D SD table,
secondary verification run paragraph, three additional survey rows)
outweigh the Phase 7 prose cuts.
Decision: accept approximately 26–28 pp (arXiv has no hard page limit);
no content was degraded to recover pages.

**NeurIPS version:**
HEAD baseline = body to p9 (Conclusion on p9, within the 9-page limit);
edited = body to **p10** (Conclusion on p10), approximately one page over
the 9-page limit.
Total file = 27 pp (References and NeurIPS checklist at pp11–15;
first appendix at p16 — these do not count toward the limit).
Primary cause: the secondary verification run paragraph in §8
(approximately 8 lines), the single most movable block.
Decision: compile and report only; no cuts were made.
To return to 9 pp, moving that paragraph to the appendix with a
one-line pointer in the body, or trimming the C3/L1 sentences,
would recover the page.

> **Note:** The committed `paper/arxiv/main.pdf`, `paper/neurips/main.pdf`,
> and associated `*.aux` / `*.bbl` files are stale relative to the edited
> `.tex` sources. Recompile before submission.
