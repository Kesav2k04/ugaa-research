# Change Log — "Token-Set Choice Confounds POPE" revision

Files edited: `paper/arxiv/main.tex`, `paper/neurips/main.tex`, `paper/html/index.html`.
No files under `experiments/` were modified (read-only ground truth).
Numbers below are verified against the experiment JSONs (see VERIFICATION_REPORT.md).

## Phase 1 — Mathematical fixes

### M1 — variable G redefined; two-rate confusion model
- **Notation table** (arxiv §3.5 / neurips App. arithmetic / html §3.5):
  - OLD: `G = true-no greedy generations`, single `ρ = fraction of token-1939 generations the two-token rule misreads as yes`.
  - NEW: `G = true-negative count (TN cell) = 1318`; added `G_tot = G + (P−T) = 1644` (total token-1939 emissions); replaced single `ρ` with two rates `ρ_N` (true-negative flip rate) and `φ` (false-negative lift rate).
- **Eq. (1)** unchanged in form; prose now states `G` is TN, so `FP_8 = N−G = 1500−1318 = 182` (matches Fig. 5 / Table 11).
- **Eq. (2)** OLD single-rate `TP_2=T, FP_2=(N−G)+ρG, FN_2=P−T, TN_2=(1−ρ)G`
  → NEW two-rate `TP_2=T+φ(P−T), FP_2=(N−G)+ρ_N·G, FN_2=(1−φ)(P−T), TN_2=(1−ρ_N)G`.
- **Eq. (3)** gap formula updated to the two-rate form.
- Numerics: `ρ_N = 660/1318 = 0.501`, `φ = 264/326 = 0.810`. Reproduces all four two-token cells exactly (TP 1438, FP 842, FN 62, TN 658).

### M2 — φ correction folded into closed form; "roughly 264" removed
- **Eq. (4)** OLD `F1_2 = 2348/3780 = 0.6212` (single-rate lower bound)
  → NEW `F1_2 = 2876/3780 = 0.7608` (two-rate, exact). Added the φ=0 lower bound `2348/3516 = 0.6678` explicitly.
- Removed "contributing roughly 264 additional TPs … Recomputing with the empirical TP=1,438"; the two-rate form now yields 0.7608 with no empirical patch.

### M1/M2 consistency propagation (same data, three prose spots each file)
- **Intro**: "In 924 of the 1,644 true-No questions … counts those 924 as false positives" → "Of the 1,644 `No` emissions … 924 flip to yes; 660 become FP, 264 become TP."
- **Confusion-matrix prose** (§3.4): "balloons from 182 to 1,106, with 924 true-No mislabeled" → "rises from 182 to 842 as 660 true-negative `No` emissions flip … other 264 become TP."
- **Fig. confusion caption**: "924 of the 1,644 true-No predictions migrate to the FP cell" → "660 of the 1,318 true-No predictions migrate to the FP cell."
- **Fig. convergence caption**: "56% of true-No questions get mis-labelled" → "56% of the model's `No` emissions get flipped to yes."
- **"Why ρ≈0.56" paragraph**: rewritten to attribute the 0.56 aggregate to φ=0.810 on false-negatives vs ρ_N=0.501 on true-negatives; removed the Liang citation misattribution on LLaMA calibration.

### M3 — Theorem 1 tie-breaking (all three files; html has no proof, so tex only)
- Added: "We assume ties in the first-position logit distribution occur with probability zero; under continuous logit distributions this holds almost surely, and in our 9,000-question run no tied argmax was observed."

### M4 — Corollary 1 softened (tex only; html references the theorem without a proof)
- OLD: "predictions disagree … exactly on questions for which t* ∈ (Y∪N)\(Y2∪N2)."
- NEW: "compares two noise-floor logits uncorrelated with t*, so it agrees with ŷ_str at the chance rate of approximately 0.5 rather than disagreeing deterministically."
- "the disagreement is total" → "the two-token comparison is uninformative on every question and recovers the generated answer only at the noise-floor chance rate."

## Phase 2 — Tables

### T1 — Table 6 (multi-model)
- No numeric change required: every cell already matches `multi_model_results.json` exactly (verified). InstructBLIP bold already on `dynamic_single`.

### T2 — Table 9 (survey)
- arxiv: removed the "Additional rows to be filled by survey work" placeholder; expanded to 8 rows (added Huo I, Jiang N, Jana R); caption rewritten to "verified entries as of submission … rows classified I remain indeterminate pending code inspection."
- neurips: table already had 8 rows and no placeholder (left as-is).
- html: no survey table present.

### T3 — Section 9.3 (community-impact methodology) — STRICT no-infer framing (per operator)
- Removed the two-denominator bullet list and the `Denominator 1 = 3, Numerator = 3, Denominator 2 = 8` sentence.
- NEW: states that the count of papers with a verifiable readout is **zero** (all "not stated"), so no numerator/denominator ratio is reported; gives a qualitative summary (Seo/Chen/Jiang ≤ 82.21; Tang/Jana R; Huo I). Also serves the Phase-7 cut of 9.3's repetitive opener.
- (html has no 9.3 block.)

## Phase 3 — Statistics

- **S1**: added uncorrected-p / Bonferroni disclaimer at the end of the signal analysis (all three files).
- **S2**: bootstrap CI now annotated "(10,000 resamples, percentile method, NumPy `default_rng` seed 42)" — verified against `analysis/diagnostic_stats.py:112`.
- **S3**: added Appendix "Group Means and Standard Deviations" table (arxiv App. D, neurips appendix) with n/mean/SD per group and pooled SDs (attention 0.0140 → d=−0.45/−0.46; logit gap 0.839 → d=1.06). (html: omitted — no contradiction introduced.)
- **S4**: citation-free pre-specification wording for the layers-14–20 window; the unsupportable "full 32-layer average gives the same result" claim was NOT added (only the aggregated window is stored). (all three files.)

## Phase 4 — Reproducibility

- **R1**: cloud provenance rewritten to "Kaggle Tesla T4" + a secondary-verification-run paragraph (operator-supplied wording): transformers v4.57.1/v5.10.1; InstructBLIP and Qwen2-VL 4-bit NF4, mPLUG-Owl2 full precision; original metrics retained. Verified precision from `instructblip_run.log` (quantize=4bit) and the regenerated audit summaries (mplug quantize=none, qwen quantize=4bit).
- **R2**: per-question-logs provenance updated (Limitations, all three files) to point at `experiments/cross_model/` and the archived verification-run artifacts.
- **R3**: added the exact prompt-template note (paper_template mode passes the LLaVA-1.5 string to each processor; InstructBLIP receives `Question: … Answer:`). arxiv footnote; neurips/html inline.

## Phase 5 — Claim softening

- **C1**: abstract, introduction, and conclusion — "larger than / exceeds the gain claimed/headline gain of several recent methods" → "… methods whose published baselines are consistent with the two-token protocol." (all three files.)
- **C2**: "This structural property now has full inferential support" → "The signal analysis across 3,000 questions is consistent with this structural property." (all three files.)
- **C3** (operator chose: cite existing 500q measurement): "account for one to three F1 points" → cites the 500q two-template comparison on LLaVA-1.6-Mistral (0.8719 vs 0.8621, ≈1 pt), flagged as cross-model and indicative, not a LLaVA-1.5 figure. (all three files.)
- **C4**: baseline-spread figure caption "(1–3 points)" removed. (all three files.)

## Phase 6 — Logit noise-floor evidence

- **L1** (softened per repo search — only set-max logits are stored, not per-token 3582/1217): added mean winning yes-set logit 24.63 (3869) and no-set logit 24.53 (1939) over 3,000 adversarial questions, plus the existing argmax-count = 0 for 3582/1217, as the noise-floor evidence. (all three files.)

## Phase 7 — Page reduction (arxiv)

- Limitations condensed (~3 paragraphs → 2, ≈60% length).
- §9.3 bullet framing removed (folded into T3).
- Recommendation 3 trimmed to 2 sentences; latency paragraph tightened to 3 sentences.
- (Compilation/page count is left to the operator per Phase 9.5.)
