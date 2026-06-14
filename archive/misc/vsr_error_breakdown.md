> **Archived / legacy note.** This file belongs to the earlier UGAA
> spatial-reasoning (VSR) line of work and is **not** part of the
> *Token-Set Choice Confounds POPE* paper. It is preserved here for
> provenance only.

## VSR Error Analysis – 31 False Positives

| Relation Type | Count | Implication for UGA A |
|----------------|-------|------------------------|
| **Lateral** (left/right/side) | **9** | Highest priority — UGAA must address these.
| **Depth** (front/behind/back) | **7** | Geometry but depth is hard to assess from attention.
| **Contact** (touching/connected) | **6** | Not spatial geometry – pure language prior; cannot be fixed by spatial attention gating.
| **Vertical** (above/below/top/under) | **3** | Easier to ground visually.
| **Proximity** (near/next/beside/far) | **3** | Relative distance – needs visual signal.
| **Other** (orientation, parallel, middle) | **3** | Miscellaneous errors.

### Take‑away for the paper
- **Contact relations** (6 FPs) are pure language priors; the model says *yes* because the action is plausible regardless of the image. These cannot be fixed by spatial attention gating and illustrate the limits of UGAA's scope.
- The remaining error categories highlight where UGAA can provide the most benefit: lateral and depth relations are the biggest sources of false positives, followed by vertical and proximity.
- Improving spatial attention gating should prioritize lateral cues, then depth reasoning, while vertical and proximity can be secondary targets.

### Cross‑version observations
- **v3**: No separate v3 predictions file is available, so we cannot quantify fixes from v3.
- **v4**: The table reflects the v4 run; no new errors introduced that were correct in earlier versions.
- **Correct predictions broken in v4**: Not applicable with the current data.
