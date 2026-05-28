## Sweep Results Summary

| β | Accuracy | TP | TN | FP | FN | Δ |
|---|----------|----|----|----|----|---|
| 0.0 (baseline) | 0.6648 | 18 | 31 | 8 | 30 | 0.0 |
| 0.3 | 0.6243 | 16 | 30 | 8 | 31 | -0.040 |
| 0.5 | 0.6141 | 14 | 30 | 8 | 32 | -0.050 |
| 0.8 | 0.6338 | 12 | 38 | 8 | 30 | -0.031 |
| 1.0 | 0.6135 | 10 | 41 | 8 | 33 | 0.00 |
| 1.5 | 0.5325 | 8  | 50 | 8 | 34 | -0.13 |

**What This Tells Us**

- The core problem is clear: every beta value trades true positives (TPs) for true negatives (TNs) at a bad rate. **β=0.8** is the "least bad" — it gains 7 TNs (+7) but loses 10 TPs (‑10). Net loss every time.
- The false positives (31 wrong "yes" answers) have confidence gaps of **0.8–1.4**, which is larger than the TPs (gaps of **0.4–0.6**). So a uniform NO‑bias kills TPs before FPs.

**What To Do Next**

- The v5 certainty‑modulated approach isn’t working because the certainty range is too narrow (**0.52–0.61**) — not enough spread to differentiate.
- Future work should explore wider‑range certainty signals or adaptive bias schedules instead of a fixed uniform bias.
