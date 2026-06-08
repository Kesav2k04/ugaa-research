# Paper Documentation

This directory contains the audit documentation associated with:

**Token-Set Choice Confounds POPE: A Systematic Audit of Yes/No Extraction in Vision-Language Model Hallucination Evaluation (2026)**

---

## Contents

| File                     | Description                                                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `VERIFICATION_REPORT.md` | Independent verification of reported results against archived experiment artifacts and regenerated evaluation outputs. |

---

## Reproducibility

All numerical claims reported in the paper are derived from artifacts archived under:

```text
experiments/
```

The repository contains:

* evaluation outputs
* diagnostic artifacts
* cross-model audits
* ablation results
* verification reruns
* supporting analysis scripts

Researchers can independently validate reported metrics by tracing claims directly to the corresponding JSON artifacts.

---

## Audit Principles

This project follows four principles:

1. Artifact-backed reporting.
2. No fabricated measurements.
3. Explicit documentation of reruns and environment differences.
4. Preservation of original evaluation provenance.

Whenever regenerated results differed from archived results, both provenance and rationale were documented in the verification report.

---

## Public Repository Scope

This repository serves as the public reproducibility companion for the project.

The public repository contains:

* source code
* evaluation pipelines
* experiment artifacts
* analysis scripts
* audit documentation

Venue-specific manuscript sources and drafting infrastructure are maintained separately from the public reproducibility repository.

---

## Citation

If this repository contributes to your work, please cite the accompanying paper and repository release.

See:

* `CITATION.cff`
* repository metadata
* official archival release
