# archive/

Centralized archive for files not needed to reproduce the paper results.
The minimum working set lives in `src/pope_audit/`, `scripts/`, `analysis/`,
and `experiments/`.

## Contents

| Directory | What |
|-----------|------|
| `src_legacy/` | 26 scripts from the pre-paper UGAA investigation (spatial-reasoning, WhatsUp, CASH runners, v6 gate experiments, smoke tests, dataset explorers). See `src_legacy/README.md` for the per-file table. |
| `experiments_legacy/` | Prediction logs from the old UGAA line of work: VSR beta-sweeps, WhatsUp/CASH evaluations, 100-sample POPE subsets, and the v6 validation JSON. None of these are cited in the paper. |
| `notebooks/` | Cloud GPU execution notebooks (Kaggle T4). Convenience wrappers, not part of the evaluation pipeline. |
| `misc/` | One-shot utilities and notes from the pre-paper line: `make_kaggle_zips.py` (Kaggle upload helper), `verify.py` (zip inspector), `run_full_diagnostic_3000q.py.bak` (backup), `vsr_error_breakdown.md` (VSR spatial-reasoning false-positive notes from the old UGAA experiments; not part of the POPE token-set paper). |

## Provenance

All files were moved here without modification on 2026-06-14 as part of
the PyPI packaging restructure. Original paths are recoverable from git
history.
