# src/archive

Scripts that were used during the investigation but are not part of
the paper pipeline. Kept here verbatim for provenance.

## Contents

| File | What it was for |
|---|---|
| `ugaa_gate_v6.py`, `run_ugaa_v6_100q.py`, `diag_clip_l_v6.py` | A confidence-gated YES-boost gate that we tried as a successor to the v5 NO-bias variants. The 100-question diagnostic in Section 5.3 of the paper is the last thing this code produced; the gate itself did not beat the corrected baseline and is not referenced in the paper. |
| `run_pope_eval.py`, `run_pope_eval_ugaa.py` | Earlier POPE eval drivers superseded by `src/run_pope_eval_full.py`. |
| `run_vsr_eval.py`, `run_vsr_eval_ugaa.py`, `vsr_eval.py`, `run_whatsup_eval.py`, `run_cash_eval.py` | Spatial-reasoning, WhatsUp, and CASH evaluation runners. Not used in the paper. |
| `diag.py`, `diag_clip_l.py` | Ad-hoc probes of CLIP-L similarity used while we were tuning the CLIP variants. |
| `clip_distance.py` | First-pass CLIP image-text similarity helper, predates `clip_l_grounding.py`. |
| `test_attention.py`, `test_llava.py`, `test_llava_hf.py` | Smoke tests written while wiring up the model loader. |
| `explore_gqa.py`, `explore_pope.py`, `check_all_gqa.py`, `check_cash.py` | Exploratory dataset checks. |
| `create_pope_100.py`, `create_vsr_100.py`, `create_cash_100.py` | Helpers that produced the 100-sample subsets used for early debugging. |
| `download_model.py`, `fix_tokenizer.py`, `run.bat` | One-shot setup scripts. |

Nothing here should be needed to reproduce the paper. The minimum
working set lives one level up in `src/`.
