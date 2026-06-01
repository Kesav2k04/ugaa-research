# Cloud GPU runbook (Kaggle / Colab)

This is the operating procedure for running the multi-model audit, the
full 3,000-question diagnostic, and the latency micro-benchmark on free
cloud GPUs. Your local 8 GB RTX 3070 Ti is enough for LLaVA-1.5-7B and
LLaVA-1.6-Mistral-7B at 4-bit, but it runs out of activation memory on
InstructBLIP / mPLUG-Owl2 / Qwen-VL even at 4-bit. Kaggle's T4 x 2
(30 GB combined) and Colab's T4 (16 GB) absorb all five comfortably.

## 1. Accounts and quotas

| Platform | GPU | RAM | Disk | Free quota |
|---|---|---|---|---|
| Kaggle | T4 x 2 (15 GB each), or P100 (16 GB) | 13 GB | 73 GB | 30 GPU-hours / week, 2 concurrent sessions |
| Colab Free | T4 (15 GB) | 12 GB | 78 GB | ~12 h/session, ~25 GPU-hours / week, 1 concurrent session |
| Colab Pro ($10/mo) | T4 or A100 (40 GB on busy days) | 51 GB | 167 GB | longer sessions, priority |

Single Google account gives you **3 concurrent jobs** if you use both
Kaggle (2) and Colab (1). With one extra Google account you can double
the Colab side (account-isolated). Kaggle's ToS forbids multiple
accounts per person.

## 2. The three cloud notebooks

| Notebook | Purpose | Output |
|---|---|---|
| `notebooks/run_on_cloud.ipynb` | Multi-model audit on one model x 3 splits x 1-2 prompt modes | `experiments/multi_model/<model>_pope_<split>_<n>q_<mode>_token_audit.json` |
| `notebooks/run_full_diagnostic_on_cloud.ipynb` | Full 3000q diagnostic for one POPE split on LLaVA-1.5-7B | `experiments/ugaa_full_<split>_3000q_diagnostic.json` |
| `notebooks/run_latency_on_cloud.ipynb` | 50-question latency micro-benchmark | `experiments/latency_microbench.json` |

Each notebook is self-contained: it clones the repo, downloads POPE
parquet from `lmms-lab/POPE`, installs deps, runs the script, and
saves outputs to `/kaggle/working/outputs/` (Kaggle) or
`./cloud_run/.../outputs/` (Colab).

## 3. Parallel execution plan

Open three browser tabs and dispatch in this order. Each row is one
notebook in one platform; rows in the same column run sequentially.

| Slot | Kaggle session 1 | Kaggle session 2 | Colab session |
|---|---|---|---|
| 1 | multi_model: `instructblip` (~2.5 h) | multi_model: `mplug_owl2` (~3 h) | multi_model: `qwen2_vl` (~2.5 h) |
| 2 | multi_model: `llava16_mistral` (~2 h) | diagnostic_3000q: `adversarial` (~1 h) | diagnostic_3000q: `popular` (~1 h) |
| 3 | diagnostic_3000q: `random` (~1 h) | latency_microbench (~5 min) | -- |

Total wall-clock with 3 concurrent jobs: roughly 5-6 hours. Total
GPU-hours consumed: roughly 15-16, comfortably inside the weekly
free quotas.

## 4. Per-notebook procedure

### Multi-model audit (`run_on_cloud.ipynb`)

1. Open the notebook on Kaggle: New Notebook -> File -> Import Notebook -> upload the `.ipynb`.
2. Settings -> Accelerator -> **T4 x 2** (or P100). Internet -> **On**.
3. Edit cell 1: set `MODEL` to one of
   `llava15 | llava16_mistral | instructblip | mplug_owl2 | qwen2_vl`.
4. Optional: trim `SPLITS` or set `SAMPLES = 500` for a smoke test.
5. Run all cells. The audit prints F1/P/R per readout per split as it
   finishes each run.
6. When done, files appear under the notebook's "Output" panel; the
   JSON names start with the model slug (e.g.
   `instructblip_pope_adversarial_3000q_paper_template_token_audit.json`).

### Full 3000q diagnostic (`run_full_diagnostic_on_cloud.ipynb`)

Same procedure with `SPLIT` (single value). Output is one diagnostic
JSON per split.

### Latency benchmark (`run_latency_on_cloud.ipynb`)

Same procedure. 5 minutes. Output is one JSON, `latency_microbench.json`.

## 5. Pulling results back to the local repo

After each notebook finishes:

1. **Kaggle**: click the notebook's Output tab, download each `.json`.
2. **Colab**: in the file browser, right-click each `.json` -> Download.
3. Save the downloaded files to a single folder on your laptop, e.g.
   `D:\cloud_results\<date>\`.
4. Run the pull script to copy them into the right place:

```powershell
python scripts/pull_cloud_results.py --dir D:\cloud_results\2026-05-30\
```

The script categorises by filename pattern and copies into:

- `experiments/multi_model/*.json`
- `experiments/ugaa_full_*_3000q_diagnostic.json`
- `experiments/latency_microbench.json`

Then run the audit chain:

```powershell
python analysis/fact_audit.py
python analysis/string_parse_equivalence.py
python analysis/diagnostic_stats.py --diagnostic experiments/ugaa_full_adversarial_3000q_diagnostic.json
```

## 6. Troubleshooting

### `transformers` version mismatch

Qwen2-VL requires `transformers>=4.45`. Older audits stay on `4.40.1`.
The cloud notebook installs the right version per `MODEL`; if you mix
them in one notebook, restart the kernel between runs.

### Out-of-memory on T4

Drop to one notebook per Kaggle account (single GPU only) and uncheck
T4 x 2 in favour of P100 (16 GB single GPU). The audit script defaults
to 4-bit NF4; if you still see OOM on Qwen2-VL or mPLUG-Owl2, edit the
notebook to set `QUANTIZE = '8bit'`.

### `trust_remote_code` warnings

mPLUG-Owl2 ships custom modeling code. The cloud notebook accepts this
automatically because the adapter passes `trust_remote_code=True`.
Review the repo `MAGAer13/mplug-owl2-llama2-7b` first if you are
concerned about untrusted code.

### Kaggle says "session ran out of GPU time"

Kaggle resets the 30-hour weekly quota every Saturday 00:00 UTC. Plan
heavy runs early in the week.

### Colab disconnects after idle

Free Colab kills sessions after ~90 minutes of browser idle. Keep the
tab visible or use the small JavaScript heartbeat trick (paste into
the Colab console):

```javascript
function keepAlive() { document.querySelector("#connect").click(); }
setInterval(keepAlive, 60000);
```

This is well-known and not against Colab ToS for short sessions, but
do not abuse it for multi-day runs.

## 7. Status tracking

Append one row to `experiments/multi_model/_status.csv` per cloud run:

```
date,platform,model,split,prompt_mode,n_valid,f1_dynamic_single,wall_clock_min
2026-05-30,kaggle,instructblip,adversarial,paper_template,3000,...,153
...
```

This file is the canonical place for the cloud-side status. The
multi-model section of the paper draws its numbers from
`experiments/multi_model/*.json` directly, so the CSV is mainly an
operational checklist.

## 8. After all cloud runs finish

1. Confirm everything is in place:

```powershell
python analysis/fact_audit.py
```

   Phases 1-6 should all be green; Phase 5 now reports per-model
   equivalence rates if `top_token_id` is recorded in each multi-model
   JSON. Update `fact_audit.py` Phase 4 with the new models' expected
   F1 numbers if you want it to assert against them.

2. Regenerate the cross-model figures (the bar chart should now have a
   group per model, not just LLaVA-1.6):

```powershell
python analysis/make_figures_main.py
```

3. Rebuild the three Overleaf ZIPs:

```powershell
python -c "from pathlib import Path; import shutil, zipfile; RD=Path('D:/RESEARCH DOCS 2026'); ..."
```

   (Or just re-run the existing zip-build cell from the previous
   session.)

That's the loop: run on cloud, pull, audit, rebuild zips, submit.
