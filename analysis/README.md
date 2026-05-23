# Analysis

Scripts that regenerate every figure and audit every numeric claim
in the paper.

## `make_figures_main.py`

Reads JSON files from `../experiments/` and produces seven figures in
PDF (for LaTeX) and PNG (for HTML). Writes into both
`../paper/neurips/figures/`, `../paper/arxiv/figures/`, and
`../paper/html/`:

- `top_token_frequency`
- `confusion_matrices_adv`
- `yes_rate_convergence`
- `baseline_spread`
- `correction_methods_bar`
- `grounding_vs_correct`
- `logit_gap_histogram`

## `make_figures_extra.py`

Three more derived figures from the same JSONs:

- `precision_recall_methods` (P-R plane of the nine correction methods)
- `yes_rate_all_splits` (per-split running yes-rate, both protocols)
- `logit_gap_two_panel` (means with error bars + per-category density)

## `fact_audit.py`

Walks every numeric claim that appears in the paper and re-derives it
from the JSONs under `../experiments/`. Prints `OK` per row and a final
`ALL FACTS VERIFIED` line.

Run all three:

```bash
python analysis/make_figures_main.py
python analysis/make_figures_extra.py
python analysis/fact_audit.py
```

## Other notes

`karthigeyan_analysis.md`, `vsr_error_breakdown.md`, and
`full_project_summary.md` are informal research diaries kept during
the investigation. They are not referenced from the paper.
