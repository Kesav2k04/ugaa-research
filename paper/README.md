# Paper sources

Three parallel builds of *Token-Set Choice Confounds POPE*. Same data,
same figures, same numbers, different layout per venue requirement.

| Folder | Use it when... | Document class | Authors visible? | Compiler |
|---|---|---|---|---|
| [`neurips/`](neurips/) | Submitting to NeurIPS 2026 (E&D track) | `neurips_2026.sty` with `[eandd]` | No (anonymous) | pdfLaTeX |
| [`arxiv/`](arxiv/) | Posting an arXiv preprint with names | standard `article` + `geometry` | Yes | pdfLaTeX |
| [`html/`](html/) | Browser-readable mirror (e.g. arXiv HTML) | static HTML + MathJax + CSS | Yes | open `index.html` |

## Numbers in the paper

Every numeric claim in `main.tex` and `index.html` is traceable to a
JSON file in [`../experiments/`](../experiments/). Run
`python ../scripts/fact_audit.py` (if you have one staged locally) or
the equivalent quick check:

```python
import json
adv = json.load(open("experiments/pope_adversarial_2tok_vs_8tok.json"))
print(adv["summary"]["eval_8token"])  # F1 = 0.8221, etc.
```

## Figures

All figures in `paper/*/figures/` are generated from real data, never
fabricated. The originating scripts live under
[`../analysis/`](../analysis/) and produce PDF (LaTeX) and PNG (HTML)
side by side.

| Figure | What it shows | Source data |
|---|---|---|
| `top_token_frequency.pdf` | First-token counts per surface form on each split | `experiments/pope_*_2tok_vs_8tok.json` |
| `confusion_matrices_adv.pdf` | 2x2 confusion under both protocols on adversarial | `experiments/pope_adversarial_2tok_vs_8tok.json` |
| `yes_rate_convergence.pdf` | Running yes-rate over 3000 adversarial questions | same |
| `yes_rate_all_splits.pdf` | Same plot for all three splits, both protocols | all three split JSONs |
| `baseline_spread.pdf` | Reported vs corrected baselines across recent papers | hard-coded values from cited PDFs |
| `correction_methods_bar.pdf` | F1 of nine inference-time corrections | `experiments/pope_full_adversarial_*_summary.json` |
| `precision_recall_methods.pdf` | Same on the precision-recall plane | same |
| `fig7_attention_histogram_3000q.pdf` | Mean image attention by correctness | `experiments/ugaa_full_adversarial_3000q_diagnostic.json` |
| `fig8_logit_gap_3000q.pdf` | Logit-gap mean + density by category | same |

## Building the LaTeX

Either folder compiles with a single pdfLaTeX call (Overleaf
auto-detects the main file):

```bash
cd paper/neurips
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

The arXiv folder is identical except `pdflatex` runs without
`neurips_2026.sty`.

## Switching the NeurIPS template mode

`paper/neurips/main.tex` currently loads the template in anonymous
submission mode:

```latex
\usepackage[eandd]{neurips_2026}
```

To produce a non-anonymous preprint with author names visible, change
that one line to `\usepackage[preprint]{neurips_2026}` and put the
authors back into the `\author{}` block (an example author block is
left as a comment in `main.tex`). For the camera-ready version after
acceptance, use `\usepackage[eandd, final]{neurips_2026}`.
