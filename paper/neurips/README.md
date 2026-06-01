# UGAA Paper - NeurIPS 2026 Overleaf Package

This is the **NeurIPS-style** package using the `neurips_2026.sty` template.

## Files
- `main.tex`           - full paper source (with 4-sample image figure)
- `references.bib`     - 13 cited references, all verified
- `checklist.tex`      - NeurIPS 2026 paper checklist
- `neurips_2026.sty`   - NeurIPS 2026 template (provided)
- `figures/`           - POPE sample images (sample_tp/fp/tn/fn.jpg)

## Upload to Overleaf
1. Compress this entire folder (including `figures/`) into a ZIP.
2. Overleaf → New Project → Upload Project → choose the ZIP.
3. Overleaf sets `main.tex` as the main document automatically.

## Compile
- Compiler: pdfLaTeX (default).
- First compile may take ~20s due to TikZ/pgfplots; subsequent compiles are fast.

## Switch submission mode
In `main.tex` line 6, the template is loaded as `\usepackage[preprint]{neurips_2026}`.
- For anonymous E&D submission:  `\usepackage[eandd]{neurips_2026}`
- For accepted camera-ready:     `\usepackage[eandd, final]{neurips_2026}`

## Validation status
All cross-references, citations, environments, and braces verified balanced.
13/13 cited keys present in `references.bib`. No undefined `\ref`s.
