# UGAA Paper — arXiv-Style Standalone Package

This is the **arXiv preprint** package using a plain `article` document class.
No template `.sty` file required — only standard CTAN packages.

## Files
- `main.tex`        — full paper source, self-contained
- `references.bib`  — 13 cited references, all verified
- `figures/`        — POPE sample images (sample_tp/fp/tn/fn.jpg)

## Upload to Overleaf
1. Compress this entire folder (including `figures/`) into a ZIP.
2. Overleaf → New Project → Upload Project → choose the ZIP.
3. Overleaf sets `main.tex` as the main document automatically.

## Compile
- Compiler: pdfLaTeX (default).
- First compile takes ~25s for the bib + TikZ render; subsequent compiles are fast.

## Differences vs. the NeurIPS package
- Uses `\documentclass[11pt,letterpaper]{article}` with `geometry` for 1-inch margins.
- `authblk` for clean author block; `titling` not needed.
- `fancyhdr` runs a discreet header on every page.
- No `neurips_2026.sty` dependency — portable to any LaTeX install.
- No NeurIPS checklist (this is the preprint form).
- Identical content, identical tables, identical figures, identical results.

## arXiv submission
This package compiles directly under arXiv's TeX Live; just upload the ZIP
to arxiv.org → submit → choose pdfLaTeX. Strip the `figures/` subfolder prefix
or keep it — arXiv accepts both.

## Validation status
All cross-references, citations, environments, and braces verified balanced.
13/13 cited keys present in `references.bib`. No undefined `\ref`s.
