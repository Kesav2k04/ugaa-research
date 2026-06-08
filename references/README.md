# References
Reading notes and BibTeX entries for the papers cited in
*Token-Set Choice Confounds POPE*. Each entry lists what the paper
does, what it reports for LLaVA-1.5-7B POPE, and what we use it for in
our own paper.

## Quick index
| Key | Authors (year) | What they do | Why we cite them |
|---|---|---|---|
| `li2023evaluating` | Li, Du, Zhou, Wang, Zhao, Wen (2023) | Introduces the POPE benchmark and its 2-token logit evaluator. | Original benchmark we audit. |
| `liu2023improved`  | Liu, Li, Li, Lee (2023) | LLaVA-1.5: CLIP-ViT-L/14-336 + Vicuna-1.5-7B + MLP projector. | The model we run the audit on. |
| `kudo2018sentencepiece` | Kudo and Richardson (2018) | SentencePiece subword tokenizer. | Explains the space-prefix tokens that break the 2-token rule. |
| `touvron2023llama2` | Touvron et al. (2023) | LLaMA-2 model family and its tokenizer. | Used by Vicuna-1.5, hence by LLaVA-1.5. |
| `radford2021learning` | Radford et al. (2021) | CLIP. | LLaVA's vision encoder; CLIP-L grounding correction method. |
| `lin2014microsoft` | Lin et al. (2014) | COCO dataset. | Image source under POPE. |
| `dettmers2023qlora` | Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023) | 4-bit NF4 quantization (QLoRA). | Our run configuration. |
| `liang2022holistic` | Liang et al. (2022) | HELM: benchmark-protocol sensitivity. | Frames our finding inside the broader evaluation-sensitivity literature. |
| `zheng2023judging` | Zheng et al. (2023) | LLM-as-judge ordering effects. | Related class of protocol-sensitivity. |
| `leng2024mitigating` | Leng et al. (2024) | Visual contrastive decoding (VCD). | Inspires our VCD-noise correction variant. |
| `visflow2025` | Tang et al. (arXiv:2506.12609, 2025) | Dual-level attention intervention for VLM hallucination. | Re-evaluated against corrected baseline: +10.81 reported gain reduces to +2.14. |
| `seo2025` | Seo, Kang, Cho, Lee, Chun (arXiv:2510.09008, 2025) | Epistemic uncertainty over visual tokens. | Re-evaluated against corrected baseline: reported result of 80.00 falls 2.21 points below 82.21. |
| `adaptvis2025` | Chen et al. (ICML 2025, arXiv:2503.01773) | AdaptVis: confidence-guided attention scaling for spatial reasoning. | Re-evaluated against corrected baseline: reported result of 81.80 falls 0.41 points below 82.21. |

## BibTeX
A copy is included here as
`references.bib` for offline reading.

## PDFs for re-evaluated prior methods
The three prior methods we re-evaluate in Section 4 (`visflow2025`, `seo2025`,
`adaptvis2025`) are large arXiv PDFs and are intentionally not included
in this repository. The arXiv IDs above resolve directly:
- `arXiv:2506.12609` -- Tang et al.
- `arXiv:2510.09008` -- Seo et al.
- `arXiv:2503.01773` -- Chen et al.

## How we computed the gain-vs-corrected-baseline column
Each prior method reports a baseline B_p and a result R_p. Our corrected
adversarial baseline is `B* = 82.21`. The gain that would remain if the
method were evaluated against B* is simply `R_p - B*` (it is `negative`
when R_p < B*). The published gain `R_p - B_p` is not directly
comparable because B_p ranged from 73.54 to 81.00 across the three
papers, a 7.46 point spread that is consistent with the token-extraction
artifact our paper documents.
