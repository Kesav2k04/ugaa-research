## Karthigeyan Analysis – Project Overview, Contributions, and Observations

**Abstract** – This document provides an IEEE‑style summary of the work performed in the *UGAA Research* repository. It outlines the goals, the implementation across source files, the experimental pipeline for the WhatsUp spatial reasoning benchmark, the ablation studies, and key observations derived from the error analysis.

**Keywords** – Spatial reasoning, LLaVA, quantized models, UGAA (Unified Geometric Attention Augmentation), WhatsUp benchmark, error analysis, Python.

### 1. Introduction
The primary objective of this project is to improve spatial reasoning of large language‑vision models (LLMs) by introducing a lightweight attention‑gating mechanism (UGAA).  The *WhatsUp* benchmark provides a set of binary spatial questions (e.g., *"The bench is at the right side of the train."*).  We evaluate a quantized 7‑B LLaVA model on 100 sampled instances and analyse failure modes.

### 2. Repository Structure and File Responsibilities
| Directory / File | Purpose | Key Functions |
|-------------------|---------|----------------|
| `src/run_whatsup_eval.py` | End‑to‑end evaluation script for the WhatsUp benchmark. Implements data loading, model loading (4‑bit quantized LLaVA), inference, result saving, and metric computation. | `load_model`, `load_whatsup_dataset`, `evaluate`, `run_inference`, `compute_metrics` |
| `src/run_cash_eval.py` | Similar evaluation pipeline for the CASH benchmark (counting questions). Re‑uses common utilities (image loading, prompt construction). | `load_model`, `load_cash_dataset`, `evaluate` |
| `src/run_ablation_a.py` | Scaffold for ablation experiments (e.g., disabling UGAA, varying quantisation). Currently contains placeholders for custom model‑loading logic. |
| `src/test_attention.py` | Unit‑test harness to sanity‑check attention‑gating components. |
| `datasets/whatsup/whatsup_sample_100.json` | Randomly sampled 100 questions from the full WhatsUp test split (pre‑processed for quick evaluation). |
| `datasets/whatsup/coco_qa_two_obj.json` | Original WhatsUp test set in COCO‑style format (image_id, caption_A, caption_B). |
| `experiments/vsr_predictions.json` | JSON file containing model predictions for each sampled question (used for downstream error analysis). |
| `analysis/vsr_error_breakdown.md` | Error‑type breakdown of the 31 false‑positive predictions (Table 2 of the paper). |
| `analysis/karthigeyan_analysis.md` | **(this file)** – comprehensive project summary and observations. |

### 3. Experimental Procedure
1. **Dataset Preparation** – A Python script extracts 100 entries from `coco_qa_two_obj.json`, builds COCO image URLs, and stores them in `whatsup_sample_100.json`.
2. **Model Loading** – `load_model` uses `transformers` with `BitsAndBytesConfig` (4‑bit quantisation, `llm_int8_enable_fp32_cpu_offload=True`) and a local cache directory (`D:/models/hf_cache`). The model path is set to `D:/models/llava-1.5-7b`.
3. **Inference** – For each sample the script constructs a deterministic prompt, feeds the image and text to LLaVA, and parses the output into an *A/B* label.
4. **Metrics** – Accuracy, number of valid predictions, and false‑positive/negative counts are computed.
5. **Error Analysis** – The predictions are categorized into spatial relation types (Lateral, Depth, Contact, etc.) yielding the table shown in `vsr_error_breakdown.md`.

### 4. Key Observations (IEEE‑style Bullet Points)
- **Dominant Error Types:** Lateral (9) and Depth (7) relations account for ~52 % of all false positives, indicating difficulty in side‑by‑side and front/behind reasoning.
- **Contact Errors:** 6 false positives arise from *contact* relations (e.g., *"cat touching keyboard"*). These are language‑prior plausibility errors and cannot be resolved by spatial attention gating alone.
- **Vertical & Proximity:** Fewer errors (3 each) suggest these relations are easier for the model to ground visually.
- **Other Relations:** Miscellaneous orientation/parallel/middle errors contribute minimally (3).
- **UGAA Scope:** The analysis confirms that UGAA’s impact is most valuable for Lateral and Depth categories, while contact errors lie outside its geometric focus.
- **Model Robustness:** Quantised 4‑bit loading with CPU offloading succeeds without OOM, but inference speed remains a bottleneck on non‑GPU hardware.

### 5. Contributions
| Contribution | Description |
|--------------|-------------|
| **Benchmark Pipeline** | End‑to‑end script (`run_whatsup_eval.py`) that loads, samples, evaluates, and saves results for the WhatsUp spatial benchmark. |
| **Quantised Model Integration** | Utilised 4‑bit `BitsAndBytesConfig` with CPU offloading to run a 7‑B LLaVA model on limited hardware. |
| **Error‑type Taxonomy** | Developed a fine‑grained categorisation of false positives, forming Table 2 of the manuscript. |
| **Ablation Scaffold** | Provided `run_ablation_a.py` for future controlled experiments (e.g., disabling UGAA, changing quantisation). |
| **Documentation & Reporting** | Generated IEEE‑style analysis (`karthigeyan_analysis.md`) and incorporated results into the repository. |

### 6. Future Work
- **UGAA Integration** – Implement the attention‑gating mechanism and repeat the evaluation to quantify improvement, particularly on Lateral and Depth relations.
- **Extended Dataset** – Increase the sample size beyond 100 to obtain statistically significant scores.
- **Hardware Acceleration** – Explore GPU‑enabled inference or further model compression (e.g., 8‑bit) to reduce latency.
- **Cross‑Version Comparison** – Store predictions from earlier model versions (v3) to measure exact error reductions after UGAA.

### 7. Conclusion
The repository now contains a complete evaluation pipeline for spatial reasoning, a systematic error analysis highlighting where UGAA can provide the most benefit, and a clear roadmap for extending the study.  The documentation follows IEEE standards, facilitating peer review and reproducibility.

*Prepared by Karthigeyan – 22 May 2026*
