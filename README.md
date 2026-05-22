# 📚 WhatsUp Benchmark for Vision‑Language Model Hallucination

## 🎯 Problem
Modern Vision‑Language Models (VLMs) often **hallucinate** spatial relationships: they generate plausible‑looking captions that **do not match the actual image content**. This undermines trust in AI systems applied to robotics, AR, and content moderation.

## 💡 Solution
We provide a **compact evaluation suite** (`run_whatsup_eval.py`) that measures a model’s ability to correctly choose between two spatially‑flipped captions. The suite:
- Loads a curated **WhatsUp** test split (100 samples).
- Generates a multiple‑choice prompt for any LLaVA‑style model.
- Reports **accuracy** on the spatial reasoning task.

## 🚀 Installation & Quick Start
```bash
# Clone the repository (already done)
# Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt  # includes torch, transformers, datasets, etc.

# Download the sample WhatsUp dataset (100 entries)
python scripts/download_whatsup_from_github.py

# Run the evaluation (no GPU needed, runs on CPU/CPU‑compatible 4‑bit quantised model)
python src/run_whatsup_eval.py
```

## 📊 Results (sample placeholder)
| Model | 4‑bit Quantised | Accuracy |
|-------|----------------|----------|
| LLaVA‑1.5‑7B | ✅ | **71.2 %** |
| LLaVA‑1.5‑7B (FP16) | ✅ | **78.5 %** |
| LLaVA‑1.5‑7B (GPT‑4‑style prompt) | ✅ | **82.0 %** |

> **Note:** Replace the table with your own scores after running the benchmark.

## 📂 Repository Structure
```
├─ datasets/
│   └─ whatsup/whatsup_sample_100.json   # evaluation data (100 samples)
├─ scripts/
│   └─ download_whatsup_from_github.py  # fetches the JSON from the original repo
├─ src/
│   ├─ run_whatsup_eval.py               # main evaluation script
│   ├─ run_cash_eval.py                  # other benchmarks
│   └─ ...
├─ README.md
├─ LICENSE
└─ requirements.txt
```

## 🤝 Contributing
Feel free to open issues or pull requests for:
- Adding more datasets (COCO‑spatial, GQA‑spatial).
- Supporting other VLM families.
- Improving the prompt engineering.

## 📄 License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.
