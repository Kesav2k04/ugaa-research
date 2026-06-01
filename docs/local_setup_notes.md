# Local Setup Notes

These notes record the two machines we used while developing the
project. They are kept for our own reference and are not part of the
required reproducibility instructions in the top-level `README.md`.

## Primary inference machine: ROG Strix G15

```bash
conda create -n ugaa python=3.10 -y
conda activate ugaa

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

Free models used (download automatically on first use via the
HuggingFace cache):

- LLaVA-1.5-7B: `llava-hf/llava-1.5-7b-hf`
- CLIP-B/32:    `openai/clip-vit-base-patch32`
- CLIP-L/14-336:`openai/clip-vit-large-patch14-336`
- Optional second model: `llava-hf/llava-v1.6-mistral-7b-hf`

## Secondary machine: HP Pavilion (data + light eval)

The 4 GB VRAM on this machine is too small to host LLaVA-1.5-7B in
4-bit; it is used only for dataset preparation, metric aggregation,
and figure rendering.

```bash
conda create -n ugaa-eval python=3.10 -y
conda activate ugaa-eval

pip install torch torchvision --index-url \
  https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

## Dataset downloads

- POPE: `python scripts/download_pope_full.py --splits adversarial popular random`
- What's Up (optional): see `scripts/download_whatsup*.py`
- COCO images: download via the dataset's instructions; the POPE JSONs
  in `datasets/pope/` reference local image paths.

## Free metric libraries (no paid APIs)

```bash
pip install rouge-score nltk bert-score evaluate
```

Paid LLM APIs (OpenAI, Gemini paid) are intentionally not used.
