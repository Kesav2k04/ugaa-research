# src/download_model.py — KESAV runs this (background, takes 30-60 min)
from huggingface_hub import snapshot_download
import os

save_path = "D:/models/llava-1.5-7b"
os.makedirs(save_path, exist_ok=True)

print("Downloading LLaVA-1.5-7B... this will take 30-60 minutes.")
snapshot_download(
    repo_id="llava-hf/llava-1.5-7b-hf",
    local_dir=save_path,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"]
)
print("Download complete. Model saved to D:/models/llava-1.5-7b")