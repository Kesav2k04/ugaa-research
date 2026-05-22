# src/fix_tokenizer.py — KESAV
# Deletes bad tokenizer.json and redownloads clean copy
import os, shutil

model_dir = r"D:\models\llava-1.5-7b"

# Step 1: Delete corrupted files
to_delete = ["tokenizer.json", "tokenizer_config.json", 
             "tokenizer.model", "special_tokens_map.json"]
for f in to_delete:
    path = os.path.join(model_dir, f)
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted: {path}")

# Also clear HuggingFace cache for this model
cache_dir = os.path.expanduser(r"~\.cache\huggingface\hub")
llava_cache = os.path.join(cache_dir, "models--llava-hf--llava-1.5-7b-hf")
if os.path.exists(llava_cache):
    shutil.rmtree(llava_cache)
    print(f"Cleared HF cache: {llava_cache}")

# Step 2: Redownload fresh
from huggingface_hub import hf_hub_download
files = ["tokenizer.json", "tokenizer_config.json", 
         "tokenizer.model", "special_tokens_map.json"]
for f in files:
    try:
        hf_hub_download(
            repo_id="llava-hf/llava-1.5-7b-hf",
            filename=f,
            local_dir=model_dir,
            force_download=True
        )
        print(f"Downloaded: {f}")
    except Exception as e:
        print(f"Skip {f}: {e}")

# Step 3: Verify
import json
tok_path = os.path.join(model_dir, "tokenizer.json")
with open(tok_path, encoding="utf-8") as f:
    data = json.load(f)
print(f"\ntokenizer.json VALID")
print(f"Keys: {list(data.keys())[:4]}")
print(f"Size: {os.path.getsize(tok_path)} bytes")