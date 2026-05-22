import os
import json
from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "Mayfull/whats_up_vlms"
FILENAME = "test.json"  # the test split JSON provided in the repo
OUTPUT_DIR = os.path.join("datasets", "whatsup")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "whatsup_sample_100.json")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading test.json from Hugging Face Hub …")
# This will download the file to a cache folder and return the local path
local_file = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
print(f"Downloaded to {local_file}")

# Load the original test data (expected to be a list of dicts)
with open(local_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Original dataset size: {len(data)} entries")

# Build a trimmed sample (first 100 entries) with the fields we need
sample = []
for i, item in enumerate(data):
    if i >= 100:
        break
    sample.append({
        "question_id": i,
        "image_url": item.get("image_url", item.get("url", "")),
        "question": item.get("question", item.get("caption_main", "")),
        "label": item.get("label", "A"),
    })

# Write the sample to the project dataset directory
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=2, ensure_ascii=False)

print(f"Done: {len(sample)} questions saved to {OUTPUT_PATH}")
