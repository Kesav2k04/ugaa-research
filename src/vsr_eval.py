# src/vsr_eval.py — KARTHIGEYAN
# Why: VSR tests spatial reasoning — exactly where AIR failed at 5.4pp
# This script loads VSR and prepares it for model evaluation
import json, os

def load_vsr(vsr_folder: str) -> list:
    """
    Load VSR dataset. Format: each entry has
    - image: filename
    - caption: describes spatial relation
    - label: True/False (does caption match image?)
    """
    data = []
    for fname in os.listdir(vsr_folder):
        if fname.endswith('.jsonl'):
            with open(os.path.join(vsr_folder, fname)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
    return data

def build_vsr_question(entry: dict) -> str:
    """Convert VSR entry to yes/no question for VLM."""
    return f"Is the following statement true about the image: '{entry['caption']}'? Answer yes or no only."

if __name__ == "__main__":
    vsr_path = r"G:\AI BASED PROJECT\visual-spatial-reasoning"
    
    # Find the actual data files
    print("Files in VSR folder:")
    for root, dirs, files in os.walk(vsr_path):
        for f in files:
            full = os.path.join(root, f)
            print(f"  {full}")
    
    data = load_vsr(vsr_path)
    if not data:
        # Try recursive search for jsonl files
        for root, dirs, files in os.walk(vsr_path):
            for f in files:
                if f.endswith('.jsonl'):
                    with open(os.path.join(root, f)) as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                data.append(json.loads(line))
    
    print(f"\nTotal VSR entries: {len(data)}")
    if data:
        print(f"Keys in entry: {list(data[0].keys())}")
        print(f"Sample entry: {data[0]}")
        print(f"Sample question: {build_vsr_question(data[0])}")