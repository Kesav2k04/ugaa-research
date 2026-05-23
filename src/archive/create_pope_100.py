import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pope_loader import load_pope_split

# Load from his pope_loader
pope_folder = r"G:\AI BASED PROJECT\POPE\output\coco"
questions = load_pope_split(pope_folder, 'adversarial')  # his existing function

# Export first 100 as portable format with image URLs
export = []
for q in questions[:100]:
    export.append({
        "question_id": q["question_id"],
        "image_url": f"http://images.cocodataset.org/val2017/{q['image']}",
        "question": q["text"],
        "label": q["label"]
    })

with open("datasets/pope/pope_sample_100.json", "w") as f:
    json.dump(export, f, indent=2)