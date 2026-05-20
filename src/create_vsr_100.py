import json
import os

vsr_file = r"G:\AI BASED PROJECT\visual-spatial-reasoning\data\data_files\all_vsr_validated_data.jsonl"

questions = []
with open(vsr_file, "r") as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

export = []
for i, q in enumerate(questions[:100]):
    label_str = "yes" if q["label"] == 1 else "no"
    export.append({
        "question_id": i + 1,
        "image_url": q["image_link"],
        "question": q["caption"],
        "label": label_str
    })

os.makedirs("datasets/vsr", exist_ok=True)
with open("datasets/vsr/vsr_sample_100.json", "w") as f:
    json.dump(export, f, indent=2)

print(f"Exported {len(export)} questions to datasets/vsr/vsr_sample_100.json")
