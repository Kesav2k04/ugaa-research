import json

with open(r"G:\AI BASED PROJECT\ugaa-research\datasets\cash\cash_v1_partial.json") as f:
    cash_data = json.load(f)

with open(r"G:\AI BASED PROJECT\ugaa-research\datasets\cash\cash_image_mappings.json") as f:
    mappings = json.load(f)

export = []
for item in cash_data[:100]:
    qid = item["id"]
    m = mappings.get(qid)
    if m and m.get("urls"):
        export.append({
            "question_id": qid,
            "image_url": m["urls"][0],
            "question": item["question"],
            "label": item["answer"],
            "category": item["category"]
        })

with open(r"G:\AI BASED PROJECT\ugaa-research\datasets\cash\cash_sample_100.json", "w") as f:
    json.dump(export, f, indent=2)

print(f"Exported {len(export)} entries")
print("Sample:", export[0])