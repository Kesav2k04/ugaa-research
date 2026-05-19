import json
import os

folder_path = r"G:\AI BASED PROJECT\ugaa-research\datasets\gqa\questions"
files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

numeric_answers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

print(f"Scanning {len(files)} files for counting questions...")

for file in files:
    file_path = os.path.join(folder_path, file)
    try:
        with open(file_path, encoding='utf-8') as f:
            qs = json.load(f)
        
        count = 0
        for v in qs.values():
            ans = v.get('answer', '').lower()
            if ans.isdigit() or ans in numeric_answers:
                count += 1
                
        print(f"{file}: {count} counting questions")
    except Exception as e:
        print(f"{file}: Error reading file - {e}")

print("Scan complete.")
