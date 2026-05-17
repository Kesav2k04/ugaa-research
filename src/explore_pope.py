# src/explore_pope.py — KARTHIGEYAN runs on HP Pavilion
# Why: understand POPE format before building the full loader
import json, os

# Update this path to where your POPE folder is on G drive
pope_path = r"G:\AI BASED PROJECT\POPE\output\coco"

# List what's inside
print("Files in POPE folder:")
for f in os.listdir(pope_path):
    print(f" -", f)

# Load one file and print first 3 entries
files = [f for f in os.listdir(pope_path) if f.endswith('.json')]
if files:
    with open(os.path.join(pope_path, files[0])) as f:
        lines = f.readlines()[:3]
    print(f"\nFirst 3 entries from {files[0]}:")
    for line in lines:
        entry = json.loads(line)
        print(entry)
        print("Keys:", list(entry.keys()))
