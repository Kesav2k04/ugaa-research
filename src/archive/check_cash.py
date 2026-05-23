# src/check_cash.py — KARTHIGEYAN
# Why: verify CASH benchmark has correct categories and no bad entries
import json

with open(r"G:\AI BASED PROJECT\ugaa-research\datasets\cash\cash_v1_partial.json") as f:
    data = json.load(f)

from collections import Counter
cats = Counter(item['category'] for item in data)
print(f"Total entries: {len(data)}")
print(f"Categories: {dict(cats)}")
print(f"\nSample spatial:   {data[0]}")
print(f"Sample counting:  {[d for d in data if d['category']=='counting'][0]}")
print(f"Sample attribute: {[d for d in data if d['category']=='attribute'][0]}")

# Check for empty questions
empty = [d for d in data if not d['question'].strip()]
print(f"\nEmpty questions: {len(empty)}")