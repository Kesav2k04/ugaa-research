# src/explore_gqa.py  (KARTHIGEYAN runs this on HP Pavilion)
import json

# Load a small sample of questions
with open('datasets/gqa/questions/val_balanced_questions.json', 'r') as f:
    questions = json.load(f)

# Look at the first 5 questions
for i, (qid, q) in enumerate(questions.items()):
    if i >= 5:
        break
    print(f"\nQuestion ID: {qid}")
    print(f"  Question: {q['question']}")
    print(f"  Answer:   {q['answer']}")
    print(f"  Types:    {q.get('types', {})}")