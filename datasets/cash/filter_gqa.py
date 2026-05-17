# datasets/cash/filter_gqa.py
# KARTHIGEYAN — filter GQA questions by semantic category
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, '..', 'gqa', 'questions', 'val_balanced_questions.json')

with open(input_path, encoding='utf-8') as f:
    all_questions = json.load(f)

spatial, counting, attribute = [], [], []

for qid, q in all_questions.items():
    qtext = q['question'].lower()
    types = q.get('types', {}).get('structural', '')
    
    # Spatial relations
    spatial_words = ['left', 'right', 'above', 'below', 
                     'behind', 'front', 'next to', 'between']
    if any(w in qtext for w in spatial_words):
        spatial.append({'id': qid, 'question': q['question'],
                        'answer': q['answer'], 'category': 'spatial'})
    
    # Counting — NOTE: GQA val_balanced has NO counting/numeric questions.
    # All answers are text-based (yes/no, colors, objects, directions).
    # TODO: Source counting questions from VQAv2 or another dataset.
    elif q['answer'].isdigit() or q['answer'].lower() in [
            'zero', 'one', 'two', 'three', 'four', 'five',
            'six', 'seven', 'eight', 'nine', 'ten']:
        counting.append({'id': qid, 'question': q['question'],
                         'answer': q['answer'], 'category': 'counting'})
    
    # Attribute binding
    elif any(w in qtext for w in ['color', 'size', 'shape', 
                                   'material', 'what is the']):
        attribute.append({'id': qid, 'question': q['question'],
                          'answer': q['answer'], 'category': 'attribute'})

# Take 100 from each category
cash_data = spatial[:100] + counting[:100] + attribute[:100]

output_path = os.path.join(script_dir, 'cash_v1_partial.json')

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(cash_data, f, indent=2)

print(f"Spatial: {len(spatial[:100])}")
print(f"Counting: {len(counting[:100])}")
print(f"Attribute: {len(attribute[:100])}")
print(f"Total saved: {len(cash_data)} questions")