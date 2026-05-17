# Fixed: spatial+attribute from GQA, counting from VQAv2
import json, os

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── SPATIAL + ATTRIBUTE from GQA ──────────────────────────
gqa_path = os.path.join(script_dir, '..', 'gqa', 'questions',
                        'val_balanced_questions.json')
with open(gqa_path, encoding='utf-8') as f:
    gqa_q = json.load(f)

spatial, attribute = [], []
for qid, q in gqa_q.items():
    text = q['question'].lower()
    if any(w in text for w in ['left of','right of','above','below',
                                'behind','in front','next to','between']):
        spatial.append({'id': qid, 'question': q['question'],
                        'answer': q['answer'], 'category': 'spatial'})
    elif any(w in text for w in ['what color','what is the color',
                                  'what shape','what material','what size']):
        attribute.append({'id': qid, 'question': q['question'],
                          'answer': q['answer'], 'category': 'attribute'})
    if len(spatial) >= 100 and len(attribute) >= 100:
        break

# ── COUNTING from VQAv2 ───────────────────────────────────
vqa_ann_path  = os.path.join(script_dir, '..', 'gqa', 'questions',
                             'v2_mscoco_train2014_annotations.json')
vqa_ques_path = os.path.join(script_dir, '..', 'gqa', 'questions',
                             'v2_OpenEnded_mscoco_train2014_questions.json')

with open(vqa_ann_path,  encoding='utf-8') as f:
    vqa_ann = json.load(f)
with open(vqa_ques_path, encoding='utf-8') as f:
    vqa_ques = json.load(f)

# Build question_id → question text map
qid_to_text = {q['question_id']: q['question']
               for q in vqa_ques['questions']}

numeric_answers = {'0','1','2','3','4','5','6','7','8','9','10',
                   'zero','one','two','three','four','five',
                   'six','seven','eight','nine','ten'}

counting = []
for ann in vqa_ann['annotations']:
    ans = ann['multiple_choice_answer'].lower().strip()
    if ans in numeric_answers:
        qtext = qid_to_text.get(ann['question_id'], '')
        if 'how many' in qtext.lower():
            counting.append({
                'id':       str(ann['question_id']),
                'question': qtext,
                'answer':   ans,
                'category': 'counting'
            })
    if len(counting) >= 100:
        break

# ── SAVE ──────────────────────────────────────────────────
cash = spatial[:100] + counting[:100] + attribute[:100]
out = os.path.join(script_dir, 'cash_v1_partial.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(cash, f, indent=2)

print(f"Spatial:   {len(spatial[:100])}")
print(f"Counting:  {len(counting[:100])}")
print(f"Attribute: {len(attribute[:100])}")
print(f"Total:     {len(cash)}")
print(f"\nSample counting: {counting[0]}")