import os
import json
from datasets import load_dataset

# Ensure output directory exists
output_dir = os.path.join('datasets', 'whatsup')
os.makedirs(output_dir, exist_ok=True)

# Load the WhatsUp dataset (Mayfull/whats_up_vlms) test split
print('Loading WhatsUp dataset...')
ds = load_dataset('Mayfull/whats_up_vlms', split='test')
print(f'Dataset size: {len(ds)}')

# Build a sample of the first 100 entries
sample = []
for i, item in enumerate(ds):
    if i >= 100:
        break
    # The dataset fields may vary; we fallback to generic keys
    image_url = item.get('image_url', item.get('url', ''))
    question = item.get('question', item.get('caption_main', ''))
    label = item.get('label', 'A')
    sample.append({
        'question_id': i,
        'image_url': image_url,
        'question': question,
        'label': label,
    })

out_path = os.path.join(output_dir, 'whatsup_sample_100.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(sample, f, indent=2, ensure_ascii=False)
print('Done:', len(sample), 'questions saved to', out_path)
