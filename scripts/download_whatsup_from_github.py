import os, json, requests

# URLs for the original WhatsUp dataset repository
BASE_URL = "https://raw.githubusercontent.com/amitakamath/whatsup_vlms/main/whats_up_vlms"
TEST_JSON_URL = f"{BASE_URL}/test.json"

def download_json(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def main():
    os.makedirs('datasets/whatsup', exist_ok=True)
    data = download_json(TEST_JSON_URL)
    # Ensure it's a list of entries; if dict with 'data' key, adjust
    if isinstance(data, dict) and 'data' in data:
        data = data['data']
    # Take first 100 items and keep needed fields
    sample = []
    for i, item in enumerate(data):
        if i >= 100:
            break
        # Expected keys: "image_url", "question", "label"
        sample.append({
            "question_id": i,
            "image_url": item.get('image_url', item.get('url', '')),
            "question": item.get('question', item.get('caption', '')),
            "label": item.get('label', 'A')
        })
    out_path = os.path.join('datasets', 'whatsup', 'whatsup_sample_100.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2)
    print('Done:', len(sample), 'questions saved to', out_path)

if __name__ == '__main__':
    main()
