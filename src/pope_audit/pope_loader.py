# Why: feeds POPE questions to the model and collects answers for evaluation
import json, os

def load_pope_split(folder_path: str, split: str) -> list:
    """
    Load one POPE split.
    split: 'adversarial', 'popular', or 'random'
    Returns list of dicts with keys: image, text, label
    """
    # Find the file matching the split name
    for fname in os.listdir(folder_path):
        if split in fname.lower() and fname.endswith('.json'):
            fpath = os.path.join(folder_path, fname)
            data = []
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            print(f"Loaded {len(data)} entries from {fname}")
            return data
    raise FileNotFoundError(f"No file found for split '{split}' in {folder_path}")

def get_ground_truths(data: list) -> list:
    """Extract ground truth labels. Handles 'label' or 'answer' key."""
    labels = []
    for item in data:
        if 'label' in item:
            labels.append(item['label'].lower().strip())
        elif 'answer' in item:
            labels.append(item['answer'].lower().strip())
        else:
            raise KeyError(f"No label/answer key in: {item}")
    return labels

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # POPE split files in the original line-delimited (JSONL) format.
    # Defaults to the repo's bundled datasets/pope folder so the demo
    # runs on any machine; pass a folder path as the first CLI argument
    # to point at a different POPE export.
    default_folder = Path(__file__).resolve().parents[2] / "datasets" / "pope"
    pope_folder = sys.argv[1] if len(sys.argv) > 1 else str(default_folder)

    for split in ['adversarial', 'popular', 'random']:
        try:
            data = load_pope_split(pope_folder, split)
            labels = get_ground_truths(data)
            yes_count = labels.count('yes')
            no_count  = labels.count('no')
            print(f"{split}: {len(data)} questions | yes={yes_count} no={no_count}")
            print(f"  Sample Q: {data[0]['text']}")
            print(f"  Sample A: {labels[0]}\n")
        except FileNotFoundError as e:
            print(f"Missing: {e}")
