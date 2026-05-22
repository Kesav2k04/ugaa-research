# scripts/download_pope_full.py
# Download the full POPE benchmark from HuggingFace (lmms-lab/POPE).
#
# Dataset parquet files embed JPEG bytes directly — no separate image downloads.
# Each split: 3000 questions with embedded COCO val2014 images.
#
# Output:
#   datasets/pope/pope_{split}_full.json      — question/label index
#   datasets/pope/images/{split}/{qid}.jpg    — extracted COCO images
#
# Usage:
#   python scripts/download_pope_full.py [--splits adversarial popular random]
#   python scripts/download_pope_full.py --splits adversarial   (fastest, ~400MB)

import argparse
import json
import os

import pandas as pd
import requests
from io import BytesIO

HF_BASE = "https://huggingface.co/datasets/lmms-lab/POPE/resolve/main/Full"
SPLITS = ["adversarial", "popular", "random"]
OUT_DIR = "datasets/pope"


def download_parquet(split: str) -> pd.DataFrame:
    url = f"{HF_BASE}/{split}-00000-of-00001.parquet"
    print(f"  Fetching: {url}")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    df = pd.read_parquet(BytesIO(r.content))
    print(f"  Rows: {len(df)} | Columns: {list(df.columns)}")
    return df


def extract_image_bytes(cell) -> bytes | None:
    """Extract JPEG bytes from HuggingFace Image feature cell.
    Cell is a dict like {'bytes': b'...', 'path': None}.
    """
    if cell is None:
        return None
    if isinstance(cell, bytes):
        return cell
    if isinstance(cell, dict):
        return cell.get("bytes")
    return None


def save_images_and_build_index(df: pd.DataFrame, split: str) -> list:
    img_dir = os.path.join(OUT_DIR, "images", split)
    os.makedirs(img_dir, exist_ok=True)

    items = []
    for idx, row in df.iterrows():
        qid = int(row.get("question_id", row.get("id", idx + 1)))
        question = str(row.get("question", row.get("text", "")))
        label = str(row.get("answer", row.get("label", ""))).lower()

        img_bytes = extract_image_bytes(row.get("image"))
        local_path = ""
        if img_bytes:
            local_path = os.path.join(img_dir, f"{qid}.jpg")
            if not os.path.exists(local_path):
                with open(local_path, "wb") as f:
                    f.write(img_bytes)

        items.append({
            "question_id": qid,
            "question": question,
            "label": label,
            "local_path": local_path,
            "split": split,
        })

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(df)} ...")

    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    for split in args.splits:
        out_path = os.path.join(OUT_DIR, f"pope_{split}_full.json")
        if os.path.exists(out_path):
            existing = json.load(open(out_path))
            if existing and existing[0].get("local_path"):
                print(f"[{split}] Already exists: {len(existing)} items at {out_path}")
                continue
            print(f"[{split}] Exists but missing local_path — re-downloading.")

        print(f"\n[{split}] Downloading parquet (~100-150MB)...")
        try:
            df = download_parquet(split)
        except Exception as e:
            print(f"[{split}] Download failed: {e}")
            continue

        print(f"[{split}] Saving images and building index...")
        items = save_images_and_build_index(df, split)

        with open(out_path, "w") as f:
            json.dump(items, f, indent=2)

        print(f"[{split}] Saved {len(items)} items to {out_path}")
        ok_images = sum(1 for x in items if x["local_path"] and os.path.exists(x["local_path"]))
        print(f"[{split}] Images saved: {ok_images}/{len(items)}")
        print(f"  Sample: q={items[0]['question']!r} label={items[0]['label']}")

    print("\nDone. Run eval with:")
    print("  python src/run_pope_eval_full.py --split adversarial --samples 500  (1.5h)")
    print("  python src/run_pope_eval_full.py --split adversarial               (overnight)")


if __name__ == "__main__":
    main()
