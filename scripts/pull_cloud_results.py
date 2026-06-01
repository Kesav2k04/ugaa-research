"""
pull_cloud_results.py
=====================

After downloading the cloud-notebook outputs from Kaggle / Colab,
point this script at the folder you saved them to. It categorises each
JSON by filename pattern, copies it into the right experiments/
sub-directory, and prints a one-line summary per file. Designed to be
idempotent: re-running on the same source folder produces the same
state.

Recognised patterns:

  <model>_pope_<split>_<n>q_<prompt_mode>_token_audit.json
        -> experiments/multi_model/

  ugaa_full_<split>_<n>q_diagnostic.json
        -> experiments/

  latency_microbench.json
        -> experiments/

  llava-hf_llava-v1.6-mistral-7b-hf_pope_<split>_<n>q_<mode>_token_audit.json
        -> experiments/cross_model/

Anything else is reported and skipped.

Usage:

    python scripts/pull_cloud_results.py --dir D:\\cloud_results\\2026-05-30
    python scripts/pull_cloud_results.py --dir ./downloads --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"

MULTI_MODEL_RE = re.compile(
    r"^(?P<model>[A-Za-z0-9_]+)_pope_(?P<split>adversarial|popular|random)_"
    r"(?P<n>\d+)q_(?P<mode>[A-Za-z_]+)_token_audit\.json$"
)
DIAG_RE = re.compile(
    r"^ugaa_full_(?P<split>adversarial|popular|random)_(?P<n>\d+)q_diagnostic\.json$"
)
CROSS_MODEL_RE = re.compile(
    r"^llava-hf_llava-v1\.6-mistral-7b-hf_pope_(?P<split>adversarial|popular|random)_"
    r"(?P<n>\d+)q_(?P<mode>[A-Za-z_]+)_token_audit\.json$"
)
LATENCY_NAME = "latency_microbench.json"


def _summarise(json_path: Path) -> str:
    """One-line summary for the printed log. Best-effort, never raises."""
    try:
        with open(json_path) as f:
            d = json.load(f)
    except Exception:
        return "(unreadable)"
    summ = d.get("summary", d)
    bits = []
    if "split" in summ: bits.append(f"split={summ['split']}")
    if "n_valid" in summ: bits.append(f"n_valid={summ['n_valid']}")
    for k in ("eval_dynamic_single", "eval_string_parse"):
        if isinstance(summ.get(k), dict) and "f1" in summ[k]:
            bits.append(f"{k}.f1={summ[k]['f1']}")
    if "f1" in summ: bits.append(f"f1={summ['f1']}")
    if not bits and "two_token" in summ and "string_parse" in summ:
        bits.append(f"latency two-tok median={summ['two_token'].get('median')}")
        bits.append(f"string-parse median={summ['string_parse'].get('median')}")
    return ", ".join(bits)


def classify(name: str):
    """Return (destination_relative_to_REPO_or_None, classification_str)."""
    if CROSS_MODEL_RE.match(name):
        return EXP / "cross_model" / name, "cross_model"
    if MULTI_MODEL_RE.match(name):
        return EXP / "multi_model" / name, "multi_model"
    if DIAG_RE.match(name):
        return EXP / name, "diagnostic_3000q"
    if name == LATENCY_NAME:
        return EXP / name, "latency"
    return None, "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                        help="Folder containing the JSON files downloaded "
                             "from the cloud notebook output panel.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be copied; do not modify anything.")
    args = parser.parse_args()

    src = Path(args.dir)
    if not src.is_dir():
        print(f"ERROR: {src} is not a directory.")
        sys.exit(2)

    files = sorted(src.glob("*.json"))
    if not files:
        print(f"No .json files found in {src}.")
        sys.exit(0)

    print(f"Scanning {len(files)} files in {src}...\n")
    counts = {"multi_model": 0, "cross_model": 0,
              "diagnostic_3000q": 0, "latency": 0, "unknown": 0}
    for f in files:
        dest, kind = classify(f.name)
        counts[kind] += 1
        if dest is None:
            print(f"  [SKIP-unknown] {f.name}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        rel = dest.relative_to(REPO)
        if args.dry_run:
            print(f"  [dry] {f.name}  ->  {rel}")
            continue
        shutil.copy2(f, dest)
        size_kb = f.stat().st_size // 1024
        summary = _summarise(dest)
        print(f"  [{kind:18s}] {f.name}  ->  {rel}  ({size_kb} KB)  {summary}")

    print()
    print("Counts:")
    for k, v in counts.items():
        print(f"  {k:18s} {v}")

    if not args.dry_run:
        print()
        print("Suggested next steps:")
        print("  python analysis/fact_audit.py")
        print("  python analysis/string_parse_equivalence.py")
        for f in files:
            if DIAG_RE.match(f.name):
                rel = (EXP / f.name).relative_to(REPO)
                print(f"  python analysis/diagnostic_stats.py --diagnostic {rel}")


if __name__ == "__main__":
    main()
