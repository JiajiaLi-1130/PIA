#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge utility for gen_*_judged.jsonl files.
Usage examples:
  python merge_1.py --source-dir ./result/attack/attack1
  python merge_1.py -s ./attack -o merged.jsonl
"""

import argparse
from pathlib import Path

def find_files(source_dir: Path):
    pattern = "gen_*_judged.jsonl"
    files = sorted(p for p in source_dir.glob(pattern) if "_elite_top35_" not in p.name)
    return files

def merge_files(files, output_file: Path):
    with output_file.open("w", encoding="utf-8") as out_f:
        for f in files:
            with f.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)

def main():
    parser = argparse.ArgumentParser(description="Merge gen_*_judged.jsonl files in a directory")
    parser.add_argument("--source-dir", "-s", type=Path, default=Path("."), help="Directory to search for files")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output file path (default: <source-dir>/merged_gen_judged.jsonl)")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_file = args.output or (source_dir / "merged_gen_judged.jsonl")

    files = find_files(source_dir)
    print(f"Found {len(files)} file(s) to merge.")
    for f in files:
        print(f" - {f.name}")

    if not files:
        print("No matching files found. Nothing to merge.")
        return

    print(f"Merging into {output_file.name}...")
    merge_files(files, output_file)
    size = output_file.stat().st_size
    print(f"Merge completed. Output size: {size} bytes")

if __name__ == "__main__":
    main()
