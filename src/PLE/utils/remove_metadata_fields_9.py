#!/usr/bin/env python3
"""
Remove specified metadata fields (e.g., `safety_assessment`) from a JSONL file.

This utility overwrites the input file after removing the listed fields.
It optionally verifies the result to ensure fields were removed.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable


def remove_metadata_fields(input_path: Path, fields: Iterable[str], verify: bool = True):
    temp_path = input_path.with_suffix(input_path.suffix + '.tmp') if input_path.suffix else Path(str(input_path) + '.tmp')
    processed = 0
    removed = 0

    with input_path.open('r', encoding='utf-8') as fin, temp_path.open('w', encoding='utf-8') as fout:
        for line_num, line in enumerate(fin, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                # keep original line if not valid JSON
                fout.write(line)
                continue

            processed += 1
            for f in fields:
                if f in obj:
                    del obj[f]
                    removed += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

    # atomically replace
    temp_path.replace(input_path)

    print(f"Processed {processed} JSON lines")
    print(f"Removed {removed} occurrences of fields: {', '.join(fields)}")

    if verify:
        remaining = 0
        with input_path.open('r', encoding='utf-8') as f:
            for line in f:
                if any(fld in line for fld in fields):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if any(fld in obj for fld in fields):
                        remaining += 1

        if remaining:
            print(f"Warning: {remaining} lines still contain the removed fields")
        else:
            print("Verification successful: removed fields not found in file")


def main():
    parser = argparse.ArgumentParser(description='Remove metadata fields from a JSONL file')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('-f', '--fields', type=str, default='safety_assessment', help='Comma-separated field names to remove')
    parser.add_argument('--no-verify', action='store_true', help='Skip verification after removal')
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}")
        return

    fields = [s.strip() for s in args.fields.split(',') if s.strip()]
    remove_metadata_fields(args.input, fields, verify=not args.no_verify)


if __name__ == '__main__':
    main()
