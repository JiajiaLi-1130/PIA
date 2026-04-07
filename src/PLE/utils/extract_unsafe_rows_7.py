#!/usr/bin/env python3
"""
Extract rows where safety_assessment.is_safe is False.

Usage:
  python extract_unsafe_rows_7.py -i matching_persona_rows.jsonl -o unsafe_rows.jsonl
"""

import argparse
import json
from pathlib import Path


def extract_unsafe_rows(input_path: Path, output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open('r', encoding='utf-8') as f_in, output_path.open('w', encoding='utf-8') as f_out:
        for line in f_in:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            sa = obj.get('safety_assessment')
            if isinstance(sa, dict) and sa.get('is_safe') is False:
                f_out.write(line)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description='Extract rows with safety_assessment.is_safe == False')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Output JSONL file')
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file {args.input} not found")
        return

    count = extract_unsafe_rows(args.input, args.output)
    print(f"Extracted {count} unsafe rows (safety_assessment.is_safe == False)")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()