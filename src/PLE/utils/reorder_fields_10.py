#!/usr/bin/env python3
"""
Reorder fields in JSONL records to a target sequence (default: persona, prompt, chosen, rejected).

This script writes a new JSONL file where each record contains only the target
fields in the specified order. Missing fields are reported.
"""

import argparse
import json
from pathlib import Path
from typing import Sequence


def reorder_fields_in_file(input_path: Path, output_path: Path, target_fields: Sequence[str]):
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    total_records = 0
    processed_records = 0
    error_records = 0

    with input_path.open('r', encoding='utf-8') as infile, output_path.open('w', encoding='utf-8') as outfile:
        for line_num, line in enumerate(infile, 1):
            s = line.strip()
            if not s:
                continue
            total_records += 1
            try:
                record = json.loads(s)
            except json.JSONDecodeError as e:
                print(f"JSON parse error at line {line_num}: {e}")
                error_records += 1
                continue

            new_record = {}
            for field in target_fields:
                if field in record:
                    new_record[field] = record[field]
                else:
                    print(f"Warning: line {line_num} missing field '{field}'")

            outfile.write(json.dumps(new_record, ensure_ascii=False) + '\n')
            processed_records += 1

    print(f"Total records: {total_records}")
    print(f"Processed records: {processed_records}")
    print(f"Error records: {error_records}")


def main():
    parser = argparse.ArgumentParser(description='Reorder fields in JSONL records')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Output JSONL file')
    parser.add_argument('-f', '--fields', type=str, default='persona,prompt,chosen,rejected', help='Comma-separated target fields in order')
    args = parser.parse_args()

    fields = [f.strip() for f in args.fields.split(',') if f.strip()]
    reorder_fields_in_file(args.input, args.output, fields)


if __name__ == '__main__':
    main()