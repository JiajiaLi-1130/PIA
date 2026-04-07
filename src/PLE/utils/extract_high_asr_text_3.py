#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract texts with high ASR scores from a JSONL file.

Reads an input JSONL file, selects records where `ASR` >= threshold,
and writes selected records as JSONL with `text` and `ASR` fields.

Example:
  python extract_high_asr_text_3.py -i data/merged_gen_nodes.jsonl -o high_asr.jsonl -t 0.6
"""

import argparse
import json
from pathlib import Path


def extract_high_asr(input_path: Path, output_path: Path, threshold: float) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open('r', encoding='utf-8') as f_in, output_path.open('w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: invalid JSON at line {line_num}")
                continue

            asr = data.get('ASR')
            text = data.get('text')
            if isinstance(asr, (int, float)) and text is not None and asr >= threshold:
                json.dump({'text': text, 'ASR': float(asr)}, f_out, ensure_ascii=False)
                f_out.write('\n')
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description='Extract texts with ASR >= threshold from a JSONL file')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Output JSONL file')
    parser.add_argument('-t', '--threshold', type=float, default=0.6, help='ASR threshold (default: 0.6)')
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file {args.input} not found")
        return

    count = extract_high_asr(args.input, args.output, args.threshold)
    print(f"Extracted {count} record(s) with ASR >= {args.threshold}")
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()
