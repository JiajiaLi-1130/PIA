#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract a range of lines from PKU-SafeRLHF-Train-unsafe.jsonl and save them to a JSONL file.

"""

import json
import os
import argparse


def extract_range_data(start_line, end_line, input_file=None, output_file=None):
    """Extract a specified line range from the harm JSONL file.

    Parameters:
    - start_line: 0-based start index (inclusive)
    - end_line: end index (exclusive or inclusive depending on caller convention)
    - input_file: path to the source JSONL (if None, uses environment/default)
    - output_file: path to write extracted JSONL (if None, uses environment/default)
    """
    # Resolve default paths from environment if not provided
    if input_file is None:
        input_file = os.getenv('INPUT_FILE', 'data/harm/PKU-SafeRLHF-Train-unsafe.jsonl')
    if output_file is None:
        output_file = os.getenv('OUTPUT_FILE', 'data/train/train1/dpo_unpersona.jsonl')

    print(f"Starting extraction of lines {start_line+1}-{end_line} from {input_file}...")

    if not os.path.exists(input_file):
        print(f"Error: input file not found: {input_file}")
        return []

    extracted_data = []

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            actual_end = min(end_line, len(lines))

            for i in range(start_line, actual_end):
                line = lines[i].strip()
                if line:
                    try:
                        data = json.loads(line)
                        extracted_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid JSON line {i+1}: {e}")

        # Save extracted data
        with open(output_file, 'w', encoding='utf-8') as f:
            for data in extracted_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        print(f"Successfully extracted and saved {len(extracted_data)} records to {output_file}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

    return extracted_data


def main():
    parser = argparse.ArgumentParser(description='Extract a range of lines from a JSONL harm file.')
    parser.add_argument('--start-line', type=int, default=int(os.getenv('START_LINE', '2050')), help='0-based start index')
    parser.add_argument('--end-line', type=int, default=int(os.getenv('END_LINE', '7050')), help='End index (exclusive)')
    parser.add_argument('--input-file', default=os.getenv('INPUT_FILE', 'data/harm/pku-saferlhf.jsonl'), help='Path to source JSONL file')
    parser.add_argument('--output-file', default=os.getenv('OUTPUT_FILE', 'data/train/train1/dpo_unpersona.jsonl'), help='Path to output JSONL file')

    args = parser.parse_args()

    # Extract the specified range
    extracted_data = extract_range_data(args.start_line, args.end_line, input_file=args.input_file, output_file=args.output_file)


if __name__ == "__main__":
    main()