#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract lines from a harm dataset and attach a random persona to each line.

This script reads a JSONL file of potentially harmful examples and a JSONL
file of persona texts, attaches a randomly selected persona text to each
selected harm example, and writes the combined results back to a JSONL file.
"""

import json
import random
import os
import argparse


def extract_and_combine_data(harm_file, persona_file, output_file, start_line=12050, num_lines=8000):
    """Extract data from `harm_file` and add a random `persona` field to each entry.

    Parameters:
    - harm_file: path to the source harm JSONL file
    - persona_file: path to the persona JSONL file (each line should contain a JSON object with a 'persona' field)
    - output_file: path to write the combined JSONL output
    - start_line: 0-based index where extraction starts
    - num_lines: number of lines to extract starting from `start_line`
    """
    print("Starting extraction and persona assignment...")

    # Check files exist
    if not os.path.exists(harm_file):
        print(f"Error: harm file not found: {harm_file}")
        return []

    if not os.path.exists(persona_file):
        print(f"Error: persona file not found: {persona_file}")
        return []

    combined_data = []

    try:
        # Read all persona texts first
        print("Reading persona file...")
        persona_texts = []
        with open(persona_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            print(f"Total persona lines: {len(all_lines)}")

            for idx, line in enumerate(all_lines):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'text' in data:
                            persona_texts.append(data['text'])
                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid persona JSON line {idx+1}: {e}")

        print(f"Loaded {len(persona_texts)} persona texts")

        # Read harm data and attach random persona
        print("Processing harm file...")
        with open(harm_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Total harm lines: {len(lines)}")

            end_line = start_line + num_lines
            actual_end = min(end_line, len(lines))

            print(f"Extracting lines: {start_line+1} to {actual_end}")

            for i in range(start_line, actual_end):
                line = lines[i].strip()
                if line:
                    try:
                        harm_data = json.loads(line)

                        # Randomly choose a persona text and attach it
                        if persona_texts:
                            random_persona = random.choice(persona_texts)
                            harm_data['persona'] = random_persona

                        combined_data.append(harm_data)

                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid harm JSON line {i+1}: {e}")

        print(f"Processed {len(combined_data)} harm records")

        # Save combined data
        with open(output_file, 'w', encoding='utf-8') as f:
            for data in combined_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        print(f"Saved combined data to {output_file}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

    return combined_data


def main():
    """CLI entry point. Uses environment variables or CLI args to avoid embedding absolute paths."""
    parser = argparse.ArgumentParser(description="Attach random persona texts to harm dataset lines.")
    parser.add_argument('--harm-file', default=os.getenv('HARM_FILE', 'data/harm/PKU-SafeRLHF-Train-unsafe.jsonl'), help='Path to harm JSONL file')
    parser.add_argument('--persona-file', default=os.getenv('PERSONA_FILE', 'result/attack/attack1/training.jsonl'), help='Path to persona JSONL file')
    parser.add_argument('--output-file', default=os.getenv('OUTPUT_FILE', 'data/train/train1/dpo_persona.jsonl'), help='Output JSONL file')
    parser.add_argument('--start-line', type=int, default=int(os.getenv('START_LINE', '12050')), help='0-based start line index')
    parser.add_argument('--num-lines', type=int, default=int(os.getenv('NUM_LINES', '8000')), help='Number of lines to extract')
    parser.add_argument('--seed', type=int, default=42, help='Optional random seed for reproducibility')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    combined_data = extract_and_combine_data(
        args.harm_file,
        args.persona_file,
        args.output_file,
        start_line=args.start_line,
        num_lines=args.num_lines,
    )

    print(f"Output file {args.output_file}: {len(combined_data)} records (each contains harm data and a random persona)")


if __name__ == "__main__":
    main()