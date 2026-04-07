#!/usr/bin/env python3
"""
Compute statistics for ASR values in a JSONL file.

Reads a JSONL file, extracts numeric `ASR` values, and prints count,
min, max, mean, median and standard deviation.
"""

import argparse
import json
import sys
from pathlib import Path
import statistics


def read_asr_values(path: Path):
    values = []
    invalid = 0
    try:
        with path.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    print(f"Warning: invalid JSON at line {line_num}")
                    invalid += 1
                    continue

                asr = data.get('ASR')
                if isinstance(asr, (int, float)):
                    values.append(float(asr))
                else:
                    invalid += 1
    except FileNotFoundError:
        print(f"Error: file {path} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    return values, invalid


def compute_stats(values):
    if not values:
        return None
    stats = {
        'count': len(values),
        'min': min(values),
        'max': max(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'std': statistics.pstdev(values),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description='Compute ASR statistics from a JSONL file')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input JSONL file')
    args = parser.parse_args()

    input_path = args.input
    asr_values, invalid = read_asr_values(input_path)
    print(f"Read {len(asr_values)} numeric ASR value(s); {invalid} invalid/missing entries")

    stats = compute_stats(asr_values)
    if not stats:
        print('No valid ASR values found.')
        return

    print(f"Total records: {stats['count']}")
    print(f"ASR range: {stats['min']:.4f} - {stats['max']:.4f}")
    print(f"Mean ASR: {stats['mean']:.4f}")
    print(f"Median ASR: {stats['median']:.4f}")
    print(f"Std deviation: {stats['std']:.4f}")


if __name__ == '__main__':
    main()