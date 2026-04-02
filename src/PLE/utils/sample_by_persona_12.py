#!/usr/bin/env python3
"""
Sample records evenly by `persona` from a JSONL file.

This script groups records by the `persona` field and performs a balanced
sampling to produce a target number of records. Random seed is configurable
for reproducibility.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict


def set_random_seed(seed: int = 42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def analyze_persona_distribution(input_path: Path) -> Dict[str, List[int]]:
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        return {}

    persona_groups: Dict[str, List[int]] = defaultdict(list)
    total = 0
    errors = 0
    with input_path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                print(f"JSON parse error at line {line_num}: {e}")
                errors += 1
                continue

            persona = obj.get('persona')
            if persona:
                persona_groups[persona].append(line_num - 1)
            else:
                print(f"Warning: line {line_num} missing 'persona' field")

    print(f"Total records: {total}")
    print(f"Error records: {errors}")
    print(f"Unique personas: {len(persona_groups)}")
    return persona_groups


def balanced_sampling(persona_groups: Dict[str, List[int]], target_count: int = 10000) -> List[int]:
    total_available = sum(len(v) for v in persona_groups.values())
    print(f"Starting balanced sampling, target: {target_count}, available: {total_available}")

    persona_sample_counts: Dict[str, int] = {}
    remaining_slots = target_count

    for persona, lines in persona_groups.items():
        proportion = len(lines) / total_available
        count = int(proportion * target_count)
        count = max(1, count)
        persona_sample_counts[persona] = min(count, len(lines))
        remaining_slots -= count

    if remaining_slots > 0:
        remaining_capacities = [(p, len(persona_groups[p]) - persona_sample_counts[p]) for p in persona_sample_counts if len(persona_groups[p]) - persona_sample_counts[p] > 0]
        remaining_capacities.sort(key=lambda x: x[1], reverse=True)
        for persona, _ in remaining_capacities:
            if remaining_slots <= 0:
                break
            persona_sample_counts[persona] += 1
            remaining_slots -= 1

    current_total = sum(persona_sample_counts.values())
    if current_total != target_count:
        if current_total > target_count:
            sorted_personas = sorted(persona_sample_counts.items(), key=lambda x: x[1], reverse=True)
            for persona, _ in sorted_personas:
                if current_total <= target_count:
                    break
                if persona_sample_counts[persona] > 1:
                    persona_sample_counts[persona] -= 1
                    current_total -= 1
        else:
            sorted_personas = sorted(persona_groups.items(), key=lambda x: len(x[1]), reverse=True)
            for persona, _ in sorted_personas:
                if current_total >= target_count:
                    break
                if len(persona_groups[persona]) > persona_sample_counts[persona]:
                    persona_sample_counts[persona] += 1
                    current_total += 1

    print("Final allocation:")
    final_total = 0
    for persona, count in persona_sample_counts.items():
        original = len(persona_groups[persona])
        final_total += count
        display = (persona[:40] + '...') if len(persona) > 40 else persona
        print(f"  {display}: {count}/{original} ({count/original*100:.1f}%)")

    selected_lines: List[int] = []
    for persona, count in persona_sample_counts.items():
        available = persona_groups[persona]
        if len(available) >= count:
            selected = random.sample(available, count)
            selected_lines.extend(selected)

    return selected_lines


def extract_selected_records(input_path: Path, output_path: Path, selected_lines: List[int]):
    selected_set = set(selected_lines)
    total = 0
    extracted = 0
    errors = 0
    with input_path.open('r', encoding='utf-8') as infile, output_path.open('w', encoding='utf-8') as outfile:
        for line_num, line in enumerate(infile, 1):
            s = line.strip()
            if not s:
                continue
            total += 1
            if line_num - 1 in selected_set:
                try:
                    obj = json.loads(s)
                    outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    extracted += 1
                except json.JSONDecodeError as e:
                    print(f"JSON parse error at line {line_num}: {e}")
                    errors += 1

    print("Extraction summary:")
    print(f"Total processed: {total}")
    print(f"Extracted: {extracted}")
    print(f"Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description='Sample records evenly by persona from a JSONL file')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Output JSONL file')
    parser.add_argument('-n', '--num', type=int, default=5000, help='Target number of records to sample')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_random_seed(args.seed)

    persona_groups = analyze_persona_distribution(args.input)
    if not persona_groups:
        print('No persona groups found, aborting')
        return

    selected = balanced_sampling(persona_groups, target_count=args.num)
    extract_selected_records(args.input, args.output, selected)


if __name__ == '__main__':
    main()