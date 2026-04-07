#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Randomly sample by ratio from two JSONL files and merge results.
Intended ratio example: or-bench-80k : databricks-dolly
"""

import json
import random
import os

def random_sample_by_ratio(or_bench_file, dolly_file, output_file):
    """Sample from two files by ratio and save merged data.

    Returns the merged list of sampled records.
    """

    # Ensure source files exist
    if not os.path.exists(or_bench_file):
        print(f"Error: file not found: {or_bench_file}")
        return []

    if not os.path.exists(dolly_file):
        print(f"Error: file not found: {dolly_file}")
        return []

    # Sampling counts (example configuration)
    total_target = 5000
    or_bench_ratio = 5
    dolly_ratio = 5

    or_bench_count = int(total_target * or_bench_ratio / (or_bench_ratio + dolly_ratio))
    dolly_count = total_target - or_bench_count

    merged_data = []

    try:
        # Sample from or-bench
        print("\nSampling from or-bench-80k...")
        with open(or_bench_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            if len(all_lines) < or_bench_count:
                print(f"Warning: file has fewer lines than target, taking all {len(all_lines)} lines")
                selected_indices = list(range(len(all_lines)))
            else:
                selected_indices = random.sample(range(len(all_lines)), or_bench_count)

            valid_count = 0
            for idx in selected_indices:
                line = all_lines[idx].strip()
                if line:
                    try:
                        data = json.loads(line)
                        data['source'] = 'or-bench-80k'
                        merged_data.append(data)
                        valid_count += 1
                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid JSON line {idx+1}: {e}")

            print(f"Successfully sampled {valid_count} records from or-bench-80k")

        # Sample from databricks-dolly
        print("Sampling from databricks-dolly...")
        with open(dolly_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

            if len(all_lines) < dolly_count:
                print(f"Warning: file has fewer lines than target, taking all {len(all_lines)} lines")
                selected_indices = list(range(len(all_lines)))
            else:
                selected_indices = random.sample(range(len(all_lines)), dolly_count)

            valid_count = 0
            for idx in selected_indices:
                line = all_lines[idx].strip()
                if line:
                    try:
                        data = json.loads(line)
                        data['source'] = 'databricks-dolly'
                        merged_data.append(data)
                        valid_count += 1
                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid JSON line {idx+1}: {e}")

            print(f"Successfully sampled {valid_count} records from databricks-dolly")

        # Shuffle and save
        random.shuffle(merged_data)
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for data in merged_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        print(f"\nMerged data saved to {output_file}")

    except Exception as e:
        print(f"Error while processing data: {e}")

    return merged_data


def main():
    """Main runner."""

    random.seed(42)

    # redacted relative paths
    or_bench_file = os.path.join("data", "train", "or-bench-80k.jsonl")
    dolly_file = os.path.join("data", "train", "databricks-dolly.jsonl")
    output_file = os.path.join("data", "train", "sft_persona_original.jsonl")

    merged_data = random_sample_by_ratio(or_bench_file, dolly_file, output_file)

    print(f"Output file {output_file}: {len(merged_data)} records")


if __name__ == "__main__":
    main()