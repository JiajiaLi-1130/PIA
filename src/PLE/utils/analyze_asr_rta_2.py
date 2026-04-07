#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR and RtA distribution analyzer for JSONL data.

This script reads a JSONL file, extracts numeric `ASR` and `RtA` values,
plots their histograms, annotates counts per bin, and prints basic stats.

Usage:
  python analyze_asr_rta_2.py --input data/merged_gen_nodes.jsonl --output asr_rta_histogram.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def read_values(input_path: Path):
    asr_values = []
    rta_values = []
    with input_path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: invalid JSON at line {line_num}")
                continue

            asr = data.get('ASR')
            rta = data.get('RtA')
            if isinstance(asr, (int, float)):
                asr_values.append(float(asr))
            if isinstance(rta, (int, float)):
                rta_values.append(float(rta))

    return asr_values, rta_values


def plot_histograms(asr_values, rta_values, output_path: Path, bins=20, font=None):
    bins_arr = np.linspace(0, 1, bins + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ASR
    patches1 = ax1.hist(asr_values, bins=bins_arr, alpha=0.7, color='skyblue', edgecolor='black')[2]
    ax1.set_title('ASR Distribution', fontsize=14)
    ax1.set_xlabel('ASR', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.linspace(0, 1, 11))
    for patch in patches1:
        h = patch.get_height()
        if h > 0:
            ax1.text(patch.get_x() + patch.get_width() / 2., h + 1, f'{int(h)}', ha='center', va='bottom', fontsize=9)

    # RtA
    patches2 = ax2.hist(rta_values, bins=bins_arr, alpha=0.7, color='lightgreen', edgecolor='black')[2]
    ax2.set_title('RtA Distribution', fontsize=14)
    ax2.set_xlabel('RtA', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(np.linspace(0, 1, 11))
    for patch in patches2:
        h = patch.get_height()
        if h > 0:
            ax2.text(patch.get_x() + patch.get_width() / 2., h + 1, f'{int(h)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_stats(asr_values, rta_values):
    print('\nStatistics:')
    if asr_values:
        print(f"ASR - min: {min(asr_values):.4f}, max: {max(asr_values):.4f}, mean: {np.mean(asr_values):.4f}")
    else:
        print('ASR - no numeric values found')

    if rta_values:
        print(f"RtA - min: {min(rta_values):.4f}, max: {max(rta_values):.4f}, mean: {np.mean(rta_values):.4f}")
    else:
        print('RtA - no numeric values found')


def main():
    parser = argparse.ArgumentParser(description='Analyze ASR and RtA distributions from a JSONL file')
    parser.add_argument('--input', '-i', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', type=Path, default=Path('asr_rta_histogram.png'), help='Output image path')
    parser.add_argument('--bins', type=int, default=20, help='Number of histogram bins')
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        print(f"Error: input file {input_path} not found")
        return

    asr_values, rta_values = read_values(input_path)
    print(f"Read {len(asr_values)} ASR values and {len(rta_values)} RtA values")

    if not asr_values and not rta_values:
        print('No numeric ASR or RtA values to plot or summarize.')
        return

    plot_histograms(asr_values, rta_values, args.output, bins=args.bins)
    print(f"Plot saved to: {args.output}")
    print_stats(asr_values, rta_values)


if __name__ == '__main__':
    main()
