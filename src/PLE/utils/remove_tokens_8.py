#!/usr/bin/env python3
"""
Remove lines that contain a specific output token count in a JSONL file.

This script removes lines containing the substring
`"output_tokens": <value>` and overwrites the input file.

Example:
  python remove_tokens_8.py -i unsafe_rows.jsonl -v 1536
"""

import argparse
from pathlib import Path


def remove_lines_with_output_tokens(file_path: Path, token_value: int) -> tuple:
    """Remove lines containing '"output_tokens": {token_value}' from the file.

    Returns a tuple: (original_count, removed_count, remaining_count)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open('r', encoding='utf-8') as f:
        lines = f.readlines()

    needle = f'"output_tokens": {token_value}'
    filtered = [ln for ln in lines if needle not in ln]

    removed = len(lines) - len(filtered)

    with file_path.open('w', encoding='utf-8') as f:
        f.writelines(filtered)

    return len(lines), removed, len(filtered)


def main():
    parser = argparse.ArgumentParser(description='Remove lines with a specific output_tokens value')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input JSONL file (will be overwritten)')
    parser.add_argument('-v', '--value', type=int, default=1536, help='Token value to remove (default: 1536)')
    args = parser.parse_args()

    try:
        original, removed, remaining = remove_lines_with_output_tokens(args.input, args.value)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Done.")
    print(f"Original lines: {original}")
    print(f"Removed lines: {removed}")
    print(f"Remaining lines: {remaining}")


if __name__ == '__main__':
    main()