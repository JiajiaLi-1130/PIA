#!/usr/bin/env python3
"""
Analyze token counts for `text` fields in a JSONL file.

Calculates min/max/avg/total token counts and reports the longest text.
Supports `tiktoken` if installed, otherwise falls back to a word-based approximation.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    import tqdm
except Exception:
    tqdm = None


def _get_encoder(name: str = 'cl100k_base'):
    if tiktoken:
        try:
            return tiktoken.get_encoding(name)
        except Exception:
            return None
    return None


def token_count_for_text(text: str, encoder) -> int:
    if encoder:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    # fallback: approximate by whitespace-separated tokens
    return len(text.split())


def check_token_count(file_path: Path, encoding_name: str = 'cl100k_base') -> Dict:
    encoder = _get_encoder(encoding_name)

    counts = []
    total_tokens = 0
    max_tokens = 0
    max_text = ''

    with file_path.open('r', encoding='utf-8') as f:
        lines = f.readlines()

    iterator = lines
    if tqdm:
        iterator = tqdm.tqdm(lines, desc='Processing lines')

    for line in iterator:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = obj.get('text', '')
        cnt = token_count_for_text(text, encoder)
        counts.append(cnt)
        total_tokens += cnt
        if cnt > max_tokens:
            max_tokens = cnt
            max_text = text

    avg = total_tokens / len(counts) if counts else 0
    minimum = min(counts) if counts else 0

    print('\n=== Token Count Analysis ===')
    print(f'Total lines processed: {len(counts)}')
    print(f'Minimum tokens: {minimum}')
    print(f'Maximum tokens: {max_tokens}')
    print(f'Average tokens: {avg:.2f}')
    print(f'Total tokens: {total_tokens}')
    print('\nLongest text ({max_tokens} tokens):')
    print((max_text[:200] + '...') if len(max_text) > 200 else max_text)

    return {
        'min': minimum,
        'max': max_tokens,
        'avg': avg,
        'total': total_tokens,
        'count': len(counts),
        'longest_text': max_text,
    }


def main():
    parser = argparse.ArgumentParser(description='Check token counts in JSONL file (field: text)')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('-e', '--encoding', type=str, default='cl100k_base', help='tiktoken encoding name (if installed)')
    args = parser.parse_args()

    if not args.input.exists():
        print(f'Error: input file not found: {args.input}')
        return

    check_token_count(args.input, encoding_name=args.encoding)


if __name__ == '__main__':
    main()
