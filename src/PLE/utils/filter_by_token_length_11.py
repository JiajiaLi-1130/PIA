#!/usr/bin/env python3
"""
Filter JSONL records by token length for `prompt`, `chosen`, and `rejected` fields.

Usage example:
  python filter_by_token_length_11.py -i unsafe_rows_reordered.jsonl -o filtered.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

try:
    import tiktoken
except Exception:
    tiktoken = None


def _get_encoder(name: str = 'cl100k_base'):
    if tiktoken:
        try:
            return tiktoken.get_encoding(name)
        except Exception:
            return None
    return None


def filter_by_token_length(input_path: Path, output_path: Path, max_prompt: int = 512, max_chosen: int = 1536, max_rejected: int = 1536, encoding_name: str = 'cl100k_base') -> Tuple[int,int,dict]:
    encoder = _get_encoder(encoding_name)

    total_lines = 0
    kept_lines = 0
    discarded_prompt = 0
    discarded_chosen = 0
    discarded_rejected = 0
    discarded_multiple = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open('r', encoding='utf-8') as f_in, output_path.open('w', encoding='utf-8') as f_out:
        for line in f_in:
            s = line.strip()
            if not s:
                continue
            total_lines += 1
            try:
                data = json.loads(s)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON at line {total_lines}: {e}")
                continue

            prompt = data.get('prompt', '') or ''
            chosen = data.get('chosen', '') or ''
            rejected = data.get('rejected', '') or ''

            def tok_len(text: str) -> int:
                if encoder:
                    try:
                        return len(encoder.encode(text))
                    except Exception:
                        pass
                return len(text.split())

            prompt_toks = tok_len(prompt)
            chosen_toks = tok_len(chosen)
            rejected_toks = tok_len(rejected)

            prompt_ok = prompt_toks < max_prompt
            chosen_ok = chosen_toks < max_chosen
            rejected_ok = rejected_toks < max_rejected

            if prompt_ok and chosen_ok and rejected_ok:
                json.dump(data, f_out, ensure_ascii=False)
                f_out.write('\n')
                kept_lines += 1
            else:
                # categorize discard reason
                reasons = [not prompt_ok, not chosen_ok, not rejected_ok]
                if sum(reasons) > 1:
                    discarded_multiple += 1
                elif not prompt_ok:
                    discarded_prompt += 1
                elif not chosen_ok:
                    discarded_chosen += 1
                elif not rejected_ok:
                    discarded_rejected += 1

    stats = {
        'total': total_lines,
        'kept': kept_lines,
        'discarded': total_lines - kept_lines,
        'discarded_prompt': discarded_prompt,
        'discarded_chosen': discarded_chosen,
        'discarded_rejected': discarded_rejected,
        'discarded_multiple': discarded_multiple,
    }

    print(f"Total lines processed: {total_lines}")
    print(f"Kept lines: {kept_lines}")
    print(f"Discarded lines: {total_lines - kept_lines}")
    print(f"  - prompt >= {max_prompt}: {discarded_prompt}")
    print(f"  - chosen >= {max_chosen}: {discarded_chosen}")
    print(f"  - rejected >= {max_rejected}: {discarded_rejected}")
    print(f"  - multiple reasons: {discarded_multiple}")

    return total_lines, kept_lines, stats


def main():
    parser = argparse.ArgumentParser(description='Filter JSONL by token length for prompt/chosen/rejected')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Output JSONL file')
    parser.add_argument('--max-prompt', type=int, default=512, help='Max tokens for prompt (default: 512)')
    parser.add_argument('--max-chosen', type=int, default=1536, help='Max tokens for chosen (default: 1536)')
    parser.add_argument('--max-rejected', type=int, default=1536, help='Max tokens for rejected (default: 1536)')
    parser.add_argument('--encoding', type=str, default='cl100k_base', help='tiktoken encoding name (default: cl100k_base)')
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}")
        return

    filter_by_token_length(args.input, args.output, max_prompt=args.max_prompt, max_chosen=args.max_chosen, max_rejected=args.max_rejected, encoding_name=args.encoding)


if __name__ == '__main__':
    main()
