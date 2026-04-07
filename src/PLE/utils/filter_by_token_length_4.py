#!/usr/bin/env python3
# Filter JSONL records by token length
import argparse
import json
from pathlib import Path

try:
    import tiktoken
except Exception:
    tiktoken = None

def get_token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    if tiktoken:
        try:
            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback: approximate token count using words
    return len(text.split())

def filter_by_token_length(input_path: Path, output_path: Path, max_tokens: int, encoding_name: str):
    total_lines = 0
    kept_lines = 0
    discarded_too_long = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open('r', encoding='utf-8') as f_in, output_path.open('w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON at line {total_lines}: {e}")
                continue

            text = data.get('text', '')
            tokens = get_token_count(text, encoding_name)
            if tokens <= max_tokens:
                json.dump(data, f_out, ensure_ascii=False)
                f_out.write('\n')
                kept_lines += 1
            else:
                discarded_too_long += 1

    print(f"Total lines processed: {total_lines}")
    print(f"Kept lines: {kept_lines}")
    print(f"Discarded lines: {total_lines - kept_lines}")
    print(f"Discarded due to token > {max_tokens}: {discarded_too_long}")

def main():
    parser = argparse.ArgumentParser(description='Filter JSONL by token length')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Output JSONL file')
    parser.add_argument('-m', '--max-tokens', type=int, default=120, help='Maximum allowed token count')
    parser.add_argument('-e', '--encoding', type=str, default='cl100k_base', help='Token encoding name for tiktoken')
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file {args.input} not found")
        return
    
    filter_by_token_length(args.input, args.output, args.max_tokens, args.encoding)

if __name__ == '__main__':
    main()
