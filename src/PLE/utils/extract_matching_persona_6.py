import argparse
import json
from pathlib import Path


def load_elite_personas(elite_path: Path):
    personas = set()
    with elite_path.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            text = obj.get('text')
            if text:
                personas.add(text)
    return personas


def extract_matching_rows(merged_path: Path, elite_set: set, output_path: Path):
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open('r', encoding='utf-8') as f_in, output_path.open('w', encoding='utf-8') as f_out:
        for line in f_in:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            persona = obj.get('persona')
            if persona and persona in elite_set:
                f_out.write(line)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description='Extract rows whose persona matches an elite list')
    parser.add_argument('-e', '--elite', type=Path, required=True, help='JSONL file with elite personas (field: text)')
    parser.add_argument('-m', '--merged', type=Path, required=True, help='Merged JSONL file to scan')
    parser.add_argument('-o', '--output', type=Path, default=None, help='Output JSONL file (default: <merged>_matching.jsonl)')
    args = parser.parse_args()

    if not args.elite.exists():
        print(f"Error: elite file {args.elite} not found")
        return
    if not args.merged.exists():
        print(f"Error: merged file {args.merged} not found")
        return

    elite_set = load_elite_personas(args.elite)
    print(f"Loaded {len(elite_set)} unique elite personas")

    output_path = args.output or args.merged.with_name(args.merged.stem + '_matching.jsonl')
    matched = extract_matching_rows(args.merged, elite_set, output_path)
    print(f"Extracted {matched} matching rows")
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
