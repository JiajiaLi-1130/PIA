import argparse
import json
from pathlib import Path


def check_jsonl_structure(file_path: Path, show_examples: int = 10):
    """Check JSONL records for consistent field names and order.

    The first non-empty line is taken as the reference field order.
    Reports lines that differ in number of fields or field order.
    """
    with file_path.open('r', encoding='utf-8') as f:
        lines = [ln for ln in (l.strip() for l in f) if ln]

    if not lines:
        print("File is empty")
        return

    try:
        first_obj = json.loads(lines[0])
    except json.JSONDecodeError as e:
        print(f"Error: first line is not valid JSON: {e}")
        return

    reference_fields = list(first_obj.keys())
    print(f"Reference field structure ({len(reference_fields)} fields):")
    for i, field in enumerate(reference_fields, 1):
        print(f"{i}. {field}")

    issues = []
    for idx, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            issues.append((idx, f"JSON parse error: {e}"))
            continue

        fields = list(obj.keys())
        if len(fields) != len(reference_fields):
            issues.append((idx, f"field count mismatch (expected {len(reference_fields)}, got {len(fields)})"))
            continue

        if fields != reference_fields:
            issues.append((idx, "field order or names differ"))

    if issues:
        print(f"Found {len(issues)} issue(s). Showing up to {show_examples} examples:")
        for idx, msg in issues[:show_examples]:
            print(f"Line {idx}: {msg}")
        if len(issues) > show_examples:
            print(f"... and {len(issues) - show_examples} more issues not shown")
    else:
        print(f"All {len(lines)} lines have consistent structure and field order.")


def main():
    parser = argparse.ArgumentParser(description='Check JSONL dataset field structure and order')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input JSONL file to check')
    parser.add_argument('-n', '--examples', type=int, default=10, help='Number of issue examples to show')
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: file not found: {args.input}")
        return

    check_jsonl_structure(args.input, show_examples=args.examples)


if __name__ == '__main__':
    main()