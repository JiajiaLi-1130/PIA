import json
import os

# Default relative path (redacted from absolute paths)
file_path = "data/train/train1/training.jsonl"

def check_jsonl_structure(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        print("File is empty")
        return

    # Use the first line's fields and order as the reference
    first_line = json.loads(lines[0].strip())
    reference_fields = list(first_line.keys())

    print(f"Reference field structure (total {len(reference_fields)} fields):")
    for i, field in enumerate(reference_fields, 1):
        print(f"{i}. {field}")
    print()

    # Check all lines
    issues = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            fields = list(data.keys())

            # Check field count
            if len(fields) != len(reference_fields):
                issues.append(f"Line {i}: field count mismatch (expected {len(reference_fields)}, got {len(fields)})")
                continue

            # Check field names and order
            if fields != reference_fields:
                issues.append(f"Line {i}: field order or names differ")
                continue

        except json.JSONDecodeError as e:
            issues.append(f"Line {i}: JSON decode error - {e}")

    # Output results
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(issue)
        if len(issues) > 10:
            print(f"... and {len(issues) - 10} more issues not shown")
    else:
        print(f"All {len(lines)} lines have consistent structure, with complete fields in correct order.")

if __name__ == "__main__":
    check_jsonl_structure(file_path)