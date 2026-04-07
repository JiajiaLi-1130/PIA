import json
import os

def filter_is_safe(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        try:
            data = json.loads(line.strip())
            # keep rows where safety_assessment.is_safe is False
            if not data.get('safety_assessment', {}).get('is_safe', False):
                filtered_lines.append(line)
        except json.JSONDecodeError:
            # skip invalid JSON lines
            continue

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)

    print("Filtering complete")
    print(f"Original lines: {len(lines)}")
    print(f"Filtered lines: {len(filtered_lines)}")
    print(f"Removed lines: {len(lines) - len(filtered_lines)}")

if __name__ == "__main__":
    # redacted relative paths
    input_file = "data/train/train1/dpo_persona.jsonl"
    output_file = "data/train/train1/dpo_persona-unsafe.jsonl"

    filter_is_safe(input_file, output_file)
    print(f"Filtered file saved: {output_file}")