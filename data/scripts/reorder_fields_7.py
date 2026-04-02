import json
import os

def reorder_fields():
    # redacted relative input path
    input_file = os.path.join("data", "train", "train1", "trainging.jsonl")
    temp_file = input_file + ".tmp"

    # desired field order
    field_order = [
        "type",
        "has_persona",
        "persona",
        "prompt",
        "chosen",
        "rejected",
    ]

    print(f"Processing file: {input_file}")
    processed_count = 0

    os.makedirs(os.path.dirname(input_file) or '.', exist_ok=True)
    with open(input_file, 'r', encoding='utf-8') as f_in, open(temp_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                processed_count += 1

                # rebuild JSON object with the specified order first
                reordered_data = {}
                for field in field_order:
                    if field in data:
                        reordered_data[field] = data[field]

                # then append remaining fields
                for field in data:
                    if field not in reordered_data:
                        reordered_data[field] = data[field]

                json.dump(reordered_data, f_out, ensure_ascii=False)
                f_out.write('\n')

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                # keep original line
                f_out.write(line)
                continue

    print(f"Processed {processed_count} lines")

    # replace original file
    os.replace(temp_file, input_file)
    print(f"Updated file: {input_file}")

    # verify result
    print("Verifying result...")
    verified_count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                current_fields = list(data.keys())
                expected_fields = [field for field in field_order if field in data]

                if current_fields[:len(expected_fields)] == expected_fields:
                    verified_count += 1
                else:
                    print(f"Line {line_num} has incorrect field order")
                    print(f"Expected: {expected_fields}")
                    print(f"Actual: {current_fields[:len(expected_fields)]}")
                    break

            except json.JSONDecodeError as e:
                print(f"Error parsing verification line {line_num}: {e}")
                continue

    if verified_count == processed_count:
        print(f"Verification successful: All {verified_count} lines have correct field order")
    else:
        print(f"Verification failed: Only {verified_count} out of {processed_count} lines have correct field order")


if __name__ == "__main__":
    reorder_fields()
