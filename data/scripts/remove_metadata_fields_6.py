import json
import os

def remove_rejected_fields():
    # redacted relative input path
    input_file = os.path.join("data", "train", "train1", "training.jsonl")
    temp_file = input_file + ".tmp"

    print(f"Processing file: {input_file}")
    processed_count = 0
    removed_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, open(temp_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                processed_count += 1

                if "rejected" in data:
                    del data["rejected"]
                    removed_count += 1

                json.dump(data, f_out, ensure_ascii=False)
                f_out.write('\n')

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                # keep original line
                f_out.write(line)
                continue

    print(f"Processed {processed_count} lines")
    print(f"Removed 'rejected' field from {removed_count} lines")

    # replace original file with processed temp file
    os.replace(temp_file, input_file)
    print(f"Updated file: {input_file}")

    # verify result
    print("Verifying result...")
    remaining_rejected = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if "rejected" in line:
                try:
                    data = json.loads(line.strip())
                    if "rejected" in data:
                        remaining_rejected += 1
                except json.JSONDecodeError:
                    continue

    if remaining_rejected > 0:
        print(f"Warning: {remaining_rejected} lines still contain 'rejected' field")
    else:
        print("Verification successful: no 'rejected' fields remain")

if __name__ == "__main__":
    remove_rejected_fields()
