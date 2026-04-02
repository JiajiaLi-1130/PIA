import json
import os

# Redacted relative paths
original_file = os.path.join("data", "train", "train1", "sft_persona_with_persona.jsonl")
target_file = os.path.join("data", "train", "train1", "sft_persona_fixed1.jsonl")
output_file = os.path.join("data", "train", "train1", "sft_persona_fixed2.jsonl")

prompt_to_response = {}
print("Reading original file...")
with open(original_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            prompt = data.get("prompt")
            response = data.get("response")
            if prompt and response:
                prompt_to_response[prompt] = response
        except json.JSONDecodeError:
            print(f"Error parsing original file line: {line}")
            continue

print(f"Loaded {len(prompt_to_response)} records from original file")

fixed_count = 0
total_lines = 0
print("Processing target file...")
os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
with open(target_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            prompt = data.get("prompt")
            metadata = data.get("metadata", {})
            output_tokens = metadata.get("output_tokens", 1536)

            if (not output_tokens or output_tokens == 1536) and prompt in prompt_to_response:
                data["response"] = prompt_to_response[prompt]
                fixed_count += 1
                print(f"Fixed: {prompt[:50]}...")

            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')
            total_lines += 1
        except json.JSONDecodeError:
            print(f"Error parsing target file line: {line}")
            outfile.write(line)
            outfile.write('\n')
            total_lines += 1
            continue

print("\nProcessing completed!")
print(f"Total lines processed: {total_lines}")
print(f"Fixed lines: {fixed_count}")
print(f"Output file: {output_file}")