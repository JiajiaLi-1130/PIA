import json
import os
from transformers import AutoTokenizer
import tqdm

# Disable proxy environment variables
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['ALL_PROXY'] = ''

# Use a local tokenizer (path redacted for open-source)
model_path = os.path.join("models", "local_tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def check_token_count(file_path):
    # Fields to analyze
    fields = ["prompt", "persona", "chosen", "rejected"]
    stats = {}

    # Initialize stats for each field
    for field in fields:
        stats[field] = {
            "counts": [],
            "max_tokens": 0,
            "max_text": "",
            "total_tokens": 0,
            "min_tokens": float('inf'),
            "max_tokens": 0
        }

    # Read file and process lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for line in tqdm.tqdm(lines, desc="Processing lines"):
            try:
                data = json.loads(line)

                for field in fields:
                    text = data.get(field, "")
                    if isinstance(text, str) and text.strip():
                        # Count tokens (excluding special tokens)
                        token_count = len(tokenizer.encode(text, add_special_tokens=False))

                        # Update stats
                        field_stats = stats[field]
                        field_stats["counts"].append(token_count)
                        field_stats["total_tokens"] += token_count

                        if token_count > field_stats["max_tokens"]:
                            field_stats["max_tokens"] = token_count
                            field_stats["max_text"] = text

                        if token_count < field_stats["min_tokens"]:
                            field_stats["min_tokens"] = token_count

            except json.JSONDecodeError as e:
                print(f"Error decoding line: {e}")
            except Exception as e:
                print(f"Error processing line: {e}")

    # Print results
    print(f"\n=== Multi-field token statistics ===")
    print(f"Total lines processed: {len(lines)}")

    results = {}

    for field in fields:
        field_stats = stats[field]
        counts = field_stats["counts"]

        if counts:
            avg_tokens = field_stats["total_tokens"] / len(counts)
            min_tokens = field_stats["min_tokens"]
            max_tokens = field_stats["max_tokens"]
            count = len(counts)
        else:
            avg_tokens = min_tokens = max_tokens = total_tokens = count = 0

        print(f"\n--- {field.upper()} field statistics ---")
        print(f"Lines with this field: {count}")
        print(f"Min tokens: {min_tokens}")
        print(f"Max tokens: {max_tokens}")
        print(f"Average tokens: {avg_tokens:.2f}")

        results[field] = {
            "min": min_tokens,
            "max": max_tokens,
            "avg": avg_tokens,
            "count": count,
            "longest_text": field_stats["max_text"]
        }

    return results

if __name__ == "__main__":
    file_path = "data/train/train1/training.jsonl"
    check_token_count(file_path)
