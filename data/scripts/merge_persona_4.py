#!/usr/bin/env python3
import json
import random
import os
from typing import List, Dict

def load_persona_data(file_path: str) -> List[Dict]:
    """Load persona records from a JSONL file."""
    personas = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                personas.append(json.loads(line))
    return personas


def load_sft_data(file_path: str) -> List[Dict]:
    """Load SFT training data from a JSONL file."""
    sft_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sft_data.append(json.loads(line))
    return sft_data


def merge_persona_with_sft(sft_data: List[Dict], persona_data: List[Dict], output_path: str):
    """Add a random persona text to each SFT item and write to output JSONL."""
    random.seed(42)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sft_item in sft_data:
            selected_persona = random.choice(persona_data)
            sft_item['persona'] = selected_persona.get('text')
            f.write(json.dumps(sft_item, ensure_ascii=False) + '\n')


def main():
    # Redacted relative file paths
    sft_file = os.path.join("data", "train", "sft_persona_original.jsonl")
    persona_file = os.path.join("data", "train", "train1", "trainpersonas.jsonl")
    output_file = os.path.join("data", "train", "train1", "sft_persona_with_persona.jsonl")

    persona_data = load_persona_data(persona_file)
    sft_data = load_sft_data(sft_file)

    print(f"Loaded {len(persona_data)} persona records")
    print(f"Loaded {len(sft_data)} SFT records")

    merge_persona_with_sft(sft_data, persona_data, output_file)

    print(f"Output written to {output_file} containing {len(sft_data)} records")


if __name__ == "__main__":
    main()