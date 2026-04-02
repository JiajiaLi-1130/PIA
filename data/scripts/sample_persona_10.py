#!/usr/bin/env python3
import json
import os
from collections import Counter
import random

random.seed(42)

SAMPLE_SIZE = 1000

# redacted relative paths
input_file = os.path.join("data", "train", "train1", "sft_persona.jsonl")
output_file = os.path.join("data", "train", "train1", "sft_persona.jsonl")

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

data = []
for line in lines:
    if line.strip():
        data.append(json.loads(line.strip()))

persona_counter = Counter(item['persona'] for item in data)
print(f"Total rows: {len(data)}")
print(f"Unique personas: {len(persona_counter)}")

total_personas = len(persona_counter)
samples_per_persona = SAMPLE_SIZE // total_personas
remainder = SAMPLE_SIZE % total_personas

sampled_data = []
persona_samples = {persona: [] for persona in persona_counter.keys()}

for item in data:
    persona = item['persona']
    if len(persona_samples[persona]) < samples_per_persona:
        persona_samples[persona].append(item)
    elif len(persona_samples[persona]) == samples_per_persona and remainder > 0:
        persona_samples[persona].append(item)
        remainder -= 1

for samples in persona_samples.values():
    sampled_data.extend(samples)

random.shuffle(sampled_data)

print(f"\nTotal after sampling: {len(sampled_data)}")

os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    for item in sampled_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nSaved to: {output_file}")