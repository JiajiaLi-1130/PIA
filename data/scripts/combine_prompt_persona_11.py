import json
import random
import os
import argparse

# ------------------------ Configuration ------------------------
# Input files (redacted to relative paths)
PROMPTS_FILE = "data/test/unsafe/XSTest_unsafe.jsonl"
PERSONAS_FILE = "data/test/unsafe/elite/attack_elite.jsonl"

# Output directory (will include persona count in filename)
OUTPUT_DIR = "data/test/unsafe/attack1_elite"

# Core setting: number of personas to sample per prompt
PERSONAS_PER_PROMPT = 1

# Random seed (for reproducibility). Default: None (no fixed seed)
DEFAULT_SEED = None
# --------------------------------------------------------


def main():
    """Main entry: combine prompts with random personas to create unique pairs."""
    args = parse_args()
    personas_per_prompt = args.num_personas
    seed = args.seed
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"Random seed set: {seed}")
    
    # Check input files
    check_files_exist()
    
    # Load input data
    prompts = load_prompts(PROMPTS_FILE)
    all_personas = load_personas(PERSONAS_FILE)
    
    # Validate configuration
    validate_config(len(all_personas), personas_per_prompt)
    
    # Generate unique combinations
    combined_results = generate_unique_combinations(prompts, all_personas, personas_per_prompt)
    
    # Build output filename using basenames
    prompt_filename = os.path.splitext(os.path.basename(PROMPTS_FILE))[0]
    persona_filename = os.path.splitext(os.path.basename(PERSONAS_FILE))[0]
    output_file = os.path.join(OUTPUT_DIR, f'{prompt_filename}_{persona_filename}_{personas_per_prompt}x.jsonl')
    
    # Save results and print stats
    save_results(combined_results, output_file)

    print_processing_stats(len(prompts), len(combined_results), output_file, personas_per_prompt)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Combine prompts with random personas to produce unique pairs')
    parser.add_argument(
        '-n', '--num-personas',
        type=int,
        default=PERSONAS_PER_PROMPT,
        help=f'Number of personas to sample per prompt (default: {PERSONAS_PER_PROMPT})'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Random seed for reproducibility (default: not set)'
    )
    return parser.parse_args()


def check_files_exist():
    """Ensure input files exist."""
    if not os.path.exists(PROMPTS_FILE):
        print(f"Error: missing file: {PROMPTS_FILE}")
        exit(1)

    if not os.path.exists(PERSONAS_FILE):
        print(f"Error: missing file: {PERSONAS_FILE}")
        exit(1)

    print("Input files verified")


def load_prompts(file_path):
    """Load prompts from a JSONL file."""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                prompts.append(data['prompt'])
    print(f"Loaded {len(prompts)} prompts")
    return prompts


def load_personas(file_path):
    """Load personas from a JSONL file."""
    personas = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                personas.append(data['persona'])
    print(f"Loaded {len(personas)} personas")
    return personas


def validate_config(total_personas, personas_per_prompt):
    """Validate configuration values."""
    if personas_per_prompt <= 0:
        print(f"Error: personas per prompt must be > 0 (current: {personas_per_prompt})")
        exit(1)

    if personas_per_prompt > total_personas:
        print(f"Warning: total personas ({total_personas}) is less than requested per-prompt ({personas_per_prompt})")
        print("This may prevent some prompts from receiving enough unique personas")


def generate_unique_combinations(prompts, personas, num_personas_per_prompt):
    """Generate the specified number of unique persona combinations per prompt."""
    unique_combinations = set()
    results = []

    for prompt in prompts:
        used_personas_for_prompt = set()
        combinations_count = 0

        while combinations_count < num_personas_per_prompt:
            if len(personas) - len(used_personas_for_prompt) == 0:
                print(f"Warning: insufficient personas; prompt '{prompt[:50]}...' generated {combinations_count} combinations")
                break

            selected_persona = random.choice(personas)
            combination_key = f"{prompt}|||{selected_persona}"

            if combination_key not in unique_combinations and selected_persona not in used_personas_for_prompt:
                unique_combinations.add(combination_key)
                used_personas_for_prompt.add(selected_persona)

                results.append({
                    "prompt": prompt,
                    "persona": selected_persona
                })
                combinations_count += 1

    print(f"Generated {len(results)} unique prompt-persona pairs")
    return results


def save_results(results, output_file_path):
    """Save results to a JSONL file."""
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Results saved to: {output_file_path}")


def print_processing_stats(total_prompts, total_combinations, output_file, personas_per_prompt):
    """Print summary statistics about processing."""
    print("\n" + "="*50)
    print("Processing summary")
    print("="*50)
    print(f"Total prompts: {total_prompts}")
    print(f"Personas per prompt: {personas_per_prompt}")
    print(f"Total unique combinations generated: {total_combinations}")
    print(f"Output file: {output_file}")
    print("="*50)

if __name__ == "__main__":
    main()
