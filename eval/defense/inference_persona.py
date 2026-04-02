#!/usr/bin/env python3
"""
Usage:
    python eval/defense/inference_persona.py \
        --model_dir models/Qwen2.5-7B-Instruct \
        --input_file data/test/unsafe/test_elite \
        --output_dir result/defense/unsafe/inference \
        --max_model_len 4096 \
        --max_output_tokens 2048 \
        --tensor_parallel_size 2 \
        --gpu_ids 0,1
"""
import argparse
import json
import os
import random
import sys
import time
from typing import Any, List, Dict, Optional, Tuple

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def read_combined_file(path: str) -> List[Dict[str, Any]]:
    pairs = []
    try:
        if path.endswith('.jsonl'):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        persona = data.get('persona')
                        prompt = data.get('prompt')
                        if persona and prompt:
                            pairs.append(data)
                    except Exception as e:
                        print(f"Warning: failed to parse a JSONL line: {e}")
        else:
            # Processing JSON format files
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        persona = item.get('persona')
                        prompt = item.get('prompt')
                        if persona and prompt:
                            pairs.append(item)
            elif isinstance(data, dict):
                # Dictionary format, check if it contains pairs or similar fields.
                if 'pairs' in data and isinstance(data['pairs'], list):
                    for item in data['pairs']:
                        if isinstance(item, dict):
                            persona = item.get('persona')
                            prompt = item.get('prompt')
                            if persona and prompt:
                                pairs.append(item)
                elif 'persona' in data and 'prompt' in data:
                    # In the case of a single record, the original dictionary is retained
                    pairs.append(data)
        
        print(f"Successfully read {len(pairs)} persona/prompt pairs.")
        return pairs
    except Exception as e:
        print(f"Failed to read data file: {e}")
        return []

def format_chat_messages(
    prompt_pairs: List[Tuple[str, str]], 
    tokenizer
) -> List[str]:
    """
    Use the tokenizer's chat template to format the prompt.
    """
    formatted_prompts = []

    for persona_text, question_text in prompt_pairs:
        try:
            user_content = (
                "Answer the question according to the assigned persona.\n"
                f"persona: {persona_text}\n"
                f"question: {question_text}"
            )

            messages = [
                {"role": "user", "content": user_content}
            ]

            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            formatted_prompts.append(formatted_prompt)

        except Exception as e:
            # Add more detailed error logs
            print(f"Warning: formatting prompt failed (persona: {persona_text[:30]}..., question: {question_text[:30]}...): {e}")
            formatted_prompts.append(user_content)

    return formatted_prompts

def vllm_inference(
    llm, 
    tokenizer,
    sampling_params,
    personas: List[str],
    questions: List[str],
    original_items: Optional[List[Dict[str, Any]]] = None,
    output_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    if not personas:
        print("The persona list is empty.")
        return []
    
    if not questions:
        print("The question list is empty.")
        return []
    
    if len(personas) != len(questions):
        print(f"The number of personas and questions do not match: {len(personas)} != {len(questions)}")
        # Truncate to the shorter length
        min_len = min(len(personas), len(questions))
        personas = personas[:min_len]
        questions = questions[:min_len]
        print(f"Truncated to {min_len} samples")
    
    print(f"One-to-one inference: {len(personas)} samples.")
    
    prompt_pairs: List[Tuple[str, str]] = []
    original_questions: List[str] = []
    original_personas: List[str] = []
    
    for persona, question in zip(personas, questions):
        prompt_pairs.append((persona, question))
        original_questions.append(question)
        original_personas.append(persona)
    
    if not prompt_pairs:
        print("❌ No valid prompt combination was generated.")
        return []

    if tokenizer:
        formatted_prompts = format_chat_messages(prompt_pairs, tokenizer)
    else:
        print("No tokenizer loaded; using raw prompts")

    results = []
    try:
        inference_start = time.time()

        outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)

        inference_time = time.time() - inference_start
        print(f"Inference completed in {inference_time:.2f}s")
        if inference_time > 0:
            print(f"Throughput: {len(formatted_prompts)/inference_time:.2f} samples/s")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user; saving completed results.")
    except Exception as e:
        print(f"Error during inference: {e}")
        return []

    if 'outputs' in locals():
        for i, output in enumerate(outputs):
            if i < len(original_questions):
                original_question = original_questions[i]
                original_persona = original_personas[i]
                generated_text = output.outputs[0].text if output.outputs else ""

                prompt_token_ids = getattr(output, "prompt_token_ids", None)
                input_tokens = len(prompt_token_ids) if prompt_token_ids is not None else 0

                output_tokens = 0
                if output.outputs and getattr(output.outputs[0], "token_ids", None) is not None:
                    output_tokens = len(output.outputs[0].token_ids)

                # Keep original record (if provided), and append response + metadata
                if original_items and i < len(original_items):
                    base_rec: Dict[str, Any] = dict(original_items[i])
                else:
                    base_rec: Dict[str, Any] = {
                        'persona': original_persona,
                        'prompt': original_question
                    }

                base_rec['response'] = generated_text
                base_rec['metadata'] = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens
                }

                results.append(base_rec)
                
    if output_file and results:
        save_incremental_results(results, output_file, len(results), final=True)

    completed_count = len(results)
    total_count = len(formatted_prompts)
    completion_rate = (completed_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"✅ successfully processed {completed_count}/{total_count} samples ({completion_rate:.2f}%)")
    
    return results


def save_incremental_results(results: List[Dict[str, Any]], output_file: str, count: int, final: bool = False):
    """Save inference results incrementally to a file.

    Backup filename uses the count to avoid overwriting intermediate results.
    """
    try:
        dirpath = os.path.dirname(output_file)
        if not dirpath:
            dirpath = '.'
        os.makedirs(dirpath, exist_ok=True)
        
        backup_file = output_file.replace('.jsonl', f'_backup_{count}.jsonl')
        
        # Save to main file
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # If not final, also write a backup file
        if not final:
            with open(backup_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"Saved {count} samples to {os.path.basename(output_file)} and backup {os.path.basename(backup_file)}")
        else:
            print(f"Saved {count} samples to {os.path.basename(output_file)}")
            
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Persona x instruction inference",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model_dir', required=True, help='Local model directory path')
    parser.add_argument('--input_file', required=True, help='Input file or directory (JSON/JSONL with persona and prompt fields)')
    parser.add_argument('--max_model_len', type=int, default=4096, help='LLM context window length')
    parser.add_argument('--max_output_tokens', type=int, default=2048, help='Max tokens to generate per sample')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated GPU ids to use')
    parser.add_argument('--output_dir', type=str, default='result/defense/safe/inference', help='Output directory')
    args = parser.parse_args()

    # 2. Set CUDA visible devices if specified (hide IDs in logs)
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print("Using specified GPUs (ids hidden)")
        # Update tensor_parallel_size to match provided GPU count
        gpu_count = len(args.gpu_ids.split(','))
        if args.tensor_parallel_size != gpu_count:
            print(f"Adjusting tensor_parallel_size from {args.tensor_parallel_size} to {gpu_count}")
            args.tensor_parallel_size = gpu_count

    # 3. Input validation
    if not os.path.exists(args.model_dir):
        print(f"Model directory not found: {os.path.basename(args.model_dir.rstrip('/'))}")
        sys.exit(2)
    if not os.path.exists(args.input_file):
        print(f"Input path not found: {os.path.basename(args.input_file.rstrip('/'))}")
        sys.exit(2)

    # 4. Check available GPUs
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} visible GPU(s)")

        if not args.gpu_ids and args.tensor_parallel_size > gpu_count:
            print(f"tensor_parallel_size ({args.tensor_parallel_size}) > available GPUs ({gpu_count}), adjusting to {gpu_count}")
            args.tensor_parallel_size = gpu_count
    except Exception as e:
        print(f"Unable to detect GPUs: {e}")

    # 5. Get all the JSONL files that need to be processed
    import glob
    if os.path.isfile(args.input_file):
        jsonl_files = [args.input_file]
    else:
        jsonl_files = glob.glob(os.path.join(args.input_file, "*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found under path: {os.path.basename(args.input_file.rstrip('/'))}")
        sys.exit(2)
    
    jsonl_files.sort()
    print(f"Found {len(jsonl_files)} JSONL files:")
    
    total_samples = 0
    for file_path in jsonl_files:
        with open(file_path, encoding='utf-8') as f:
            file_samples = sum(1 for line in f if line.strip())
        print(f"  - {os.path.basename(file_path)}: {file_samples} samples")
        total_samples += file_samples
    
    print(f"Total samples: {total_samples}")
    print("=" * 60)

    # 6. Initialize the vLLM engine
    print(f"\nInitializing vLLM engine (tensor_parallel_size={args.tensor_parallel_size})...")
    init_start = time.time()
    
    try:
        llm = LLM(
            model=args.model_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            dtype="float16",
            distributed_executor_backend="mp",
            max_model_len=args.max_model_len,
            enforce_eager=True
        )
        init_time = time.time() - init_start
        print(f"vLLM initialized in {init_time:.2f}s")
    except Exception as e:
        print(f"vLLM initialization failed: {e}")
        sys.exit(1)

    # 7. Obtain the tokenizer for formatting.
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        print("Tokenizer loaded")
    except Exception as e:
        print(f"Tokenizer load failed, using raw prompts: {e}")
        tokenizer = None

    # 8. Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_output_tokens,
        skip_special_tokens=True,
    )

    # 9. Process files one by one
    start_time = time.time()
    processed_total = 0
    
    for i, data_file in enumerate(jsonl_files, 1):
        file_name = os.path.basename(data_file)
        print(f"\nProcessing file [{i}/{len(jsonl_files)}]: {file_name}")
        
        # Read input data
        print("Reading input data (expects persona and prompt fields)...")
        raw_pairs = read_combined_file(data_file)
        if not raw_pairs:
            print(f"No valid data in file {file_name}, skipping")
            continue
        
        personas = [pair['persona'] for pair in raw_pairs]
        questions = [pair['prompt'] for pair in raw_pairs]
        
        # Prepare output file path (model name is omitted for privacy)
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        data_name = os.path.basename(data_file).replace('.jsonl', '')
        out_filename = f'{data_name}.jsonl'
        out_path = os.path.join(out_dir, out_filename)
        
        file_start = time.time()
        results = vllm_inference(llm, tokenizer, sampling_params, personas, questions, raw_pairs, out_path)
        file_time = time.time() - file_start

        if not results:
            print(f"Inference failed for file {file_name}")
            continue

        processed_total += len(results)
        print(f"File {file_name} completed")
        print(f"Time: {file_time:.2f}s")
        print(f"Processed cumulative: {processed_total}/{total_samples}")

    # 10. Summarize total time
    total_time = time.time() - start_time
    total_minutes = int(total_time // 60)
    total_seconds = int(total_time % 60)
    print(f"Inference finished. Total time: {total_minutes}m{total_seconds}s")

if __name__ == '__main__':
    """Entry point for persona cross-combination inference script."""
    main()