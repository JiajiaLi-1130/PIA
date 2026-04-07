#!/usr/bin/env python3
"""
Example usage (replace placeholders):
    python eval/defense/inference.py \
        --model_dir models/Qwen2.5-7B-Instruct \
        --input_file data/test/safe \
        --output_dir result/defense/safe/inference \
        --max_output_tokens 2048 \
        --max_model_len 4096 \
        --tensor_parallel_size 4
"""
import argparse
import json
import os
import sys
import time
import logging
from typing import Any, Dict

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_first_n_jsonl(path: str, n: int) -> list[dict[str, Any]]:
    """Read the first N records from a JSONL file; if n == -1, read all records."""
    samples = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n > 0 and i >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception as e:
                print(f"Failed to parse JSON on line {i+1}: {e}")
    return samples


def format_chat_messages(prompts: list[str], tokenizer) -> list[str]:
    """
    Use the tokenizer's chat template to format prompts and only use user prompt words.
    """
    formatted_prompts = []
    
    for prompt_text in prompts:
        try:
            messages = [
                {"role": "user", "content": prompt_text}
            ]
            
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
                
        except Exception as e:
            print(f"Warning: failed to format prompt, using raw prompt: {e}")
            # Fallback to the original prompt
            formatted_prompts.append(prompt_text)
    
    return formatted_prompts


def vllm_inference(
    llm, 
    tokenizer,
    sampling_params,
    prompts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    # 1. Extract the prompt text and record the original index.
    prompt_texts = []
    orig_indices = []
    for i, p in enumerate(prompts):
        prompt_text = (
            p.get('prompt') or 
            p.get('instruction') or 
            p.get('input') or 
            p.get('question')
        )
        if not prompt_text:
            print(f"Warning: sample {i+1} has no valid prompt field")
            continue
        prompt_texts.append(prompt_text)
        orig_indices.append(i)

    if not prompt_texts:
        print("No valid prompts found")
        return []

    # 2. Format prompts
    if tokenizer:
        formatted_prompts = format_chat_messages(prompt_texts, tokenizer)
    else:
        formatted_prompts = prompt_texts

    print(f"Preparing to infer {len(formatted_prompts)} samples...")

    # 3. Batch inference
    try:
        print("Starting batch inference...")
        inference_start = time.time()
        outputs = llm.generate(formatted_prompts, sampling_params)
        inference_time = time.time() - inference_start
        print(f"Inference completed in {inference_time:.2f}s")
        if inference_time > 0:
            print(f"Throughput: {len(formatted_prompts)/inference_time:.2f} samples/s")
    except Exception as e:
        print(f"Error during inference: {e}")
        return []

    # 4. Processing results
    results = []
    for i, output in enumerate(outputs):
        if i < len(prompt_texts):
            generated_text = output.outputs[0].text if output.outputs else ""

            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            input_tokens = len(prompt_token_ids) if prompt_token_ids is not None else 0

            output_tokens = 0
            if output.outputs and getattr(output.outputs[0], "token_ids", None) is not None:
                output_tokens = len(output.outputs[0].token_ids)

            orig_idx = orig_indices[i]
            orig_rec = prompts[orig_idx]
            base_rec: Dict[str, Any] = dict(orig_rec)
            base_rec['response'] = generated_text
            base_rec['metadata'] = {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            }
            results.append(base_rec)

    print(f"✅ Successfully processed {len(results)} samples")
    return results


def main():
    # 1. Command line parameter parsing
    parser = argparse.ArgumentParser(
        description="Local model batch inference tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model_dir', required=True, help='Local model directory')
    parser.add_argument('--input_file', required=True, help='Input path: single JSONL file or directory containing JSONL files')
    parser.add_argument('--max_output_tokens', type=int, default=2048, help='Max tokens to generate per sample')
    parser.add_argument('--output_dir', type=str, default='result/defense/safe/inference', help='Output directory')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--max_model_len', type=int, default=4096, help='Model maximum context length')
    parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated GPU ids to use')
    args = parser.parse_args()

    # 2. Configure GPU
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

    # 4. Check the number of GPUs
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} visible GPU(s)")
        
        if not args.gpu_ids and args.tensor_parallel_size > gpu_count:
            print(f"tensor_parallel_size ({args.tensor_parallel_size}) > available GPUs ({gpu_count}); adjusting to {gpu_count}")
            args.tensor_parallel_size = gpu_count
    except Exception as e:
        print(f"Unable to detect GPUs: {e}")

    if args.max_output_tokens > args.max_model_len:
        print(f"max_output_tokens ({args.max_output_tokens}) > max_model_len ({args.max_model_len}); adjusting to {args.max_model_len}")
        args.max_output_tokens = args.max_model_len

    # 5. Get all jsonl files
    import glob
    if os.path.isfile(args.input_file):
        jsonl_files = [args.input_file]
    else:
        jsonl_files = glob.glob(os.path.join(args.input_file, "*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found under path: {os.path.basename(args.input_file.rstrip('/'))}")
        sys.exit(2)

    jsonl_files.sort()  # sort by name
    logger.info(f"Found {len(jsonl_files)} JSONL files:")

    # Count total samples
    total_samples = 0
    for file_path in jsonl_files:
        with open(file_path, encoding='utf-8') as f:
            file_samples = sum(1 for _ in f)
        print(f"  - {os.path.basename(file_path)}: {file_samples} samples")
        total_samples += file_samples

    logger.info(f"Total samples: {total_samples}")
    print("=" * 60)

    # 6. Initialize vLLM engine
    print(f"\nInitializing vLLM engine (tensor_parallel_size={args.tensor_parallel_size})...")
    init_start = time.time()
    
    try:
        llm = LLM(
            model=args.model_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.8,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            distributed_executor_backend="mp",
            dtype="float16",
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
        logger.info("Tokenizer loaded")
    except Exception as e:
        logger.warning(f"Tokenizer load failed, using raw prompts: {e}")
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
        print(f"\nProcess files [{i}/{len(jsonl_files)}]: {file_name}")
        
        samples = read_first_n_jsonl(data_file, -1)
        print(f"📊 {len(samples)} samples have been read")
        if not samples:
            print("⚠️  File is empty, skip")
            continue
        
        file_start = time.time()
        results = vllm_inference(
            llm, tokenizer, sampling_params,
            samples
        )
        file_time = time.time() - file_start
        
        if not results:
            print(f"Inference failed for file {file_name}")
            continue

        # Save results (omit model name from filename)
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

        data_name = os.path.basename(data_file).replace('.jsonl', '')
        out_filename = f'{data_name}.jsonl'
        out_path = os.path.join(out_dir, out_filename)

        logger.info(f"Saving results to {os.path.basename(out_path)}...")
        with open(out_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        processed_total += len(results)
        file_throughput = len(results) / file_time if file_time > 0 else 0
        print(f"File {file_name} completed")
        print(f"Time: {file_time:.2f}s")
        print(f"Processed cumulative: {processed_total}/{total_samples}")

    # 10. Summarize
    total_time = time.time() - start_time
    total_minutes = int(total_time // 60)
    total_seconds = int(total_time % 60)
    overall_throughput = processed_total / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"All files inferred. Total time: {total_minutes}m{total_seconds}s")
    print(f"Files processed: {len(jsonl_files)}")
    print("=" * 60)


if __name__ == '__main__':
    main()