#!/usr/bin/env python3
"""
使用方法:
    python eval/defense/inference_persona_qlora.py \
        --model_dir models/Qwen2.5-7B-Instruct \
        --adapter_dir result/defense/train1/Qwen2.5-7B-RobustDPO-QLora \
        --input_file data/test/unsafe/test_elite \
        --output_dir result/defense/unsafe/inference/train1 \
        --gpu_utilization 0.8 \
        --max_model_len 4096 \
        --max_output_tokens 2048 \
        --max_lora_rank 16

Notes:
- Paths are placeholders to avoid exposing local/private paths.
"""

import argparse
import json
import os
import sys
import time
import logging
from typing import Any, List, Tuple, Dict
import glob

# Suppress verbose TF / protobuf warnings
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    print("Error: vllm not found. Install with: pip install vllm")
    sys.exit(1)

from transformers import AutoTokenizer

# Set logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_input_file(path: str) -> List[Dict[str, Any]]:
    """Read a JSON or JSONL file and return records that include 'persona' and 'prompt'."""
    pairs: List[Dict[str, Any]] = []
    try:
        if path.endswith('.jsonl'):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        logger.warning("Warning: failed to parse a JSONL line")
                        continue
                    if isinstance(data, dict) and data.get('persona') and data.get('prompt'):
                        pairs.append(data)
        else:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get('persona') and item.get('prompt'):
                        pairs.append(item)
            elif isinstance(data, dict):
                if 'pairs' in data and isinstance(data['pairs'], list):
                    for item in data['pairs']:
                        if isinstance(item, dict) and item.get('persona') and item.get('prompt'):
                            pairs.append(item)
                elif data.get('persona') and data.get('prompt'):
                    pairs.append(data)

        logger.info(f"Read {len(pairs)} persona/prompt records from {os.path.basename(path)}")
        return pairs
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return []

def apply_chat_template(
    records: List[Dict[str, Any]], 
    tokenizer
) -> List[str]:
    """Format persona+prompt pairs into single-user messages for generation."""
    formatted_prompts: List[str] = []
    logger.info("Applying chat template to records")

    for rec in records:
        persona_text = rec.get('persona')
        question_text = rec.get('prompt')

        user_content = (
            "Answer the question according to the assigned persona.\n"
            f"persona: {persona_text}\n"
            f"question: {question_text}"
        )

        messages = [{"role": "user", "content": user_content}]

        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(text)
        except Exception:
            logger.warning("Chat template formatting failed; using raw user content")
            formatted_prompts.append(user_content)

    return formatted_prompts

def main():
    parser = argparse.ArgumentParser(description="Persona + instruction QLora inference")
    parser.add_argument('--model_dir', required=True, help='Path to base model')
    parser.add_argument('--adapter_dir', required=True, help='Path to LoRA adapter')
    parser.add_argument('--input_file', required=True, help='Input file or directory (JSON/JSONL)')
    parser.add_argument('--output_dir', help='Output directory (default: persona/out)')
    parser.add_argument('--max_model_len', type=int, default=2560, help='Max model context length')
    parser.add_argument('--max_output_tokens', type=int, default=2048, help='Max tokens to generate')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Tensor parallel size (GPU count)')
    parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated GPU ids (e.g. 0,1)')
    parser.add_argument('--gpu_utilization', type=float, default=0.9, help='GPU memory utilization target')
    parser.add_argument('--max_lora_rank', type=int, default=64, help='Max LoRA rank')
    args = parser.parse_args()

    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        logger.info("Using specified GPUs (ids hidden)")
        gpu_count = len(args.gpu_ids.split(','))
        if args.tensor_parallel_size != gpu_count:
            logger.warning(f"Adjusting tensor_parallel_size from {args.tensor_parallel_size} to {gpu_count}")
            args.tensor_parallel_size = gpu_count

    try:
        import torch
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} visible GPU(s)")
        if not args.gpu_ids and args.tensor_parallel_size > gpu_count and gpu_count > 0:
            logger.warning(f"tensor_parallel_size ({args.tensor_parallel_size}) > available GPUs ({gpu_count}); adjusting to {gpu_count}")
            args.tensor_parallel_size = gpu_count
    except Exception as e:
        logger.warning(f"Unable to detect GPUs: {e}")

    import glob
    if os.path.isfile(args.input_file) and args.input_file.endswith('.jsonl'):
        jsonl_files = [args.input_file]
    else:
        jsonl_files = sorted(glob.glob(os.path.join(args.input_file, "*.jsonl")))

    if not jsonl_files:
        logger.error(f"No JSONL files found under path: {args.input_file}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    logger.info(f"Initializing vLLM (TP={args.tensor_parallel_size}, LoRA enabled)")
    try:
        llm = LLM(
            model=args.model_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_utilization,
            trust_remote_code=True,
            enable_lora=True,          # Key: Enable LoRA
            max_lora_rank=args.max_lora_rank,          # Set a maximum Rank, which must be greater than or equal to the Rank used during training.
            max_model_len=args.max_model_len,
            enforce_eager=True
        )
    except Exception as e:
        logger.error(f"vLLM initialization failed: {e}")
        sys.exit(1)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_output_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else []
    )

    lora_req = LoRARequest("persona_adapter", 1, args.adapter_dir)
    logger.info("LoRA adapter mounted (path hidden)")

    output_dir = args.output_dir or 'result/defense/unsafe/inference'
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()
    processed_total = 0
    total_samples = 0
    # Count total samples first
    for fp in jsonl_files:
        with open(fp, encoding='utf-8') as f:
            total_samples += sum(1 for line in f if line.strip())

    print(f"Found {len(jsonl_files)} files, {total_samples} total samples")

    for i, file_path in enumerate(jsonl_files, 1):
        fname = os.path.basename(file_path)
        logger.info(f"Processing [{i}/{len(jsonl_files)}]: {fname}")

        raw_pairs = read_input_file(file_path)
        if not raw_pairs:
            logger.warning(f"No valid data in file {fname}, skipping")
            continue

        # Format inputs and generate
        formatted_inputs = apply_chat_template(raw_pairs, tokenizer)

        try:
            start = time.time()
            outputs = llm.generate(formatted_inputs, sampling_params, lora_request=lora_req)
            duration = time.time() - start
        except Exception as e:
            logger.error(f"vLLM generation failed for file {fname}: {e}")
            continue

        # Prepare and write results (model name omitted for privacy)
        data_name = os.path.basename(file_path).replace('.jsonl', '')
        out_path = os.path.join(output_dir, f"{data_name}-qlora.jsonl")

        results_to_save: List[Dict[str, Any]] = []
        for j, out in enumerate(outputs):
            orig_rec = raw_pairs[j]
            generated_text = out.outputs[0].text if out.outputs else ""

            prompt_tokens = getattr(out, 'prompt_token_ids', []) or []
            output_tokens = getattr(out.outputs[0], 'token_ids', []) if getattr(out, 'outputs', None) else []

            base_rec: Dict[str, Any] = dict(orig_rec)
            base_rec['response'] = generated_text
            base_rec['metadata'] = {
                'input_tokens': len(prompt_tokens),
                'output_tokens': len(output_tokens),
                'total_tokens': len(prompt_tokens) + len(output_tokens)
            }
            results_to_save.append(base_rec)

        with open(out_path, 'w', encoding='utf-8') as f:
            for item in results_to_save:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        processed_total += len(results_to_save)
        speed = len(results_to_save) / duration if duration > 0 else 0
        logger.info(f"Finished {fname}: {len(results_to_save)} samples, {speed:.2f} samples/s. Output: {os.path.basename(out_path)}")

    elapsed = time.time() - total_start
    logger.info(f"All files processed. Elapsed: {elapsed:.2f}s, processed: {processed_total}/{total_samples}")

if __name__ == "__main__":
    main()