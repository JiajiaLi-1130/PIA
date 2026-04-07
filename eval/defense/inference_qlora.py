#!/usr/bin/env python3
"""
Example usage:
    python eval/defense/inference_qlora.py \
        --model_dir models/Qwen2.5-7B-Instruct \
        --adapter_dir result/defense/train1/Qwen2.5-7B-RobustDPO-QLora \
        --input_file data/test/safe \
        --output_dir result/defense/safe/inference/train1 \
        --tensor_parallel_size 4 \
        --gpu_utilization 0.8 \
        --max_model_len 4096 \
        --max_output_tokens 2048 \
        --max_lora_rank 16
"""
import os
import warnings
import argparse
import json
import sys
import time
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def safe_len(obj: Any) -> int:
    """Return len(obj) or 0 if unavailable."""
    try:
        return len(obj)  # type: ignore[arg-type]
    except Exception:
        return 0


def apply_chat_template(tokenizer, prompts: List[str]) -> List[str]:
    """Format prompts using tokenizer chat template; use user-only messages."""
    formatted = []

    for p in prompts:
        messages = [
            {"role": "user", "content": p}
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted.append(text)
        except Exception:
            formatted.append(p)
    return formatted

def main():
    parser = argparse.ArgumentParser(description="vLLM batch LoRA inference")
    parser.add_argument('--model_dir', required=True, help='Path to base model')
    parser.add_argument('--adapter_dir', required=True, help='Path to LoRA adapter')
    parser.add_argument('--input_file',  required=True, help='Input path: single JSONL file or directory containing JSONL files')
    parser.add_argument('--output_dir', default='./output', help='Output directory')
    parser.add_argument('--max_model_len', type=int, default=4096, help='Model maximum context length')
    parser.add_argument('--max_output_tokens', type=int, default=2048, help='Max tokens to generate per sample')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Tensor parallel size (number of GPUs)')
    parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated GPU ids to use (e.g. 0,1)')
    parser.add_argument('--gpu_utilization', type=float, default=0.9, help='Target GPU memory utilization')
    parser.add_argument('--max_lora_rank', type=int, default=16, help='Max LoRA rank')
    args = parser.parse_args()

    # 1. Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Set CUDA visible devices if specified (IDs hidden in logs)
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        logger.info("Using specified GPUs (ids hidden)")
        # Update tensor_parallel_size to match provided GPU count
        gpu_count = len(args.gpu_ids.split(','))
        if args.tensor_parallel_size != gpu_count:
            logger.warning(f"Adjusting tensor_parallel_size from {args.tensor_parallel_size} to {gpu_count}")
            args.tensor_parallel_size = gpu_count

    # 3. Find file: compatible with passing in a single file or directory
    import glob
    if os.path.isfile(args.input_file) and args.input_file.endswith(".jsonl"):
        files = [args.input_file]
    else:
        files = sorted(glob.glob(os.path.join(args.input_file, "*.jsonl")))
    if not files:
        logger.error("No JSONL files found")
        sys.exit(1)

    # 4. Initialize tokenizer (for template formatting)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # 5. Initialize vLLM engine and validate paths
    # Check model and adapter paths (log basenames only)
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory not found: {os.path.basename(args.model_dir.rstrip('/'))}")
        sys.exit(2)
    if not os.path.exists(args.adapter_dir):
        logger.error(f"LoRA adapter directory not found: {os.path.basename(args.adapter_dir.rstrip('/'))}")
        sys.exit(2)

    try:
        import torch
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} visible GPU(s)")
        if not args.gpu_ids and args.tensor_parallel_size > gpu_count and gpu_count > 0:
            logger.warning(f"Requested tensor_parallel_size ({args.tensor_parallel_size}) > available GPUs ({gpu_count}); adjusting to {gpu_count}")
            args.tensor_parallel_size = gpu_count
    except Exception as e:
        logger.warning(f"Unable to detect GPU count: {e}")

    if args.max_output_tokens > args.max_model_len:
        logger.warning(f"max_output_tokens ({args.max_output_tokens}) > max_model_len ({args.max_model_len}); adjusting to {args.max_model_len}")
        args.max_output_tokens = args.max_model_len

    logger.info(f"Initializing vLLM engine (TP={args.tensor_parallel_size})...")
    try:
        llm = LLM(
            model=args.model_dir,
            enable_lora=True,
            max_lora_rank=args.max_lora_rank,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_utilization,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            enforce_eager=True
        )
    except Exception as e:
        logger.error(f"vLLM initialization failed: {e}")
        sys.exit(1)

    stop_ids = [getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)]
    stop_ids = [i for i in stop_ids if i is not None]
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_output_tokens,
        stop_token_ids=stop_ids
    )

    # Define LoRA request
    lora_req = LoRARequest("my_adapter", 1, args.adapter_dir)
    logger.info("LoRA adapter mounted (path hidden)")

    # 6. Inference loop
    total_start = time.time()
    
    for file_path in files:
        fname = os.path.basename(file_path)
        logger.info(f"Processing: {fname}")

        raw_data = read_jsonl(file_path)
        if not raw_data:
            continue

        prompts = []
        valid_indices = []
        for i, item in enumerate(raw_data):
            p = item.get('prompt') or item.get('instruction') or item.get('input')
            if p:
                prompts.append(p)
                valid_indices.append(i)
        
        # Apply template
        inputs = apply_chat_template(tokenizer, prompts)

        # Generate; vLLM handles batching
        logger.info(f"Generating {len(inputs)} samples...")
        start_gen = time.time()
        
        try:
            outputs = llm.generate(
                inputs,
                sampling_params,
                lora_request=lora_req
            )
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            logger.error("Check GPU drivers, CUDA version, model/LoRA files, or try reducing --tensor_parallel_size / --max_model_len / --gpu_utilization")
            sys.exit(1)
        
        duration = time.time() - start_gen
        
        results = []
        for i, out in enumerate(outputs):
            idx = valid_indices[i]

            generated_text = None
            try:
                if getattr(out, "outputs", None) and len(out.outputs) > 0:
                    generated_text = getattr(out.outputs[0], "text", None)
                else:
                    generated_text = getattr(out, "text", None)
            except Exception:
                generated_text = None

            if generated_text is None:
                generated_text = ""

            input_tokens = safe_len(getattr(out, "prompt_token_ids", None))
            output_tokens = 0
            if getattr(out, "outputs", None) and len(out.outputs) > 0:
                output_tokens = safe_len(getattr(out.outputs[0], "token_ids", None))

            orig_rec = raw_data[idx]
            base_rec: Dict[str, Any] = dict(orig_rec)
            base_rec['response'] = generated_text
            base_rec['metadata'] = {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            }
            results.append(base_rec)
            
        # Save results (omit model name from output filename)
        data_name = os.path.basename(file_path).replace('.jsonl', '')
        out_name = f"{data_name}-qlora.jsonl"
        with open(os.path.join(args.output_dir, out_name), 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        speed = len(results) / duration
        logger.info(f"Finished {fname}: {speed:.2f} samples/s")

    logger.info(f"All done. Total elapsed: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()