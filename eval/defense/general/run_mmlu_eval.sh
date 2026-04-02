#!/bin/bash
# MMLU evaluation script (offline-friendly, uses local MMLU data)
# Example: bash persona/run_mmlu_eval.sh 1,2

# Offline mode flags (requires local MMLU datasets)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# GPU selection: first script arg overrides GPU_IDS env var
GPU_IDS=${1:-${GPU_IDS:-}}
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    echo "No GPU specified; using system default GPU visibility"
fi

# Ensure lm_eval is called from the persona directory to avoid
# treating a local 'mmlu' directory as an lm_eval task directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration (use environment variables to avoid embedding private paths)
MODEL_PATH="${MODEL_PATH:-/path/to/base/model}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/result/general/mmlu/Qwen2.5-7B-Instruct-qlora}"
# Use lm-eval built-in MMLU task group (dataset_path should point to your local copy)
TASKS="${TASKS:-mmlu}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# QLoRA settings (toggle via env var)
USE_QLORA="${USE_QLORA:-true}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MODEL_PATH}"
PEFT_MODEL_PATH="${PEFT_MODEL_PATH:-/path/to/qlora/adapter}"

echo "======================================================================"
echo "  MMLU evaluation (local data + lm-eval)"
echo "======================================================================"
echo "Model: $MODEL_PATH"
if [ "$USE_QLORA" = true ]; then
    echo "Mode: QLoRA adapter"
    echo "Base model: $BASE_MODEL_PATH"
    echo "Adapter (PEFT): $PEFT_MODEL_PATH"
else
    echo "Mode: Full-parameter model"
fi
echo "Tasks: $TASKS"
echo "Few-shot: $NUM_FEWSHOT"
echo "Batch size: $BATCH_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "======================================================================"

# Choose evaluation command based on QLoRA usage
if [ "$USE_QLORA" = true ]; then
    echo -e "\nUsing vLLM backend + 2 GPUs + QLoRA adapter\n"
    cd "$SCRIPT_DIR" || exit 1
    lm_eval --model vllm \
        --model_args pretrained=$BASE_MODEL_PATH,tensor_parallel_size=2,gpu_memory_utilization=0.8,dtype=float16,enable_lora=true,max_lora_rank=16,max_num_seqs=8,enforce_eager=True \
        --tasks $TASKS \
        --num_fewshot $NUM_FEWSHOT \
        --batch_size auto \
        --output_path $OUTPUT_DIR \
        --log_samples

else
    echo -e "\nUsing vLLM backend + 2 GPUs (offline)\n"
    cd "$SCRIPT_DIR" || exit 1
    lm_eval --model vllm \
        --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,gpu_memory_utilization=0.8,dtype=auto,enforce_eager=True \
        --tasks $TASKS \
        --num_fewshot $NUM_FEWSHOT \
        --batch_size auto \
        --output_path $OUTPUT_DIR \
        --log_samples
fi

echo -e "\n======================================================================"
echo "Evaluation finished. Results saved to: $OUTPUT_DIR"
echo "======================================================================"