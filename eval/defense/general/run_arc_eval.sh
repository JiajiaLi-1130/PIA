#!/bin/bash
# AI2 ARC evaluation script (offline-friendly)
# Usage: ./run_arc_eval.sh [GPU_IDS]

# Offline mode flags (set these in your environment for fully offline runs)
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

# Configuration (use environment variables to avoid embedding private paths)
# Set these before running or rely on the defaults below
MODEL_PATH="${MODEL_PATH:-/path/to/base/model}"
OUTPUT_DIR="${OUTPUT_DIR:-persona/data/test/arc/sft_train2}"
TASKS="${TASKS:-arc_easy_local,arc_challenge_local}"
NUM_FEWSHOT="${NUM_FEWSHOT:-25}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# QLoRA settings (toggle via env var)
USE_QLORA="${USE_QLORA:-true}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MODEL_PATH}"
PEFT_MODEL_PATH="${PEFT_MODEL_PATH:-/path/to/qlora/adapter}"

echo "======================================================================"
echo "  AI2 ARC evaluation"
echo "======================================================================"
echo "Model: $MODEL_PATH"
if [ "$USE_QLORA" = true ]; then
    echo "Mode: QLoRA (adapter)"
    echo "Base model: $BASE_MODEL_PATH"
    echo "Adapter (PEFT): $PEFT_MODEL_PATH"
else
    echo "Mode: Full-parameter model"
fi
echo "Tasks: $TASKS"
echo "Few-shot: $NUM_FEWSHOT"
echo "Batch size: $BATCH_SIZE"
echo "======================================================================"

# Choose evaluation command based on QLoRA usage
if [ "$USE_QLORA" = true ]; then
    echo -e "\nUsing vLLM backend + 2 GPUs + QLoRA adapter\n"
    lm_eval --model vllm \
        --model_args pretrained=$BASE_MODEL_PATH,tensor_parallel_size=2,gpu_memory_utilization=0.8,dtype=auto,enable_lora=true,max_lora_rank=16,max_num_seqs=16 \
        --tasks $TASKS \
        --num_fewshot $NUM_FEWSHOT \
        --batch_size auto \
        --output_path $OUTPUT_DIR \
        --log_samples \
        --apply_chat_template \
        --fewshot_as_multiturn

    echo -e "\nNote: vLLM's QLoRA support may be limited. If you encounter issues, use the HuggingFace backend:\n"
    echo "lm_eval --model hf \\"
    echo "    --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH \\""
    echo "    --tasks $TASKS \\""
    echo "    --num_fewshot $NUM_FEWSHOT --batch_size $BATCH_SIZE \\""
    echo "    --output_path $OUTPUT_DIR/qlora_hf_results --log_samples"
else
    echo -e "\nUsing vLLM backend + 2 GPUs (offline)\n"
    lm_eval --model vllm \
        --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,gpu_memory_utilization=0.8,dtype=auto,max_num_seqs=16 \
        --tasks $TASKS \
        --num_fewshot $NUM_FEWSHOT \
        --batch_size auto \
        --output_path $OUTPUT_DIR \
        --log_samples
fi

echo -e "\n======================================================================"
echo "Evaluation finished. Results saved to: $OUTPUT_DIR"
echo "======================================================================"
