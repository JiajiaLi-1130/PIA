#!/bin/bash
# Training runner (uses `accelerate`).
# To launch TensorBoard: `tensorboard --logdir /path/to/runs --port 6006`.

# Log output (append to log file). Path redacted for OSS.
LOG_FILE="result/defense/train1/training.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# 1. Environment setup (do not change core commands)
# Activate virtual environment if available (path redacted)
if [ -n "$VIRTUAL_ENV" ]; then
    :
else
    if [ -f "/path/to/venv/bin/activate" ]; then
        source "/path/to/venv/bin/activate"
    fi
fi

# Debug log level for accelerate
export ACCELERATE_LOG_LEVEL=info

# Optimize PyTorch memory allocator
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 2. Launch command (paths redacted; replace placeholders before running)
accelerate launch \
        --num_processes 3 \
        --num_machines 1 \
        --mixed_precision fp16 \
        --main_process_port 29500 \
        run_training.py \
        --model_name_or_path models/Qwen2.5-7B-Instruct \
        --data_files data/train/train1/training.jsonl \
        --mixing_strategy original \
        --output_dir result/defense/train1 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 14 \
        --learning_rate 1.0e-6 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --beta 0.1 \
        --logging_steps 1 \
        --save_strategy no \
        --eval_strategy no \
        --max_length 4096 \
        --max_prompt_length 2048 \
        --gradient_checkpointing \
        --remove_unused_columns False \
        --report_to tensorboard \
        --load_in_4bit \
        --bnb_4bit_quant_type nf4 \
        --use_bnb_nested_quant \
        --dtype float16 \
        --use_peft \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --lora_target_modules q_proj k_proj v_proj o_proj \
        --sft_alpha 0.1 \
        --pic_lambda 1 \
        --pic_top_k 100 \
        --seed 42