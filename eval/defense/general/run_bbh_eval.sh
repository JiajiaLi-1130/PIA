#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -------- Configurable parameters --------
# Default MODEL_PATH is a placeholder; set MODEL_PATH in your environment to avoid embedding private paths
MODEL_PATH="${MODEL_PATH:-/path/to/base/model}"
TASKS="${TASKS:-bbh_zeroshot_local_boolean_expressions,bbh_zeroshot_local_causal_judgement,bbh_zeroshot_local_date_understanding,bbh_zeroshot_local_disambiguation_qa,bbh_zeroshot_local_dyck_languages,bbh_zeroshot_local_formal_fallacies,bbh_zeroshot_local_geometric_shapes,bbh_zeroshot_local_hyperbaton,bbh_zeroshot_local_logical_deduction_five_objects,bbh_zeroshot_local_logical_deduction_seven_objects,bbh_zeroshot_local_logical_deduction_three_objects,bbh_zeroshot_local_movie_recommendation,bbh_zeroshot_local_multistep_arithmetic_two,bbh_zeroshot_local_navigate,bbh_zeroshot_local_object_counting,bbh_zeroshot_local_penguins_in_a_table,bbh_zeroshot_local_reasoning_about_colored_objects,bbh_zeroshot_local_ruin_names,bbh_zeroshot_local_salient_translation_error_detection,bbh_zeroshot_local_snarks,bbh_zeroshot_local_sports_understanding,bbh_zeroshot_local_temporal_sequences,bbh_zeroshot_local_tracking_shuffled_objects_five_objects,bbh_zeroshot_local_tracking_shuffled_objects_seven_objects,bbh_zeroshot_local_tracking_shuffled_objects_three_objects,bbh_zeroshot_local_web_of_lies,bbh_zeroshot_local_word_sorting}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"  # official BBH public eval is zero-shot
BATCH_SIZE="${BATCH_SIZE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/persona/data/test/bbh}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"

echo "Running local BBH zero-shot evaluation on tasks: ${TASKS}"

cd "${SCRIPT_DIR}" || exit 1

lm_eval \
  --tasks "${TASKS}" \
  --num_fewshot "${NUM_FEWSHOT}" \
  --apply_chat_template \
  --model vllm \
  --model_args "pretrained=${MODEL_PATH},tensor_parallel_size=2,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},dtype=auto" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUTPUT_DIR}" \
  --log_samples
