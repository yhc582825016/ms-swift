#!/usr/bin/env bash
set -xeuo pipefail

# Megatron + GSPO LoRA finetuning for instruct_following reward in ms-swift.
# vLLM server mode:
# 1) Start vLLM rollout server first (example):
#    CUDA_VISIBLE_DEVICES=0,1,2,3 swift rollout --model /opt/users/ye/checkpoints/Qwen3-Omni-30B-A3B-Instruct-220/checkpoint-44398
# 2) Then run this script.

export SWANLAB_MODE="cloud"
export SWANLAB_API_KEY="rDo9t8YJf2yXZO8ilEyWZ"
export SWANLAB_PROJECT="swift"
export SWANLAB_EXPERIMENT_NAME="Qwen3-Omni-30B-A3B-Instruct-gspo-if-lora-server"

# Run from any directory.
cd /mnt/code/yehangcheng/github/ms-swift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MS_SWIFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PLUGIN_PATH="${MS_SWIFT_ROOT}/examples/train/grpo/plugin/instruct_following_plugin.py"

# Dataset columns expected by external_if_strict:
# - messages
# - extra_info (dict/json-string) with:
#     instruction_id_list: List[str]
#     instruction_kwargs: List[Dict]
TRAIN_DATASET="/mnt/code/yehangcheng/github/ms-swift/megatron_output/if_rl_dataset_train_swift_len12k.parquet"
VAL_DATASET="/mnt/code/yehangcheng/github/ms-swift/megatron_output/if_rl_dataset_val_swift_from_ifeval.parquet"
MASTER_PORT="${MASTER_PORT:-29612}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_SERVER_TIMEOUT="${VLLM_SERVER_TIMEOUT:-600}"

LOG_FILE="${MS_SWIFT_ROOT}/logs/Qwen3-Omni-30B-A3B-Instruct-gspo-if-lora-server.log"
OUTPUT_DIR="${MS_SWIFT_ROOT}/megatron_output/Qwen3-Omni-30B-A3B-Instruct-gspo-if-lora-server"
mkdir -p "${MS_SWIFT_ROOT}/logs" "${OUTPUT_DIR}"

PYTORCH_ALLOC_CONF='expandable_segments:True' \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
nohup megatron rlhf \
    --rlhf_type grpo \
    --model /opt/users/ye/checkpoints/Qwen3-Omni-30B-A3B-Instruct-220/checkpoint-44398 \
    --model_type qwen3_omni_moe \
    --dataset "${TRAIN_DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs external_if_strict \
    --num_train_epochs 1 \
    --train_iters 1 \
    --global_batch_size 128 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 16 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "${VLLM_SERVER_HOST}" \
    --vllm_server_port "${VLLM_SERVER_PORT}" \
    --vllm_server_timeout "${VLLM_SERVER_TIMEOUT}" \
    --max_length 8192 \
    --max_completion_length 8192 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --lr 8e-5 \
    --bf16 true \
    --beta 0.0 \
    --importance_sampling_level sequence \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --dynamic_sample true \
    --overlong_filter true \
    --loss_type dapo \
    --offload_model false \
    --offload_bridge false \
    --offload_optimizer false \
    --save_steps 500 \
    --eval_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --recompute_granularity selective \
    --finetune \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim \
    --no_save_rng \
    --attention_backend flash \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k -1 \
    --padding_free true \
    --sequence_parallel true \
    --log_completions true \
    --report_to swanlab \
    --eval_iters 5 \
    --swanlab_project "${SWANLAB_PROJECT}" \
    --swanlab_exp_name "${SWANLAB_EXPERIMENT_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &

echo "Started server-mode training in background."
echo "vLLM server: ${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
