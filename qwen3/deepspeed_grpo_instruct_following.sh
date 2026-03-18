#!/usr/bin/env bash
set -xeuo pipefail

export SWANLAB_MODE="cloud"
export SWANLAB_API_KEY="rDo9t8YJf2yXZO8ilEyWZ"
export SWANLAB_PROJECT="swift"
export SWANLAB_EXPERIMENT_NAME="Qwen3-4B-Instruct-2507-grpo-if-deepspeed"

cd /mnt/code/yehangcheng/ms-swift
MS_SWIFT_ROOT=/mnt/code/yehangcheng/ms-swift
PLUGIN_PATH="${MS_SWIFT_ROOT}/plugin/instruct_following_plugin.py"
TRAIN_DATASET="${MS_SWIFT_ROOT}/megatron_output/if_rl_dataset_train_swift_len12k_clean.parquet"
VAL_DATASET="${MS_SWIFT_ROOT}/megatron_output/if_rl_dataset_val_swift_from_ifeval_clean.parquet"
MASTER_PORT="${MASTER_PORT:-29621}"

mkdir -p "${MS_SWIFT_ROOT}/logs"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
nohup swift rlhf \
    --rlhf_type grpo \
    --model /opt/users/models/Qwen3-4B-Instruct-2507 \
    --model_type qwen3 \
    --dataset "${TRAIN_DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs external_if_strict \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_generations 8 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 24000 \
    --max_length 8192 \
    --max_completion_length 8192 \
    --tuner_type full \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --beta 0.0 \
    --importance_sampling_level sequence \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --dynamic_sample false \
    --overlong_filter false \
    --loss_type grpo \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --attn_impl flash_attn \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k -1 \
    --padding_free true \
    --sequence_parallel_size 4 \
    --log_completions true \
    --deepspeed zero3 \
    --report_to swanlab \
    --swanlab_project "${SWANLAB_PROJECT}" \
    --swanlab_exp_name "${SWANLAB_EXPERIMENT_NAME}" \
    --output_dir "${MS_SWIFT_ROOT}/megatron_output/Qwen3-4B-Instruct-2507-grpo-if-deepspeed" \
    > "${MS_SWIFT_ROOT}/logs/Qwen3-4B-Instruct-2507-grpo-if-deepspeed.log" 2>&1 &
