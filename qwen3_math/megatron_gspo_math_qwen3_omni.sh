#!/usr/bin/env bash
set -xeuo pipefail

export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-swift}"
export SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-Qwen3-Omni-30B-A3B-Instruct-309-gspo-math}"
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
# Run from any directory.
MS_SWIFT_ROOT=/mnt/code/yehangcheng/ms-swift
cd "${MS_SWIFT_ROOT}"

TRAIN_DATASET="${MS_SWIFT_ROOT}/megatron_output/dapo_math_17k_swift_.jsonl"
VAL_DATASET="${MS_SWIFT_ROOT}/megatron_output/aime_2024_swift_.jsonl"
MASTER_PORT="${MASTER_PORT:-29682}"
LOG_PATH="${MS_SWIFT_ROOT}/logs/Qwen3-Omni-30B-A3B-Instruct-309-gspo-math.log"

PYTORCH_ALLOC_CONF='expandable_segments:True' \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model /opt/users/ye/checkpoints/Qwen3-Omni-30B-A3B-Instruct-gspo-if-306-lora/checkpoint-250-merged \
    --model_type qwen3_omni_moe \
    --dataset "${TRAIN_DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --reward_funcs accuracy \
    --num_train_epochs 1 \
    --global_batch_size 128 \
    --micro_batch_size 1 \
    --steps_per_generation 4 \
    --num_generations 8 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 8 \
    --vllm_max_model_len 20480 \
    --max_length 8192 \
    --max_completion_length 12000 \
    --tuner_type full \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 4 \
    --pipeline_model_parallel_size 2 \
    --context_parallel_size 1 \
    --lr 1e-6 \
    --bf16 true \
    --beta 0.0 \
    --dynamic_sample false \
    --overlong_filter false \
    --loss_type cispo \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --sleep_level 1 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --importance_sampling_level sequence \
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
    --swanlab_project "${SWANLAB_PROJECT}" \
    --swanlab_exp_name "${SWANLAB_EXPERIMENT_NAME}" \
    --output_dir "${MS_SWIFT_ROOT}/megatron_output/Qwen3-Omni-30B-A3B-Instruct-309-gspo-math"