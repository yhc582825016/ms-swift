#!/usr/bin/env bash
set -xeuo pipefail

export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-swift}"
export SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-Qwen3-4B-Instruct-2507-general_total_425w_no_ace_reason_cat_online_datas_457w_shuffle-309-cispo-math}"
export NCCL_DEBUG=WARN
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
# Run from any directory.
MS_SWIFT_ROOT=/mnt/code/yehangcheng/ms-swift
cd "${MS_SWIFT_ROOT}"

TRAIN_DATASET="${MS_SWIFT_ROOT}/megatron_output/dapo_math_17k_swift_.jsonl"
VAL_DATASET="${MS_SWIFT_ROOT}/megatron_output/aime_2024_swift_.jsonl"
MASTER_PORT="${MASTER_PORT:-29682}"

PYTORCH_ALLOC_CONF='expandable_segments:True' \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
nohup megatron rlhf \
    --rlhf_type grpo \
    --model /opt/users/ye/checkpoints/Qwen3-4B-Instruct-2507-general_total_425w_no_ace_reason_cat_online_datas_457w_shuffle/checkpoint-2337 \
    --model_type qwen3 \
    --dataset "${TRAIN_DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --external_plugins "/mnt/code/yehangcheng/ms-swift/plugin/math_dapo.py" \
    --reward_funcs external_math_dapo \
    --num_train_epochs 10 \
    --generation_batch_size 512 \
    --global_batch_size 128 \
    --micro_batch_size 1 \
    --num_generations 8 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 8 \
    --vllm_max_model_len 32768 \
    --max_length 8192 \
    --max_completion_length 24000 \
    --tuner_type full \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 4 \
    --pipeline_model_parallel_size 2 \
    --context_parallel_size 1 \
    --lr 1e-6 \
    --lr_decay_style constant \
    --bf16 true \
    --beta 0.0 \
    --dynamic_sample false \
    --overlong_filter false \
    --loss_type cispo \
    --epsilon 0.2 \
    --epsilon_high 5.0 \
    --sleep_level 1 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
    --save_steps 100 \
    --eval_steps 10 \
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
    --log_entropy true \
    --report_to swanlab \
    --swanlab_project "${SWANLAB_PROJECT}" \
    --swanlab_exp_name "${SWANLAB_EXPERIMENT_NAME}" \
    --output_dir "${MS_SWIFT_ROOT}/megatron_output/Qwen3-4B-Instruct-2507-general_total_425w_no_ace_reason_cat_online_datas_457w_shuffle-309-cispo-math" \
    > "/mnt/code/yehangcheng/ms-swift/megatron_output/logs/Qwen3-4B-Instruct-2507-general_total_425w_no_ace_reason_cat_online_datas_457w_shuffle-309-cispo-math.log" 2>&1 &