#!/usr/bin/env bash
set -xeuo pipefail
# rDo9t8YJf2yXZO8ilEyWZ
# export SWANLAB_MODE="cloud"
# export SWANLAB_API_KEY="rDo9t8YJf2yXZO8ilEyWZ"
# export SWANLAB_PROJECT="swift"
# export SWANLAB_EXPERIMENT_NAME="Qwen3-4B-Instruct-2507-457w-math"
# Megatron + GSPO LoRA finetuning for instruct_following reward in ms-swift.
# Run from any directory.
cd /mnt/code/yehangcheng/ms-swift
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MS_SWIFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MS_SWIFT_ROOT=/mnt/code/yehangcheng/ms-swift
# AI-MO/NuminaMath-TIR
# TRAIN_DATASET="${MS_SWIFT_ROOT}/megatron_output/dapo_math_17k_swift_.jsonl"
TRAIN_DATASET="open-r1/DAPO-Math-17k-Processed"
VAL_DATASET="${MS_SWIFT_ROOT}/megatron_output/aime_2024_swift_.jsonl"
MASTER_PORT="${MASTER_PORT:-29611}"
PYTORCH_ALLOC_CONF='expandable_segments:True' \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model /opt/users/ye/checkpoints/Qwen3-4B-Instruct-2507-general_total_425w_no_ace_reason_cat_online_datas_457w_shuffle/checkpoint-2337 \
    --model_type qwen3 \
    --dataset "${TRAIN_DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --reward_funcs accuracy \
    --num_train_epochs 10 \
    --global_batch_size 128 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 8 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 8 \
    --vllm_max_model_len 32000 \
    --max_length 6000 \
    --max_completion_length 24000 \
    --tuner_type full \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 2 \
    --context_parallel_size 1 \
    --lr 1e-5 \
    --lr_decay_style constant \
    --lr_warmup_iters 0 \
    --bf16 true \
    --beta 0.0 \
    --importance_sampling_level sequence \
    --epsilon 0.0003 \
    --epsilon_high 0.0004 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --sleep_level 1 \
    --offload_model false \
    --offload_bridge false \
    --offload_optimizer false \
    --save_steps 100 \
    --eval_steps 20 \
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
    --report_to tensorboard \
    --output_dir "${MS_SWIFT_ROOT}/megatron_output/Qwen3-4B-Instruct-2507-457w-math" \
    > "/mnt/code/yehangcheng/ms-swift/megatron_output/logs/Qwen3-4B-Instruct-2507-457w-math.log" 2>&1 &
    #  tensorboard --logdir /mnt/code/yehangcheng/ms-swift/megatron_output/Qwen3-Omni-30B-A3B-Instruct-gspo-if-lora --host 0.0.0.0 --port 6006

    # --swanlab_project "${SWANLAB_PROJECT}" \
    # --swanlab_exp_name "${SWANLAB_EXPERIMENT_NAME}" \