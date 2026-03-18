#!/usr/bin/env bash
set -xeuo pipefail
# rDo9t8YJf2yXZO8ilEyWZ
export SWANLAB_MODE="cloud"
export SWANLAB_API_KEY="rDo9t8YJf2yXZO8ilEyWZ"
export SWANLAB_PROJECT="swift"
export SWANLAB_EXPERIMENT_NAME="Qwen3-Omni-30B-A3B-Instruct-gspo-if-306-lora"
# Megatron + GSPO LoRA finetuning for instruct_following reward in ms-swift.
# Run from any directory.
cd /mnt/code/yehangcheng/ms-swift
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MS_SWIFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MS_SWIFT_ROOT=/mnt/code/yehangcheng/ms-swift
PLUGIN_PATH="${MS_SWIFT_ROOT}/plugin/instruct_following_plugin.py"

# Dataset columns expected by external_if_strict:
# - messages
# - extra_info (dict/json-string) with:
#     instruction_id_list: List[str]
#     instruction_kwargs: List[Dict]
TRAIN_DATASET="/mnt/code/yehangcheng/ms-swift/data/if_rl_dataset_train_swift_len12k_clean.parquet"
VAL_DATASET="/mnt/code/yehangcheng/ms-swift/data/if_rl_dataset_val_swift_from_ifeval_clean.parquet"
MASTER_PORT="${MASTER_PORT:-29611}"
# Convert once (offline) with:
# python examples/models/qwen3_omni/convert_verl_if_to_swift.py --input /mnt/code/yehangcheng/verl/recipe/insturct_following/ifeval_test.parquet --output "${VAL_DATASET}"
PYTORCH_ALLOC_CONF='expandable_segments:True' \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model /opt/users/ye/checkpoints/Qwen3-Omni-30B-A3B-Instruct-220/checkpoint-44398 \
    --model_type qwen3_omni_moe \
    --dataset "${TRAIN_DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs external_if_strict \
    --num_train_epochs 1 \
    --global_batch_size 512 \
    --micro_batch_size 16 \
    --steps_per_generation 1 \
    --num_generations 8 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 8 \
    --vllm_max_model_len 24000 \
    --max_length 8192 \
    --max_completion_length 8192 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 4 \
    --pipeline_model_parallel_size 2 \
    --context_parallel_size 1 \
    --lr 1e-4 \
    --lr_decay_style constant \
    --lr_warmup_iters 0 \
    --bf16 true \
    --beta 0.0 \
    --importance_sampling_level sequence \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --dynamic_sample false \
    --overlong_filter false \
    --loss_type grpo \
    --sleep_level 1 \
    --offload_model false \
    --offload_bridge false \
    --offload_optimizer false \
    --save_steps 50 \
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
    --swanlab_project "${SWANLAB_PROJECT}" \
    --swanlab_exp_name "${SWANLAB_EXPERIMENT_NAME}" \
    --output_dir "${MS_SWIFT_ROOT}/megatron_output/Qwen3-Omni-30B-A3B-Instruct-gspo-if-306-lora" \
    > "/mnt/code/yehangcheng/ms-swift/megatron_output/logs/Qwen3-Omni-30B-A3B-Instruct-gspo-if-306-lora.log" 2>&1 &
    #  tensorboard --logdir /mnt/code/yehangcheng/ms-swift/megatron_output/Qwen3-Omni-30B-A3B-Instruct-gspo-if-lora --host 0.0.0.0 --port 6006

