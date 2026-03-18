#!/usr/bin/env bash
set -xeuo pipefail
# rDo9t8YJf2yXZO8ilEyWZ
export SWANLAB_MODE="cloud"
export SWANLAB_API_KEY="rDo9t8YJf2yXZO8ilEyWZ"
export SWANLAB_PROJECT="swift"
export SWANLAB_EXPERIMENT_NAME="Qwen3-4B-Instruct-2507-gspo-if-lora"
cd /mnt/code/yehangcheng/ms-swift
MS_SWIFT_ROOT=/mnt/code/yehangcheng/ms-swift
PLUGIN_PATH="${MS_SWIFT_ROOT}/plugin/instruct_following_plugin.py"
TRAIN_DATASET="/mnt/code/yehangcheng/ms-swift/megatron_output/if_rl_dataset_train_swift_len12k_clean.parquet"
VAL_DATASET="/mnt/code/yehangcheng/ms-swift/megatron_output/if_rl_dataset_val_swift_from_ifeval_clean.parquet"
MASTER_PORT="${MASTER_PORT:-29611}"
# Convert once (offline) with:
# python examples/models/qwen3_omni/convert_verl_if_to_swift.py --input /mnt/code/yehangcheng/verl/recipe/insturct_following/ifeval_test.parquet --output "${VAL_DATASET}"
PYTORCH_ALLOC_CONF='expandable_segments:True' \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model /opt/users/models/Qwen3-4B-Instruct-2507 \
    --model_type qwen3 \
    --dataset "${TRAIN_DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs external_if_strict \
    --num_train_epochs 1 \
    --global_batch_size 512 \
    --micro_batch_size 4 \
    --steps_per_generation 1 \
    --num_generations 8 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 24000 \
    --max_length 8192 \
    --max_completion_length 8192 \
    --tuner_type full \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --lr 2e-6 \
    --bf16 true \
    --beta 0.0 \
    --importance_sampling_level sequence \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --dynamic_sample false \
    --overlong_filter false \
    --loss_type grpo \
    --sleep_level 1 \
    --offload_model true \
    --offload_bridge true \
    --offload_optimizer true \
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
    --output_dir "${MS_SWIFT_ROOT}/megatron_output/Qwen3-4B-Instruct-2507-gspo-if-lora" \
    # > "/mnt/code/yehangcheng/ms-swift/logs/Qwen3-4B-Instruct-2507-gspo-if-lora.log" 2>&1 &

    #     --lora_rank 8 \
    # --lora_alpha 32 \
    # --target_modules all-linear \