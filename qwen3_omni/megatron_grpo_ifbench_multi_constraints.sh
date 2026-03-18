#!/usr/bin/env bash
set -xeuo pipefail
export SWANLAB_MODE="cloud"
export SWANLAB_API_KEY="rDo9t8YJf2yXZO8ilEyWZ"
export SWANLAB_PROJECT="swift"
export SWANLAB_EXPERIMENT_NAME="Qwen3-Omni-30B-A3B-Instruct-gspo-if-305-ifbench-lora"
# Run from any directory.
cd /mnt/code/yehangcheng/ms-swift

MS_SWIFT_ROOT=/mnt/code/yehangcheng/ms-swift
PLUGIN_PATH="${MS_SWIFT_ROOT}/plugin/ifbench_instruct_following_plugin.py"
# CONVERT_SCRIPT="${MS_SWIFT_ROOT}/models/qwen3_omni/convert_if_multi_constraints_to_swift.py"

RAW_DATASET="/mnt/code/yehangcheng/all_data/rlhf_data/IF_multi_constraints_upto5"
TRAIN_DATASET="${MS_SWIFT_ROOT}/megatron_output/if_multi_constraints_upto5_train_swift.parquet"
VAL_DATASET="${MS_SWIFT_ROOT}/megatron_output/if_rl_dataset_val_swift_from_ifeval_clean.parquet"
LOG_FILE="${MS_SWIFT_ROOT}/logs/Qwen3-Omni-30B-A3B-Instruct-gspo-if-305-ifbench-multi.log"
OUTPUT_DIR="${MS_SWIFT_ROOT}/megatron_output/Qwen3-Omni-30B-A3B-Instruct-gspo-if-305-ifbench-multi-lora"
MASTER_PORT="${MASTER_PORT:-29621}"

mkdir -p "${MS_SWIFT_ROOT}/logs" "${MS_SWIFT_ROOT}/megatron_output"

# 1) Convert IF_multi_constraints_upto5 into ms-swift training format:
#    Required columns:
#      - messages
#      - extra_info.instruction_id_list
#      - extra_info.instruction_kwargs
# python "${CONVERT_SCRIPT}" \
#     --input "${RAW_DATASET}" \
#     --train_output "${TRAIN_DATASET}" \
#     --val_output "${VAL_DATASET}" \
#     --val_size 2000 \
#     --seed 42

# 2) GRPO training with IFbench strict reward function.
PYTORCH_ALLOC_CONF='expandable_segments:True' \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
nohup megatron rlhf \
    --rlhf_type grpo \
    --model /opt/users/ye/checkpoints/Qwen3-Omni-30B-A3B-Instruct-gspo-if-305/checkpoint-500-merged \
    --model_type qwen3_omni_moe \
    --dataset "${TRAIN_DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs external_ifbench_strict \
    --num_train_epochs 1 \
    --global_batch_size 128 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 16 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 4 \
    --vllm_max_model_len 24000 \
    --max_length 8192 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --max_completion_length 8192 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --lr 1e-4 \
    --bf16 true \
    --beta 0.0 \
    --importance_sampling_level sequence \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --sleep_level 0 \
    --offload_model true \
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
    --output_dir "${OUTPUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &

echo "Started training in background."
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
    # --tuner_type lora \
    # --lora_rank 8 \
    # --lora_alpha 32 \
    # --target_modules all-linear \