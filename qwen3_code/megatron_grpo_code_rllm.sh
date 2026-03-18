#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MS_SWIFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PLUGIN_PATH="${MS_SWIFT_ROOT}/examples/train/grpo/plugin/rllm_code_plugin.py"
TRAIN_DATASET="${MS_SWIFT_ROOT}/megatron_output/deepcoder_train_swift_len7k.jsonl"
MASTER_PORT="${MASTER_PORT:-29621}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model /opt/users/models/Qwen3-4B-Instruct-2507 \
    --model_type qwen3 \
    --dataset "${TRAIN_DATASET}#1000" \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs external_rllm_code \
    --num_train_epochs 1 \
    --global_batch_size 16 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 2 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.68 \
    --vllm_tensor_parallel_size 4 \
    --vllm_max_model_len 4096 \
    --max_length 4096 \
    --max_completion_length 512 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --lr 5e-5 \
    --bf16 true \
    --beta 0.001 \
    --importance_sampling_level sequence \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --sleep_level 2 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
    --save_steps 100 \
    --eval_steps 100 \
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
    --padding_free true \
    --sequence_parallel true \
    --log_completions true \
    --report_to tensorboard \
    --output_dir "${MS_SWIFT_ROOT}/megatron_output/Qwen3-4B-Instruct-2507-grpo-code-rllm"
