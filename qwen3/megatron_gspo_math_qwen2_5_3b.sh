set -xeuo pipefail
# rDo9t8YJf2yXZO8ilEyWZ
export SWANLAB_MODE="cloud"
export SWANLAB_API_KEY="rDo9t8YJf2yXZO8ilEyWZ"
export SWANLAB_PROJECT="swift"
export SWANLAB_EXPERIMENT_NAME="Qwen2.5-3B-Instruct-gspo-logic-rl"
# Megatron + GSPO LoRA finetuning for logic-rl reward in ms-swift.
# Run from any directory.
cd /mnt/code/yehangcheng/ms-swift
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MS_SWIFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MS_SWIFT_ROOT=/mnt/code/yehangcheng/ms-swift
TRAIN_DATASET="/mnt/code/yehangcheng/verl/recipe/logic_rl/data/train.parquet"
VAL_DATASET="/mnt/code/yehangcheng/verl/recipe/logic_rl/data/test.parquet"
MASTER_PORT="${MASTER_PORT:-29611}"
PYTORCH_ALLOC_CONF='expandable_segments:True' \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron rlhf \
    --rlhf_type grpo \
    --model /opt/users/models/Qwen2.5-3B-Instruct \
    --model_type qwen2 \
    --template qwen2_5 \
    --dataset "${TRAIN_DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --external_plugins "/mnt/code/yehangcheng/ms-swift/examples/train/grpo/plugin/logic_rl_reward_plugin.py" \
    --reward_funcs external_logic_rl_reward \
    --columns '{"prompt":"messages"}' \
    --num_train_epochs 10 \
    --global_batch_size 128 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 8 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 24000 \
    --max_length 8192 \
    --max_completion_length 12000 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --lr 5e-6 \
    --lr_decay_style constant \
    --lr_warmup_iters 0 \
    --bf16 true \
    --beta 0.0 \
    --importance_sampling_level sequence \
    --epsilon 0.0003 \
    --epsilon_high 0.0004 \
    --dynamic_sample false \
    --overlong_filter false \
    --loss_type grpo \
    --sleep_level 1 \
    --offload_model false \
    --offload_bridge false \
    --offload_optimizer false \
    --save_steps 1000000 \
    --eval_steps 25 \
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
    --swanlab_project "${SWANLAB_PROJECT}" \
    --swanlab_exp_name "${SWANLAB_EXPERIMENT_NAME}" \
    --report_to tensorboard \
    --output_dir "${MS_SWIFT_ROOT}/megatron_output/Qwen2.5-3B-Instruct-gspo-logic-rl" \
    > "/mnt/code/yehangcheng/ms-swift/megatron_output/logs/Qwen2.5-3B-Instruct-gspo-logic-rl.log" 2>&1 &
    #  tensorboard --logdir /mnt/code/yehangcheng/ms-swift/megatron_output/Qwen3-Omni-30B-A3B-Instruct-gspo-if-lora --host 0.0.0.0 --port 6006
# tensorboard --logdir /mnt/code/yehangcheng/ms-swift/megatron_output/Qwen2.5-3B-Instruct-gspo-logic-rl/v3-20260312-182639/runs --port 6007 --bind_all