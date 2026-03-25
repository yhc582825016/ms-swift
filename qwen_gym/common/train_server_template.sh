#!/usr/bin/env bash
set -xeuo pipefail
TASK_NAME="${TASK_NAME:?TASK_NAME is required}"
TRAIN_DATASET="${TRAIN_DATASET:?TRAIN_DATASET is required}"
VAL_DATASET="${VAL_DATASET:?VAL_DATASET is required}"
NEMO_GYM_VERIFY_URL="${NEMO_GYM_VERIFY_URL:?NEMO_GYM_VERIFY_URL is required}"

export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-swift}"
export SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-qwen3-4b-${TASK_NAME}-gym}"

MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/mnt/code/yehangcheng/ms-swift}"
MODEL_PATH="${MODEL_PATH:-/opt/users/ye/checkpoints/Qwen3-4B-Instruct-2507-general_total_425w_no_ace_reason_cat_online_datas_457w_shuffle/checkpoint-2337}"
ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29690}"
cd "${MS_SWIFT_ROOT}"
mkdir -p "${MS_SWIFT_ROOT}/megatron_output/logs"
export NEMO_GYM_VERIFY_URL
export NEMO_GYM_SEED_SESSION_URL="${NEMO_GYM_SEED_SESSION_URL:-${NEMO_GYM_VERIFY_URL%/verify}/seed_session}"

PYTORCH_ALLOC_CONF='expandable_segments:True' \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
nohup swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  --dataset "${TRAIN_DATASET}" \
  --val_dataset "${VAL_DATASET}" \
  --split_dataset_ratio 0 \
  --tuner_type full \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host "${ROLLOUT_HOST}" \
  --vllm_server_port "${ROLLOUT_PORT}" \
  --vllm_server_pass_dataset true \
  --deepspeed zero2 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
  --num_generations "${NUM_GENERATIONS:-8}" \
  --steps_per_generation "${STEPS_PER_GENERATION:-2}" \
  --max_length "${MAX_LENGTH:-8192}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-4096}" \
  --learning_rate "${LEARNING_RATE:-1e-6}" \
  --beta 0.0 \
  --loss_type "${LOSS_TYPE:-cispo}" \
  --epsilon "${EPSILON:-0.2}" \
  --epsilon_high "${EPSILON_HIGH:-5.0}" \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k -1 \
  --warmup_ratio 0.03 \
  --logging_steps 1 \
  --eval_steps "${EVAL_STEPS:-50}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --save_total_limit 2 \
  --dataloader_num_workers 8 \
  --dataset_num_proc 8 \
  --log_completions true \
  --report_to swanlab \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_exp_name "${SWANLAB_EXPERIMENT_NAME}" \
  --output_dir "${MS_SWIFT_ROOT}/megatron_output/qwen3_4b_${TASK_NAME}_gym" \
  > "${MS_SWIFT_ROOT}/megatron_output/logs/qwen3_4b_${TASK_NAME}_gym.log" 2>&1 &
