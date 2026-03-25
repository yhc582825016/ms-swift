#!/usr/bin/env bash
set -xeuo pipefail
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/mnt/code/yehangcheng/ms-swift}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-/opt/users/ye/checkpoints/Qwen3-4B-Instruct-2507-general_total_425w_no_ace_reason_cat_online_datas_457w_shuffle/checkpoint-2337}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8000}"
cd "${MS_SWIFT_ROOT}"
export NEMO_GYM_VERIFY_URL="${NEMO_GYM_VERIFY_URL:-http://127.0.0.1:18001/verify}"
export NEMO_GYM_SEED_SESSION_URL="${NEMO_GYM_SEED_SESSION_URL:-http://127.0.0.1:18001/seed_session}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
swift rollout \
  --model "${ROLLOUT_MODEL}" \
  --host 0.0.0.0 \
  --port "${ROLLOUT_PORT}" \
  --multi_turn_scheduler gym_scheduler \
  --use_gym_env true \
  --gym_env nemo_gym_env \
  --max_turns 1 \
  --vllm_use_async_engine true \
  --vllm_gpu_memory_utilization 0.75 \
  --vllm_max_model_len 32768
