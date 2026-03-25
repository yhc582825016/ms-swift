#!/usr/bin/env bash
set -xeuo pipefail
export TASK_NAME=agent
export TRAIN_DATASET="${TRAIN_DATASET:-/mnt/code/yehangcheng/ms-swift/qwen_gym/data/agent/train.jsonl}"
export VAL_DATASET="${VAL_DATASET:-/mnt/code/yehangcheng/ms-swift/qwen_gym/data/agent/val.jsonl}"
export NEMO_GYM_VERIFY_URL="${NEMO_GYM_VERIFY_URL:-http://127.0.0.1:18003/verify}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-8192}"
export LOSS_TYPE="${LOSS_TYPE:-grpo}"
bash /mnt/code/yehangcheng/ms-swift/qwen_gym/common/train_server_template.sh
