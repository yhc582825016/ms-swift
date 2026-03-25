#!/usr/bin/env bash
set -xeuo pipefail
export TASK_NAME=instruction_following
export TRAIN_DATASET="${TRAIN_DATASET:-/mnt/code/yehangcheng/ms-swift/qwen_gym/data/instruction_following/train.jsonl}"
export VAL_DATASET="${VAL_DATASET:-/mnt/code/yehangcheng/ms-swift/qwen_gym/data/instruction_following/val.jsonl}"
export NEMO_GYM_VERIFY_URL="${NEMO_GYM_VERIFY_URL:-http://127.0.0.1:18002/verify}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
export LOSS_TYPE="${LOSS_TYPE:-grpo}"
bash /mnt/code/yehangcheng/ms-swift/qwen_gym/common/train_server_template.sh
