#!/usr/bin/env bash
set -xeuo pipefail
export TASK_NAME=math
export TRAIN_DATASET="${TRAIN_DATASET:-/mnt/code/yehangcheng/ms-swift/qwen_gym/data/math/train.jsonl}"
export VAL_DATASET="${VAL_DATASET:-/mnt/code/yehangcheng/ms-swift/qwen_gym/data/math/val.jsonl}"
export NEMO_GYM_VERIFY_URL="${NEMO_GYM_VERIFY_URL:-http://127.0.0.1:18001/verify}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-24000}"
bash /mnt/code/yehangcheng/ms-swift/qwen_gym/common/train_server_template.sh
