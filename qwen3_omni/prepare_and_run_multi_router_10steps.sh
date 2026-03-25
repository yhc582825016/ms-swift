#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MS_SWIFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

IF_DATASET="${MS_SWIFT_ROOT}/megatron_output/if_rl_dataset_train_swift_len12k.parquet"
MATH_DATASET="${MS_SWIFT_ROOT}/megatron_output/dapo_math_17k_swift.jsonl"
CODE_DATASET="${MS_SWIFT_ROOT}/megatron_output/deepcoder_train_swift_len7k.jsonl"
MERGED_DATASET="${MS_SWIFT_ROOT}/megatron_output/multi_router_train.jsonl"
PLUGIN_PATH="${MS_SWIFT_ROOT}/examples/train/grpo/plugin/multi_reward_router_plugin.py"
OUTPUT_DIR="${MS_SWIFT_ROOT}/megatron_output/Qwen3-Omni-30B-A3B-Instruct-grpo-multi-router"
MASTER_PORT="${MASTER_PORT:-29691}"

if [[ "${REBUILD_MERGED_DATASET:-0}" == "1" || ! -f "${MERGED_DATASET}" ]]; then
python - <<'PY'
import json
import pandas as pd

if_path = '/mnt/code/yehangcheng/github/ms-swift/megatron_output/if_rl_dataset_train_swift_len12k.parquet'
math_path = '/mnt/code/yehangcheng/github/ms-swift/megatron_output/dapo_math_17k_swift.jsonl'
code_path = '/mnt/code/yehangcheng/github/ms-swift/megatron_output/deepcoder_train_swift_len7k.jsonl'
out_path = '/mnt/code/yehangcheng/github/ms-swift/megatron_output/multi_router_train.jsonl'

rows = []


def norm_messages(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return list(x)


def to_builtin(x):
    if isinstance(x, dict):
        return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_builtin(v) for v in x]
    if isinstance(x, tuple):
        return [to_builtin(v) for v in x]
    if hasattr(x, 'tolist'):
        return to_builtin(x.tolist())
    if hasattr(x, 'item'):
        return x.item()
    return x

# instruct_following parquet -> route if
df_if = pd.read_parquet(if_path)
for row in df_if.to_dict('records')[:600]:
    rows.append({
        'messages': norm_messages(row['messages']),
        'solution': row.get('solution', ''),
        'extra_info': json.dumps(to_builtin(row['extra_info']), ensure_ascii=False),
        'reward_model': json.dumps({}, ensure_ascii=False),
        'reward_route': 'if',
        'data_source': row.get('data_source', 'if'),
    })

# dapo math jsonl -> route math
with open(math_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 600:
            break
        row = json.loads(line)
        rows.append({
            'messages': norm_messages(row['messages']),
            'solution': row['solution'],
            'extra_info': json.dumps(to_builtin(row.get('extra_info', {})), ensure_ascii=False),
            'reward_model': json.dumps(to_builtin(row.get('reward_model', {})), ensure_ascii=False),
            'reward_route': 'math',
            'data_source': row.get('data_source', 'math_dapo'),
        })

# deepcoder jsonl -> route code
with open(code_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 600:
            break
        row = json.loads(line)
        rows.append({
            'messages': norm_messages(row['messages']),
            'solution': row.get('solution', ''),
            'extra_info': json.dumps(to_builtin(row.get('extra_info', {})), ensure_ascii=False),
            'reward_model': json.dumps(to_builtin(row['reward_model']), ensure_ascii=False),
            'reward_route': 'code',
            'data_source': row.get('data_source', 'primeintellect'),
        })

with open(out_path, 'w', encoding='utf-8') as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(f'Wrote {len(rows)} rows to {out_path}')
PY
else
    echo "Use existing merged dataset: ${MERGED_DATASET}"
fi

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
ENABLE_AUDIO_OUTPUT=1 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model /opt/users/models/Qwen3-Omni-30B-A3B-Instruct \
    --model_type qwen3_omni_moe \
    --dataset "${MERGED_DATASET}#1800" \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs external_multi_router \
    --num_train_epochs 1 \
    --global_batch_size 16 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 2 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 4 \
    --vllm_max_model_len 24576 \
    --max_length 24576 \
    --max_completion_length 12000 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
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
    --output_dir "${OUTPUT_DIR}"
