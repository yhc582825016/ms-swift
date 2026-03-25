#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Dict, Any

from huggingface_hub import hf_hub_download

MS_SWIFT_ROOT = Path('/mnt/code/yehangcheng/ms-swift')
GYM_ROOT = Path('/mnt/code/yehangcheng/Gym')
OUT_ROOT = MS_SWIFT_ROOT / 'qwen_gym' / 'data'


def _read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_hf_jsonl(repo_id: str, filename: str):
    fpath = _hf_download(repo_id=repo_id, filename=filename)
    yield from _read_jsonl(fpath)


def _write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _inject_env(row: Dict[str, Any], prompt_key: str = 'prompt'):
    row = dict(row)
    env_config = dict(row.get('env_config') or {})
    env_config.setdefault('name', 'nemo_gym_env')
    env_config.setdefault('prompt_key', prompt_key)
    env_config.setdefault('reward_key', 'reward')
    env_config.setdefault('done_on_verify', True)
    row['env_config'] = env_config
    return row


def _hf_download(repo_id: str, filename: str) -> Path:
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset'))
    except Exception as first_error:
        mirror = os.environ.get('HF_ENDPOINT_FALLBACK', 'https://hf-mirror.com')
        old_endpoint = os.environ.get('HF_ENDPOINT')
        if old_endpoint != mirror:
            os.environ['HF_ENDPOINT'] = mirror
            try:
                print(f'[prepare_all_gym_data] retry with HF mirror: {mirror} for {repo_id}/{filename}')
                return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset'))
            except Exception:
                pass
        raise first_error


def prepare_math():
    train_src = MS_SWIFT_ROOT / 'megatron_output' / 'dapo_math_17k_swift_nemo_gym.jsonl'
    val_src = MS_SWIFT_ROOT / 'megatron_output' / 'aime_2024_swift_nemo_gym.jsonl'
    if not train_src.exists():
        raise FileNotFoundError(train_src)
    if not val_src.exists():
        raise FileNotFoundError(val_src)
    train_rows = list(_read_jsonl(train_src))
    val_rows = list(_read_jsonl(val_src))
    _write_jsonl(train_rows, OUT_ROOT / 'math' / 'train.jsonl')
    _write_jsonl(val_rows, OUT_ROOT / 'math' / 'val.jsonl')
    print(f'math: train={len(train_rows)}, val={len(val_rows)}')


def prepare_instruction_following():
    try:
        rows = [_inject_env(r, prompt_key='prompt') for r in _read_hf_jsonl(
            'nvidia/Nemotron-RL-instruction_following', 'instruction_following.jsonl')]
        src_name = 'hf:nvidia/Nemotron-RL-instruction_following/instruction_following.jsonl'
    except Exception:
        src_example = GYM_ROOT / 'resources_servers' / 'instruction_following' / 'data' / 'example.jsonl'
        rows = [_inject_env(r, prompt_key='prompt') for r in _read_jsonl(src_example)]
        src_name = f'local:{src_example.name}'
    _write_jsonl(rows, OUT_ROOT / 'instruction_following' / 'train.jsonl')
    _write_jsonl(rows[: min(200, len(rows))], OUT_ROOT / 'instruction_following' / 'val.jsonl')
    print(f'instruction_following: src={src_name}, train={len(rows)}, val={min(200, len(rows))}')


def prepare_agent():
    try:
        rows = [_inject_env(r, prompt_key='prompt') for r in _read_hf_jsonl(
            'nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1', 'train.jsonl')]
        src_name = 'hf:nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1/train.jsonl'
    except Exception:
        src_example = GYM_ROOT / 'resources_servers' / 'single_step_tool_use_with_argument_comparison' / 'data' / 'example.jsonl'
        rows = [_inject_env(r, prompt_key='prompt') for r in _read_jsonl(src_example)]
        src_name = f'local:{src_example.name}'
    _write_jsonl(rows, OUT_ROOT / 'agent' / 'train.jsonl')
    _write_jsonl(rows[: min(200, len(rows))], OUT_ROOT / 'agent' / 'val.jsonl')
    print(f'agent: src={src_name}, train={len(rows)}, val={min(200, len(rows))}')


def prepare_swe_agent():
    try:
        rows = [_inject_env(r, prompt_key='prompt') for r in _read_hf_jsonl(
            'nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1', 'train.jsonl')]
        src_name = 'hf:nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1/train.jsonl'
    except Exception:
        src_example = GYM_ROOT / 'resources_servers' / 'single_step_tool_use_with_argument_comparison' / 'data' / 'example.jsonl'
        rows = [_inject_env(r, prompt_key='prompt') for r in _read_jsonl(src_example)]
        src_name = f'local:{src_example.name}'
    _write_jsonl(rows, OUT_ROOT / 'swe_agent' / 'train.jsonl')
    _write_jsonl(rows[: min(200, len(rows))], OUT_ROOT / 'swe_agent' / 'val.jsonl')
    print(f'swe_agent: src={src_name}, train={len(rows)}, val={min(200, len(rows))}')


def prepare_workplace_assistant():
    train_rows = [_inject_env(r, prompt_key='prompt') for r in _read_hf_jsonl(
        'nvidia/Nemotron-RL-agent-workplace_assistant', 'train.jsonl')]
    val_rows = [_inject_env(r, prompt_key='prompt') for r in _read_hf_jsonl(
        'nvidia/Nemotron-RL-agent-workplace_assistant', 'validation.jsonl')]
    _write_jsonl(train_rows, OUT_ROOT / 'workplace_assistant' / 'train.jsonl')
    _write_jsonl(val_rows, OUT_ROOT / 'workplace_assistant' / 'val.jsonl')
    print(f'workplace_assistant: src=hf:nvidia/Nemotron-RL-agent-workplace_assistant, '
          f'train={len(train_rows)}, val={len(val_rows)}')


def prepare_calendar():
    train_rows = [_inject_env(r, prompt_key='prompt') for r in _read_hf_jsonl(
        'nvidia/Nemotron-RL-agent-calendar_scheduling', 'train.jsonl')]
    val_rows = [_inject_env(r, prompt_key='prompt') for r in _read_hf_jsonl(
        'nvidia/Nemotron-RL-agent-calendar_scheduling', 'validation.jsonl')]
    _write_jsonl(train_rows, OUT_ROOT / 'calendar' / 'train.jsonl')
    _write_jsonl(val_rows, OUT_ROOT / 'calendar' / 'val.jsonl')
    print(f'calendar: src=hf:nvidia/Nemotron-RL-agent-calendar_scheduling, '
          f'train={len(train_rows)}, val={len(val_rows)}')


if __name__ == '__main__':
    prepare_math()
    prepare_instruction_following()
    prepare_agent()
    prepare_swe_agent()
    prepare_workplace_assistant()
    prepare_calendar()
