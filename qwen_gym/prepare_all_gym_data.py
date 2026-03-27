#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Dict, Any

from huggingface_hub import hf_hub_download

MS_SWIFT_ROOT = Path('/dev/shm/ye/ms-swift')
GYM_ROOT = Path('/dev/shm/ye/Gym')
OUT_ROOT = MS_SWIFT_ROOT / 'data'


def _get_hf_endpoint() -> str:
    return os.environ.get('HF_ENDPOINT', os.environ.get('HF_ENDPOINT_FALLBACK', 'https://hf-mirror.com'))


def _get_hf_cache_dir() -> Path:
    return Path(os.environ.get('HF_CACHE_DIR', str(MS_SWIFT_ROOT / '.hf_cache')))


def _use_hf_runtime() -> tuple[str, Path]:
    endpoint = _get_hf_endpoint()
    cache_dir = _get_hf_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['HF_ENDPOINT'] = endpoint
    os.environ.setdefault('HF_HOME', str(cache_dir))
    os.environ.setdefault('HF_HUB_CACHE', str(cache_dir / 'hub'))
    os.environ.setdefault('HF_DATASETS_CACHE', str(cache_dir / 'datasets'))
    return endpoint, cache_dir


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
    endpoint, cache_dir = _use_hf_runtime()
    print(f'[prepare_all_gym_data] use HF endpoint: {endpoint} for {repo_id}/{filename}')
    return Path(hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type='dataset', endpoint=endpoint, cache_dir=str(cache_dir)))


def _load_local_dataset(path: Path, split: str = 'train'):
    from datasets import load_dataset
    _, cache_dir = _use_hf_runtime()
    if path.suffix == '.parquet':
        return load_dataset('parquet', data_files=str(path), split=split, cache_dir=str(cache_dir))
    if path.suffix in {'.jsonl', '.json'}:
        return load_dataset('json', data_files=str(path), split=split, cache_dir=str(cache_dir))
    raise ValueError(f'Unsupported dataset file format: {path}')


def _make_math_prompt(question: str) -> str:
    return ('Solve the following math problem. Make sure to put the answer '
            f'(and only answer) inside \\boxed{{}}.\n\n{question}')


def _format_math_train_row(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = row['prompt']
    if isinstance(prompt, list) and prompt:
        prompt_text = prompt[0]['content']
        responses_input = prompt
    else:
        prompt_text = str(prompt)
        responses_input = [{'role': 'user', 'content': prompt_text}]
    return {
        'prompt': prompt_text,
        'question': prompt_text,
        'expected_answer': str(row['reward_model']['ground_truth']),
        'responses_create_params': {'input': responses_input},
    }


def _format_math_val_row(row: Dict[str, Any]) -> Dict[str, Any]:
    question = str(row['problem'])
    prompt = _make_math_prompt(question)
    return {
        'prompt': prompt,
        'question': question,
        'expected_answer': str(row['answer']),
        'responses_create_params': {'input': [{'role': 'user', 'content': prompt}]},
    }


def prepare_math():
    try:
        train_src = _hf_download('YouJiacheng/DAPO-Math-17k-dedup', 'distinct-prompts-with-rewards.parquet')
        val_src = _hf_download('HuggingFaceH4/aime_2024', 'data/train-00000-of-00001.parquet')
        train_rows = [_inject_env(_format_math_train_row(r), prompt_key='prompt')
                      for r in _load_local_dataset(train_src, split='train')]
        val_rows = [_inject_env(_format_math_val_row(r), prompt_key='prompt')
                    for r in _load_local_dataset(val_src, split='train')]
        src_name = 'hf:YouJiacheng/DAPO-Math-17k-dedup + HuggingFaceH4/aime_2024'
    except Exception:
        src_example = GYM_ROOT / 'resources_servers' / 'math_with_judge' / 'data' / 'example.jsonl'
        rows = [_inject_env(dict(r, prompt=r.get('question') or _make_math_prompt(str(r.get('problem', '')))),
                            prompt_key='prompt') for r in _read_jsonl(src_example)]
        train_rows = rows
        val_rows = rows[: min(200, len(rows))]
        src_name = f'local:{src_example.name}'
    _write_jsonl(train_rows, OUT_ROOT / 'math' / 'train.jsonl')
    _write_jsonl(val_rows, OUT_ROOT / 'math' / 'val.jsonl')
    print(f'math: src={src_name}, train={len(train_rows)}, val={len(val_rows)}')


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
    # prepare_math()
    prepare_instruction_following()
    # prepare_agent()
    # prepare_swe_agent()
    # prepare_workplace_assistant()
    # prepare_calendar()
