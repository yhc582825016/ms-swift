#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List

import pandas as pd


def _to_messages(prompt_field: Any) -> List[Dict[str, Any]]:
    if isinstance(prompt_field, list):
        return prompt_field
    if isinstance(prompt_field, tuple):
        return list(prompt_field)
    try:
        return list(prompt_field)
    except Exception:
        return []


def _to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def convert_one(input_path: str, output_path: str, max_prompt_chars: int) -> None:
    df = pd.read_parquet(input_path)
    rows = []
    for row in df.to_dict('records'):
        messages = _to_messages(row.get('prompt', []))
        prompt_chars = sum(len((m or {}).get('content', '')) for m in messages)
        if prompt_chars > max_prompt_chars:
            continue

        reward_model = _to_dict(row.get('reward_model', {}))
        ground_truth = reward_model.get('ground_truth')
        if ground_truth is None:
            continue

        rows.append({
            'messages': messages,
            'solution': str(ground_truth),
            'data_source': row.get('data_source', ''),
            'reward_model': reward_model,
            'extra_info': _to_dict(row.get('extra_info', {})),
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f'Converted {len(rows)} rows -> {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Convert DAPO/AIME parquet to ms-swift jsonl.')
    parser.add_argument('--train_input', required=True)
    parser.add_argument('--test_input', required=True)
    parser.add_argument('--train_output', required=True)
    parser.add_argument('--test_output', required=True)
    parser.add_argument('--max_prompt_chars', type=int, default=7000)
    args = parser.parse_args()

    convert_one(args.train_input, args.train_output, args.max_prompt_chars)
    convert_one(args.test_input, args.test_output, args.max_prompt_chars)


if __name__ == '__main__':
    main()
