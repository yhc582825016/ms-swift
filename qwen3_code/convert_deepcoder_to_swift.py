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


def _to_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def main():
    parser = argparse.ArgumentParser(description='Convert deepcoder parquet to ms-swift GRPO parquet.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_prompt_chars', type=int, default=7000)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    rows = []
    for row in df.to_dict('records'):
        messages = _to_messages(row.get('prompt', []))
        prompt_chars = sum(len((m or {}).get('content', '')) for m in messages)
        if prompt_chars > args.max_prompt_chars:
            continue

        rows.append({
            'messages': messages,
            'data_source': row.get('data_source', ''),
            'reward_model': _to_dict(row.get('reward_model', {})),
            'extra_info': _to_dict(row.get('extra_info', {})),
            'solution': '',
        })

    if args.output.endswith('.jsonl'):
        with open(args.output, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'Converted rows: {len(rows)}')
        print('Format: jsonl')
        if rows:
            print(f'Columns: {list(rows[0].keys())}')
    else:
        out_df = pd.DataFrame(rows)
        out_df.to_parquet(args.output, index=False)
        print(f'Converted rows: {len(out_df)}')
        print(f'Columns: {list(out_df.columns)}')


if __name__ == '__main__':
    main()
