#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List

import pandas as pd


def _normalize_messages(prompt_field: Any) -> List[Dict[str, Any]]:
    if isinstance(prompt_field, list):
        return prompt_field
    if isinstance(prompt_field, tuple):
        return list(prompt_field)
    # verl parquet often stores prompt as numpy object array
    try:
        return list(prompt_field)
    except Exception:
        return []


def _normalize_extra_info(extra_info: Any) -> Dict[str, Any]:
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        try:
            parsed = json.loads(extra_info)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def main():
    parser = argparse.ArgumentParser(description='Convert verl instruct_following parquet to ms-swift parquet.')
    parser.add_argument('--input', required=True, help='Input verl parquet path')
    parser.add_argument('--output', required=True, help='Output ms-swift parquet path')
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    records = []
    for row in df.to_dict('records'):
        messages = _normalize_messages(row.get('prompt', []))
        extra_info = _normalize_extra_info(row.get('extra_info', {}))

        records.append({
            'messages': messages,
            'extra_info': extra_info,
            'data_source': row.get('data_source', 'if'),
            'solution': '',  # placeholder for compatibility with generic reward pipelines
        })

    out_df = pd.DataFrame(records)
    out_df.to_parquet(args.output, index=False)
    print(f'Converted {len(out_df)} rows -> {args.output}')
    print(f'Columns: {list(out_df.columns)}')


if __name__ == '__main__':
    main()
