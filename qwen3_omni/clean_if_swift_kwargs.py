#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd


VERL_IF_DIR = '/mnt/code/yehangcheng/verl/recipe/insturct_following'
if VERL_IF_DIR not in sys.path:
    sys.path.insert(0, VERL_IF_DIR)

import instructions_registry  # noqa: E402


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, 'tolist'):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


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


def _normalize_value(value: Any) -> Any:
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def _build_signature_cache() -> Dict[str, set]:
    cache: Dict[str, set] = {}
    for inst_id, inst_cls in instructions_registry.INSTRUCTION_DICT.items():
        inst = inst_cls(inst_id)
        cache[inst_id] = set(inspect.signature(inst.build_description).parameters.keys())
    return cache


def _clean_one_extra_info(extra_info: Any, accepted_cache: Dict[str, set]) -> Tuple[Dict[str, Any], int]:
    info = _to_dict(extra_info)
    instruction_id_list = [str(x) for x in _to_list(info.get('instruction_id_list', []))]
    kwargs_list = _to_list(info.get('instruction_kwargs', []))

    cleaned_kwargs: List[Dict[str, Any]] = []
    removed_keys = 0

    pair_count = min(len(instruction_id_list), len(kwargs_list))
    for idx in range(pair_count):
        inst_id = instruction_id_list[idx]
        raw_kwargs = kwargs_list[idx] if isinstance(kwargs_list[idx], dict) else {}

        accepted = accepted_cache.get(inst_id)
        if accepted is None:
            # Unknown instruction id: keep an empty kwargs dict to avoid runtime crashes.
            cleaned_kwargs.append({})
            removed_keys += len(raw_kwargs.keys())
            continue

        filtered: Dict[str, Any] = {}
        for k, v in raw_kwargs.items():
            if k in accepted and v is not None:
                filtered[k] = _normalize_value(v)
            elif k not in accepted:
                removed_keys += 1
        cleaned_kwargs.append(filtered)

    if len(instruction_id_list) > pair_count:
        cleaned_kwargs.extend({} for _ in range(len(instruction_id_list) - pair_count))

    info['instruction_id_list'] = instruction_id_list
    info['instruction_kwargs'] = cleaned_kwargs
    return info, removed_keys


def main():
    parser = argparse.ArgumentParser(description='Clean ms-swift IF parquet kwargs to match instruction signatures.')
    parser.add_argument('--input', required=True, help='Input ms-swift parquet')
    parser.add_argument('--output', required=True, help='Output cleaned parquet')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f'Input not found: {args.input}')

    df = pd.read_parquet(args.input)
    accepted_cache = _build_signature_cache()

    cleaned_rows = []
    total_removed_keys = 0
    changed_rows = 0

    for row in df.to_dict('records'):
        old_info = _to_dict(row.get('extra_info', {}))
        new_info, removed_keys = _clean_one_extra_info(old_info, accepted_cache)
        total_removed_keys += removed_keys
        if removed_keys > 0:
            changed_rows += 1
        # Keep sparse kwargs by serializing as JSON string.
        # If stored as parquet nested struct, pyarrow may re-expand null fields.
        row['extra_info'] = json.dumps(new_info, ensure_ascii=False)
        cleaned_rows.append(row)

    out_df = pd.DataFrame(cleaned_rows)
    out_df.to_parquet(args.output, index=False)

    print(f'Input rows: {len(df)}')
    print(f'Changed rows: {changed_rows}')
    print(f'Removed unexpected kwargs keys: {total_removed_keys}')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
