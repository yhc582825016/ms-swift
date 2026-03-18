#!/usr/bin/env python3
import argparse
import ast
import random
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
from datasets import load_dataset


def _normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    if isinstance(messages, list):
        return messages
    if isinstance(messages, tuple):
        return list(messages)
    if hasattr(messages, 'tolist'):
        converted = messages.tolist()
        return converted if isinstance(converted, list) else []
    return []


def _parse_ground_truth(ground_truth: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        return [], []

    parsed: Any = None
    try:
        parsed = ast.literal_eval(ground_truth)
    except Exception:
        return [], []

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, Sequence):
        return [], []

    instruction_id_list: List[str] = []
    instruction_kwargs: List[Dict[str, Any]] = []

    for item in parsed:
        if not isinstance(item, dict):
            continue
        ids = item.get('instruction_id', [])
        kwargs = item.get('kwargs', [])
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(kwargs, dict):
            kwargs = [kwargs]

        if not isinstance(ids, Sequence) or not isinstance(kwargs, Sequence):
            continue

        for inst_id, inst_kwargs in zip(ids, kwargs):
            if not isinstance(inst_id, str):
                continue
            if not isinstance(inst_kwargs, dict):
                inst_kwargs = {}
            instruction_id_list.append(inst_id)
            instruction_kwargs.append(inst_kwargs)

    return instruction_id_list, instruction_kwargs


def _convert_row(row: Dict[str, Any]) -> Dict[str, Any]:
    instruction_id_list, instruction_kwargs = _parse_ground_truth(row.get('ground_truth', ''))
    return {
        'messages': _normalize_messages(row.get('messages', [])),
        'extra_info': {
            'instruction_id_list': instruction_id_list,
            'instruction_kwargs': instruction_kwargs,
            'key': row.get('key', ''),
            'constraint_type': row.get('constraint_type', ''),
            'constraint': row.get('constraint', ''),
        },
        'data_source': row.get('dataset', 'if_multi_constraints_upto5'),
        'solution': '',
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert IF_multi_constraints_upto5 to ms-swift instruct-following parquet.'
    )
    parser.add_argument(
        '--input',
        default='/mnt/code/yehangcheng/all_data/rlhf_data/IF_multi_constraints_upto5',
        help='Dataset local path or HF dataset id.',
    )
    parser.add_argument('--train_output', required=True, help='Output train parquet path')
    parser.add_argument('--val_output', required=True, help='Output val parquet path')
    parser.add_argument('--val_size', type=int, default=2000, help='Validation size sampled from train split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    args = parser.parse_args()

    ds = load_dataset(args.input)
    rows = [_convert_row(row) for row in ds['train']]

    valid_rows = [r for r in rows if len(r['extra_info']['instruction_id_list']) > 0]
    if not valid_rows:
        raise ValueError('No valid rows with instruction_id_list found after conversion.')

    rng = random.Random(args.seed)
    indices = list(range(len(valid_rows)))
    rng.shuffle(indices)

    val_size = min(max(args.val_size, 1), len(valid_rows) - 1)
    val_idx = set(indices[:val_size])

    train_records = [valid_rows[i] for i in range(len(valid_rows)) if i not in val_idx]
    val_records = [valid_rows[i] for i in range(len(valid_rows)) if i in val_idx]

    pd.DataFrame(train_records).to_parquet(args.train_output, index=False)
    pd.DataFrame(val_records).to_parquet(args.val_output, index=False)

    print(f'Converted total: {len(valid_rows)}')
    print(f'Train: {len(train_records)} -> {args.train_output}')
    print(f'Val: {len(val_records)} -> {args.val_output}')


if __name__ == '__main__':
    main()
