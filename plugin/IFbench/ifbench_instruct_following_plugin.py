import inspect
import json
import sys
from typing import Any, Dict, List

from swift.rewards import ORM, orms

# Use IFbench checkers under this repo.
IFBENCH_DIR = '/dev/shm/ye/ms-swift/plugin/IFbench'
if IFBENCH_DIR not in sys.path:
    sys.path.insert(0, IFBENCH_DIR)

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


def _normalize_value(value: Any) -> Any:
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    return value


def _to_extra_info(extra_info: Any) -> Dict[str, Any]:
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        try:
            parsed = json.loads(extra_info)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _strict_hits(solution_str: str, instruction_id_list: List[str], kwargs_list: List[Dict[str, Any]]) -> List[bool]:
    assert len(instruction_id_list) == len(kwargs_list), 'instruction_id_list and instruction_kwargs length mismatch'
    hits: List[bool] = []
    for idx, inst_id in enumerate(instruction_id_list):
        inst_cls = instructions_registry.INSTRUCTION_DICT[inst_id]
        inst = inst_cls(inst_id)

        raw_kwargs = kwargs_list[idx] or {}
        if not isinstance(raw_kwargs, dict):
            raw_kwargs = {}
        accepted = set(inspect.signature(inst.build_description).parameters.keys())
        filtered_kwargs = {k: _normalize_value(v) for k, v in raw_kwargs.items() if k in accepted}
        inst.build_description(**filtered_kwargs)

        ok = bool(solution_str.strip()) and bool(inst.check_following(solution_str))
        hits.append(ok)
    return hits


class IFbenchStrictORM(ORM):
    """
    Strict reward with IFbench instruction checkers.
    Reward policy:
      - 1.0 if all instructions are satisfied
      - 0.0 otherwise
    """

    def __call__(self, completions, extra_info=None, **kwargs) -> List[float]:
        rewards: List[float] = []
        extra_info = extra_info or [{} for _ in completions]

        for completion, one_extra in zip(completions, extra_info):
            info = _to_extra_info(one_extra)
            instruction_id_list = [str(x) for x in _to_list(info.get('instruction_id_list', []))]
            kwargs_list = _to_list(info.get('instruction_kwargs', []))

            if len(instruction_id_list) == 0:
                rewards.append(0.0)
                continue

            # try:
            if "<think>\n\n</think>\n\n" in completion:
                completion = completion.replace("<think>\n\n</think>\n\n", "")
            print({'completion':completion, 'instruction_id_list':instruction_id_list, 'kwargs_list':kwargs_list})
            hits = _strict_hits(completion, instruction_id_list, kwargs_list)
            rewards.append(1.0 if hits and all(hits) else 0.0)
            # except Exception:
            #     rewards.append(0.0)
        return rewards


orms['external_ifbench_strict'] = IFbenchStrictORM
