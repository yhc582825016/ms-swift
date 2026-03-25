import json
import inspect
import re
import subprocess
import sys
from typing import Any, Dict, List

from swift.rewards import ORM, orms
from swift.rewards.orm import MathAccuracy

VERL_IF_DIR = '/mnt/code/yehangcheng/verl/recipe/insturct_following'
if VERL_IF_DIR not in sys.path:
    sys.path.insert(0, VERL_IF_DIR)

import instructions_registry  # noqa: E402


def _extract_code_from_model(model_response: str) -> str:
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', model_response, re.DOTALL)
    if len(code_blocks) == 0:
        return ''
    return code_blocks[-1].strip()


def _run_one_test(code: str, test_input: str, expected_output: str, timeout_s: int = 8) -> bool:
    proc = subprocess.run(
        ['python', '-c', code],
        input=test_input,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        return False
    return proc.stdout.strip() == expected_output.strip()


def _strict_if_reward(solution_str: str, info: Dict[str, Any]) -> float:
    instruction_id_list = info['instruction_id_list']
    kwargs_list = info['instruction_kwargs']
    assert len(instruction_id_list) == len(kwargs_list)
    for idx, inst_id in enumerate(instruction_id_list):
        inst_cls = instructions_registry.INSTRUCTION_DICT[inst_id]
        inst = inst_cls(inst_id)
        kwargs = kwargs_list[idx]
        accepted = set(inspect.signature(inst.build_description).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        inst.build_description(**filtered_kwargs)
        try:
            ok = bool(solution_str.strip()) and bool(inst.check_following(solution_str))
        except Exception:
            ok = False
        if not ok:
            return 0.0
    return 1.0


def _code_reward(completion: str, reward_model: Dict[str, Any]) -> float:
    code = _extract_code_from_model(completion)
    if code == '':
        return 0.0
    tests = reward_model['ground_truth']
    if isinstance(tests, str):
        tests = json.loads(tests)
    if len(tests) > 15:
        tests = sorted(tests, key=lambda t: len(t['input']), reverse=True)[:15]
    for test in tests:
        if test['type'] != 'stdin_stdout':
            continue
        if not _run_one_test(code, test['input'], test['output']):
            return 0.0
    return 1.0


class MultiRewardRouterORM(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.math_orm = MathAccuracy(args)

    def __call__(self, completions, reward_route, solution, extra_info, reward_model, **kwargs) -> List[float]:
        rewards = []
        math_rewards = self.math_orm(completions, solution)

        for idx, completion in enumerate(completions):
            route = reward_route[idx]
            if route == 'if':
                info = extra_info[idx]
                if isinstance(info, str):
                    info = json.loads(info)
                rewards.append(_strict_if_reward(completion, info))
            elif route == 'code':
                rm = reward_model[idx]
                if isinstance(rm, str):
                    rm = json.loads(rm)
                rewards.append(_code_reward(completion, rm))
            elif route == 'math':
                rewards.append(math_rewards[idx])
            else:
                raise ValueError(f'Unknown reward_route: {route}')
        return rewards


orms['external_multi_router'] = MultiRewardRouterORM
