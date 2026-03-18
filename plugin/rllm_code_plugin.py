import json
import re
import subprocess
from typing import Any, Dict, List, Optional

from swift.rewards import ORM, orms


def _extract_code_from_model(model_response: str) -> Optional[str]:
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def _parse_reward_model(reward_model: Any) -> Dict[str, Any]:
    if isinstance(reward_model, dict):
        return reward_model
    if isinstance(reward_model, str):
        try:
            parsed = json.loads(reward_model)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_ground_truth(ground_truth: Any) -> List[Dict[str, Any]]:
    if isinstance(ground_truth, list):
        return [x for x in ground_truth if isinstance(x, dict)]
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except Exception:
            return []
    return []


def _run_one_test(code: str, test_input: str, expected_output: str, timeout_s: int = 8) -> bool:
    try:
        proc = subprocess.run(
            ['python', '-c', code],
            input=test_input,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except Exception:
        return False

    if proc.returncode != 0:
        return False
    return proc.stdout.strip() == (expected_output or '').strip()


class RLLMCodeRewardORM(ORM):
    """
    Migrated from verl/recipe/rllm/rewards/rl_reward.py (code branch).
    Returns 1.0 when generated code passes all selected tests, else 0.0.
    """

    def __call__(self, completions, data_source=None, reward_model=None, **kwargs) -> List[float]:
        rewards: List[float] = []
        data_source = data_source or [''] * len(completions)
        reward_model = reward_model or [{} for _ in completions]

        for completion, ds, rm in zip(completions, data_source, reward_model):
            if ds not in {
                    'apps', 'taco', 'code_contests', 'codeforces', 'livecodebench', 'kodcode', 'leetcode',
                    'primeintellect', 'humanevalplus'
            }:
                rewards.append(0.0)
                continue

            code = _extract_code_from_model(completion or '')
            if not code:
                rewards.append(0.0)
                continue

            rm_obj = _parse_reward_model(rm)
            tests = _parse_ground_truth(rm_obj.get('ground_truth', []))
            if not tests:
                rewards.append(0.0)
                continue

            # Keep the execution budget bounded like the verl implementation.
            if len(tests) > 15:
                tests = sorted(
                    tests, key=lambda t: len((t.get('input') or '')), reverse=True)[:15]

            ok = True
            for test in tests:
                if test.get('type') != 'stdin_stdout':
                    continue
                if not _run_one_test(code, test.get('input', ''), test.get('output', '')):
                    ok = False
                    break
            rewards.append(1.0 if ok else 0.0)
        return rewards


orms['external_rllm_code'] = RLLMCodeRewardORM
