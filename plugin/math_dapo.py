import json
import re
from typing import Any, Dict, List, Optional

from swift.rewards import ORM, orms


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind('\\boxed{')
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    left = '\\boxed{'
    assert s[:len(left)] == left, f'box error: {s}'
    assert s[-1] == '}', f'box error: {s}'
    return s[len(left):-1]


SUBSTITUTIONS = [
    ('an ', ''),
    ('a ', ''),
    ('.$', '$'),
    ('\\$', ''),
    (r'\ ', ''),
    (' ', ''),
    ('mbox', 'text'),
    (',\\text{and}', ','),
    ('\\text{and}', ','),
    ('\\text{m}', '\\text{}'),
]

REMOVED_EXPRESSIONS = [
    'square',
    'ways',
    'integers',
    'dollars',
    'mph',
    'inches',
    'hours',
    'km',
    'units',
    '\\ldots',
    'sue',
    'points',
    'feet',
    'minutes',
    'digits',
    'cents',
    'degrees',
    'cm',
    'gm',
    'pounds',
    'meters',
    'meals',
    'edges',
    'students',
    'childrentickets',
    'multiples',
    '\\text{s}',
    '\\text{.}',
    '\\text{\ns}',
    '\\text{}^2',
    '\\text{}^3',
    '\\text{\n}',
    '\\text{}',
    r'\mathrm{th}',
    r'^\circ',
    r'^{\circ}',
    r'\;',
    r',\!',
    '{,}',
    '"',
    '\\dots',
]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split('=')[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')
    return final_answer.strip()


def is_correct_minerva(
    solution_str: str,
    gt: str,
    gt_need_extract: bool = False,
    answer_pattern: str = r'(?i)Answer\s*:\s*([^\n]+)',
) -> tuple[bool, str]:
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else '[INVALID]'
    pred = normalize_final_answer(extracted_answer)
    if gt_need_extract:
        boxed_gt = last_boxed_only_string(gt)
        if boxed_gt is None:
            return False, pred
        gt = normalize_final_answer(remove_boxed(boxed_gt))
    else:
        gt = normalize_final_answer(gt)
    return pred == gt, pred


def is_correct_strict_box(
    pred: str,
    gt: str,
    pause_tokens_index: Optional[List[int]] = None,
) -> tuple[int, Optional[str]]:
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100:]
    else:
        pred = pred[-100:]
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None
    return (1 if extracted_pred == gt else -1), extracted_pred


def verify(
    solution_str: str,
    answer: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[List[int]] = None,
) -> tuple[bool, Optional[str]]:
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred
    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[List[int]] = None,
) -> Dict[str, Any]:
    solution_str = (solution_str or '')[-300:]
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)
    reward = 1.0 if correct else -1.0
    return {'score': reward, 'acc': correct, 'pred': pred}


class MathDAPORewardORM(ORM):
    """Math DAPO style reward: +1 for correct, -1 for incorrect."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.default_strict_box_verify = bool(getattr(args, 'strict_box_verify', False)) if args is not None else False

    @staticmethod
    def _to_list(value: Any, n: int, default: Any = None) -> List[Any]:
        if value is None:
            return [default] * n
        if isinstance(value, list):
            if len(value) < n:
                return value + [default] * (n - len(value))
            return value[:n]
        return [value] * n

    @staticmethod
    def _parse_reward_model_item(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def __call__(
        self,
        completions,
        solution=None,
        ground_truth=None,
        answer=None,
        reward_model=None,
        strict_box_verify=None,
        pause_tokens_index=None,
        **kwargs,
    ) -> List[float]:
        n = len(completions)
        solution = self._to_list(solution, n)
        ground_truth = self._to_list(ground_truth, n)
        answer = self._to_list(answer, n)
        reward_model = self._to_list(reward_model, n, default={})
        strict_box_verify = self._to_list(
            strict_box_verify if strict_box_verify is not None else self.default_strict_box_verify, n, default=False)
        pause_tokens_index = self._to_list(pause_tokens_index, n, default=None)

        rewards: List[float] = []
        for completion, sol, gt, ans, rm, strict, pause in zip(
                completions, solution, ground_truth, answer, reward_model, strict_box_verify, pause_tokens_index):
            rm_obj = self._parse_reward_model_item(rm)
            cur_gt = gt if gt not in (None, '') else (sol if sol not in (None, '') else ans)
            if cur_gt in (None, ''):
                cur_gt = rm_obj.get('ground_truth') or rm_obj.get('solution') or rm_obj.get('answer') or ''
            if isinstance(rm_obj.get('strict_box_verify'), bool):
                strict = rm_obj['strict_box_verify']
            if pause is None and isinstance(rm_obj.get('pause_tokens_index'), list):
                pause = rm_obj['pause_tokens_index']

            try:
                score_info = compute_score(completion or '', str(cur_gt), bool(strict), pause)
                # Print last 100 characters (tokens) of completion and the related answers for debugging
                print({
                    # 'solution': sol,
                    "completion": (completion or '')[-100:],
                    'ground_truth': str(cur_gt),
                    'extract_answer': score_info.get('pred'),
                })
                rewards.append(score_info['score'])
            except Exception:
                rewards.append(-1.0)
        return rewards


orms['external_math_dapo'] = MathDAPORewardORM
