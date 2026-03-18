import json
import re
from typing import Any, Dict, List, Optional

from swift.rewards import ORM, orms


class LogicRLRewardORM(ORM):
    """Reward for logic role classification with strict <answer> format."""

    _ROLE_LINE_PATTERN = re.compile(
        r'^\s*(?:\(\s*\d+\s*\)\s*)?([A-Za-z][A-Za-z\'_-]*)\s+is\s+a\s+(knight|knave)\s*[.。]?\s*$',
        re.IGNORECASE,
    )

    @staticmethod
    def _extract_last_answer(solution_str: str) -> Optional[str]:
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = list(re.finditer(answer_pattern, solution_str or '', re.DOTALL))
        if not matches:
            return None
        return matches[-1].group(1).strip()

    @staticmethod
    def _parse_solution_text_format(solution_text: str) -> Dict[str, str]:
        status_dict: Dict[str, str] = {}
        for line in (solution_text or '').split('\n'):
            line = line.strip()
            if not line:
                continue
            match = LogicRLRewardORM._ROLE_LINE_PATTERN.fullmatch(line)
            if match:
                name, role = match.groups()
                status_dict[name] = role.lower()
        return status_dict

    @staticmethod
    def _parse_model_answer(answer_text: str, expected_names: List[str]) -> Optional[Dict[str, str]]:
        # Parse only strict answer lines inside <answer>...</answer>.
        status_dict: Dict[str, str] = {}
        for raw_line in (answer_text or '').split('\n'):
            line = raw_line.strip()
            if not line:
                continue
            match = LogicRLRewardORM._ROLE_LINE_PATTERN.fullmatch(line)
            if not match:
                continue
            name, role = match.groups()
            status_dict[name] = role.lower()

        expected_set = set(expected_names)
        if set(status_dict.keys()) != expected_set:
            return None
        return status_dict

    @staticmethod
    def _validate_response_structure(processed_str: str) -> bool:
        count_answer_start = (processed_str or '').count('<answer>')
        count_answer_end = (processed_str or '').count('</answer>')
        first_start = (processed_str or '').find('<answer>')
        first_end = (processed_str or '').find('</answer>')
        if count_answer_start != 1 or count_answer_end != 1:
            return False
        if first_start == -1 or first_end == -1 or first_start > first_end:
            return False
        return True

    @staticmethod
    def _normalize_ground_truth(gt_item: Any) -> Dict[str, Any]:
        if isinstance(gt_item, dict):
            return gt_item
        if isinstance(gt_item, str):
            try:
                parsed = json.loads(gt_item)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {'solution_text_format': gt_item}
        return {}

    def _score_one(
        self,
        completion: str,
        gt_item: Any,
        format_reward: float = 1.0,
        full_match_reward: float = 2.0,
        mismatch_reward: float = -1.5,
        unparseable_reward: float = -2.0,
    ) -> float:
        gt_obj = self._normalize_ground_truth(gt_item)
        gt_status = self._parse_solution_text_format(gt_obj.get('solution_text_format', ''))
        expected_names = list(gt_status.keys())

        # 强约束：如果拿不到有效 ground truth，视为数据或输入配置有问题，直接抛错。
        # 这样可以强制你在数据/recipe 侧提供正确的 `solution_text_format`，而不是在这里默默给高分。
        if not gt_status:
            raise ValueError(
                f'LogicRLRewardORM expects non-empty ground truth `solution_text_format`, '
                f'got: {gt_item!r}'
            )

        answer_text = self._extract_last_answer(completion or '')
        format_correct = self._validate_response_structure(completion or '')
        format_score = format_reward if format_correct else -abs(format_reward)

        answer_score = 0.0
        if format_correct and answer_text:
            pred_status = self._parse_model_answer(answer_text, expected_names)
            if pred_status is None:
                answer_score = unparseable_reward
            elif pred_status == gt_status:
                answer_score = full_match_reward
            else:
                answer_score = mismatch_reward
        return format_score + answer_score

    def __call__(self, completions, reward_model=None, **kwargs) -> List[float]:
        n = len(completions)
        reward_model = reward_model or [{} for _ in range(n)]
        if not isinstance(reward_model, list):
            reward_model = [reward_model] * n

        ground_truths: List[Any] = []
        for rm_item in reward_model[:n]:
            if isinstance(rm_item, dict):
                ground_truths.append(rm_item.get('ground_truth', {}))
            else:
                ground_truths.append({})

        if len(ground_truths) < n:
            ground_truths.extend([{} for _ in range(n - len(ground_truths))])

        return [self._score_one(completion, gt_item) for completion, gt_item in zip(completions, ground_truths)]


orms['external_logic_rl_reward'] = LogicRLRewardORM
