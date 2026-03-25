import json
import re
from typing import Any, Dict, List, Optional

from swift.rewards import ORM, orms


class OpenRLRewardORM(ORM):
    """Open-RL reward: +1 for correct final answer, -1 for incorrect."""

    _ANSWER_TAG_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.IGNORECASE | re.DOTALL)
    _ANSWER_LINE_PATTERN = re.compile(r'(?is)(?:final\s+)?answer\s*:\s*(.+)$')
    _BOXED_PATTERN = re.compile(r'\\boxed\{([^{}]+)\}')

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        text = (text or '').strip()
        if not text:
            return ''

        tag_matches = OpenRLRewardORM._ANSWER_TAG_PATTERN.findall(text)
        if tag_matches:
            return tag_matches[-1].strip()

        line_match = OpenRLRewardORM._ANSWER_LINE_PATTERN.search(text)
        if line_match:
            return line_match.group(1).strip().splitlines()[0].strip()

        boxed = OpenRLRewardORM._BOXED_PATTERN.findall(text[-800:])
        if boxed:
            return boxed[-1].strip()

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else text

    @staticmethod
    def _strip_latex_wrappers(text: str) -> str:
        text = text.strip()
        wrappers = [
            (r'^\\\((.*)\\\)$', r'\1'),
            (r'^\\\[(.*)\\\]$', r'\1'),
            (r'^\$(.*)\$$', r'\1'),
            (r'^\\boxed\{(.*)\}$', r'\1'),
            (r'^\\text\{(.*)\}$', r'\1'),
        ]
        for pattern, repl in wrappers:
            text = re.sub(pattern, repl, text, flags=re.DOTALL).strip()
        return text

    @staticmethod
    def _normalize(text: str) -> str:
        text = OpenRLRewardORM._strip_latex_wrappers(text)
        text = text.replace('\\,', '').replace('\\!', '').replace('\\;', '')
        text = re.sub(r'\\left|\\right', '', text)
        text = re.sub(r'\s+', '', text)
        text = text.rstrip('.。;；,，')
        return text.lower()

    @staticmethod
    def _to_float(text: str) -> Optional[float]:
        cleaned = text.replace(',', '').strip()
        cleaned = re.sub(r'(?i)\s*(g/mol|mol|kg|cm|mm|m|s|h|hours|hour|degrees|°c|k)$', '', cleaned).strip()
        if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', cleaned):
            return None
        try:
            return float(cleaned)
        except Exception:
            return None

    @staticmethod
    def _get_ground_truth(solution: Any, answer: Any, reward_model_item: Any) -> str:
        if isinstance(solution, str) and solution.strip():
            return solution.strip()
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        rm_obj: Dict[str, Any] = {}
        if isinstance(reward_model_item, dict):
            rm_obj = reward_model_item
        elif isinstance(reward_model_item, str):
            try:
                parsed = json.loads(reward_model_item)
                if isinstance(parsed, dict):
                    rm_obj = parsed
            except Exception:
                rm_obj = {}

        for key in ('ground_truth', 'solution', 'answer'):
            value = rm_obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ''

    def __call__(self, completions, solution=None, answer=None, reward_model=None, **kwargs) -> List[float]:
        n = len(completions)
        if not isinstance(solution, list):
            solution = [solution] * n
        if not isinstance(answer, list):
            answer = [answer] * n
        if not isinstance(reward_model, list):
            reward_model = [reward_model] * n

        rewards: List[float] = []
        for completion, sol, ans, rm in zip(completions, solution, answer, reward_model):
            gt = self._get_ground_truth(sol, ans, rm)
            pred_raw = self._extract_final_answer(completion or '')
            gt_raw = self._extract_final_answer(gt)

            pred_norm = self._normalize(pred_raw)
            gt_norm = self._normalize(gt_raw)

            correct = False
            if pred_norm and gt_norm and pred_norm == gt_norm:
                correct = True
            else:
                pred_num = self._to_float(pred_norm)
                gt_num = self._to_float(gt_norm)
                if pred_num is not None and gt_num is not None:
                    tol = max(1e-6, abs(gt_num) * 1e-6)
                    correct = abs(pred_num - gt_num) <= tol

            rewards.append(1.0 if correct else -1.0)
        return rewards


orms['external_open_rl_reward'] = OpenRLRewardORM
