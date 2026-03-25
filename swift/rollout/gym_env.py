# Copyright (c) ModelScope Contributors. All rights reserved.
# GYM Environment and Context Manager implementations for GRPO training.

import os
import time
import uuid
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import aiohttp

from swift.infer_engine.protocol import RolloutInferRequest
from swift.rewards.orm import MathAccuracy
from swift.template import Messages


class ContextManager(ABC):
    """Base context manager interface for managing conversation history."""

    def __init__(self, ctx_config):
        self.ctx_config = ctx_config

    @abstractmethod
    def manage_context(self, history: Messages, trajectory_id: str) -> Messages:
        """Manage conversation context and history.

        Args:
            history: Current conversation history
            trajectory_id: Current agent trajectory_id
        Returns:
            Modified conversation history with context management applied
        """
        pass


class DummyContextManager(ContextManager):

    def __init__(self, ctx_config):
        super().__init__(ctx_config)

    def manage_context(self, history: Messages, trajectory_id: str) -> Messages:
        return history


# Registry for context managers
context_managers = {'dummyContextManager': DummyContextManager}


class Env(ABC):
    """Base environment interface for GRPO training."""

    def __init__(self, env_config):
        """Initialize environment."""
        self.env_config = env_config

    @abstractmethod
    async def reset(self, config: RolloutInferRequest) -> Tuple[str, Dict[str, Any], str]:
        """Reset environment to initial state.

        Args:
            config: Initial configuration containing dataset information

        Returns:
            Tuple of (observation, info, system_message):
            - observation: Initial query string for the agent
            - info: Environment debug information as dict
            - system_message: System prompt for this trajectory
        """
        pass

    @abstractmethod
    async def step(self, action: Messages) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: LLM response choice containing the action to execute

        Returns:
            Tuple of (next_observation, reward, done, info):
            - next_observation: Next observation string
            - reward: Reward value for this step
            - done: Whether the episode is finished
            - info: Additional information as dict
        """
        pass

    @abstractmethod
    async def close(self):
        """Clean up environment resources."""
        pass


@lru_cache(maxsize=1)
def _get_qwen_tokenizer():
    from modelscope import AutoTokenizer

    model_name = (
        os.environ.get('SWIFT_GYM_TOKENIZER')
        or os.environ.get('ROLLOUT_MODEL')
        or os.environ.get('MODEL_PATH')
        or 'Qwen/Qwen2.5-3B-Instruct')
    return AutoTokenizer.from_pretrained(model_name)


def count_qwen_tokens(messages: List[Dict[str, Any]], max_tokens: int = 2048) -> Tuple[int, bool]:
    """
    Calculate token count for Qwen messages and check if it exceeds the 16k limit

    Args:
        messages: List of messages in OpenAI format
        max_tokens: Maximum token limit, default 2k

    Returns:
        Tuple[int, bool]: (token count, whether within limit)
    """
    try:
        tokenizer = _get_qwen_tokenizer()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_count = len(tokenizer.encode(text))

        return token_count, token_count >= max_tokens

    except Exception as e:
        print(f'Token calculation failed: {e}')
        return 0, False


class SimpleMathEnv(Env):
    tips_prompt = 'The answer is not correct, It seems You made a mistake, you need to recheck very carefully.'

    def __init__(self, env_config):
        super().__init__(env_config)
        self.acc_func = MathAccuracy()
        self.solution = ''

    async def reset(self, config: RolloutInferRequest) -> Tuple[str, Dict[str, Any], str]:
        obs = config.data_dict['problem']
        info = {}
        self.solution = config.data_dict['solution']
        system_prompt = """A conversation between User and Assistant.
        The user asks a question, and the Assistant solves it.
        The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
        The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags,
        respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>
        """
        return obs, info, system_prompt

    async def step(self, action: Messages) -> Tuple[str, float, bool, Dict[str, Any]]:
        next_obs = self.tips_prompt

        reward = 0.0
        done = False
        info = {}
        acc = self.acc_func([action[-1]['content']], [self.solution])[0]
        if count_qwen_tokens(action)[1]:
            done = True
            info['stop_reason'] = 'Exceeded maximum length'

        if acc == 1:
            done = True
            reward = 1.0
            info['stop_reason'] = 'Correct'
        info['math_reward'] = reward
        return next_obs, reward, done, info

    async def close(self):
        pass


class NeMoGymEnv(Env):

    def __init__(self, env_config):
        super().__init__(env_config)
        self.verify_url = os.environ.get('NEMO_GYM_VERIFY_URL')
        if not self.verify_url:
            raise ValueError('NEMO_GYM_VERIFY_URL is required when using nemo_gym_env')
        self.prompt_key = env_config.get('prompt_key', 'prompt')
        self.reward_key = env_config.get('reward_key', 'reward')
        self.done_on_verify = env_config.get('done_on_verify', True)
        self.info_keys = tuple(env_config.get('info_keys', []))
        self.data_dict: Dict[str, Any] = {}
        self.responses_create_params: Dict[str, Any] = {}
        self.request_timeout = float(env_config.get('request_timeout', 120.0))

    async def reset(self, config: RolloutInferRequest) -> Tuple[str, Dict[str, Any], str]:
        self.data_dict = dict(config.data_dict or {})
        self.responses_create_params = dict(self.data_dict.get('responses_create_params') or {})
        observation = self.data_dict.get(self.prompt_key)
        if observation is None:
            raise KeyError(
                f"Prompt key '{self.prompt_key}' not found in data_dict. Available keys: {list(self.data_dict.keys())}")
        return observation, {}, ''

    def _build_verify_payload(self, action: Messages) -> Dict[str, Any]:
        assistant_text = action[-1]['content']
        responses_create_params = self.responses_create_params or {
            'input': [{'role': 'user', 'content': self.data_dict[self.prompt_key], 'type': 'message'}]
        }

        payload = {
            'responses_create_params': responses_create_params,
            'response': {
                'id': f'swift-rollout-{uuid.uuid4().hex}',
                'created_at': time.time(),
                'model': os.environ.get('ROLLOUT_MODEL') or os.environ.get('MODEL_PATH') or 'swift-rollout',
                'object': 'response',
                'output': [{
                    'id': f'msg-{uuid.uuid4().hex}',
                    'content': [{'annotations': [], 'text': assistant_text, 'type': 'output_text'}],
                    'role': 'assistant',
                    'status': 'completed',
                    'type': 'message',
                }],
                'parallel_tool_calls': False,
                'tool_choice': 'none',
                'tools': [],
            },
        }

        for key in ('question', 'expected_answer', 'problem', 'solution', 'answer', 'verifier_metadata'):
            if key in self.data_dict:
                payload[key] = self.data_dict[key]
        return payload

    async def step(self, action: Messages) -> Tuple[str, float, bool, Dict[str, Any]]:
        payload = self._build_verify_payload(action)
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.verify_url, json=payload) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f'NeMo Gym verify failed: {resp.status}, {text[:1000]}')
                verify_result = await resp.json()

        reward = float(verify_result.get(self.reward_key, verify_result.get('reward', 0.0)))
        info = {'nemo_gym_verify_result': verify_result}
        for key in self.info_keys:
            if key in verify_result:
                info[key] = verify_result[key]
        done = bool(self.done_on_verify)
        next_obs = ''
        if count_qwen_tokens(action)[1]:
            done = True
            info['stop_reason'] = 'Exceeded maximum length'
        elif done:
            info['stop_reason'] = 'Verified'
        return next_obs, reward, done, info

    async def close(self):
        pass


# Registry for environments
envs = {'math_env': SimpleMathEnv, 'nemo_gym_env': NeMoGymEnv}
