from __future__ import annotations

import numpy as np

from src.data.prompt_formats import direct_action_user_prompt, text_action_user_prompt
from src.envs.empty_env import NAME_TO_ACTION, get_frame, make_empty_env
from src.models.nanovlm_policy import NanoVLMPolicy, parse_action


def evaluate_policy(
    policy: NanoVLMPolicy,
    mode: str,
    size: int = 8,
    random_start: bool = True,
    max_steps: int = 100,
    episodes: int = 100,
    seed: int = 123,
):
    success, returns = [], []

    for ep in range(episodes):
        env = make_empty_env(size=size, random_start=random_start, max_steps=max_steps, seed=seed + ep)
        _obs, _info = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0

        while not done:
            frame = get_frame(env)
            prompt = direct_action_user_prompt() if mode == "action" else text_action_user_prompt()
            out = policy.generate(frame, prompt, max_new_tokens=96 if mode == "text_action" else 4)
            action_name = parse_action(out) or "forward"
            action = NAME_TO_ACTION[action_name]
            _obs, reward, terminated, truncated, _info = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
            if terminated:
                success.append(1)
            elif truncated:
                success.append(0)

        returns.append(ep_ret)
        env.close()

    return {
        "success_rate": float(np.mean(success)) if success else 0.0,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "episodes": episodes,
    }
