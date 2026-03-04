from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
from minigrid.core.world_object import Goal


ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2
ACTION_NAMES = {0: "left", 1: "right", 2: "forward"}
NAME_TO_ACTION = {v: k for k, v in ACTION_NAMES.items()}


@dataclass
class EnvStep:
    frame: np.ndarray
    reward: float
    terminated: bool
    truncated: bool


def make_empty_env(size: int = 8, random_start: bool = True, max_steps: int | None = None, seed: int | None = None):
    """Create MiniGrid EmptyEnv, trying several env-id variants for compatibility."""
    kwargs = {}
    if max_steps is not None:
        kwargs["max_steps"] = max_steps

    candidates = [f"MiniGrid-Empty-{'Random-' if random_start else ''}{size}x{size}-v0"]
    for s in [6, 8, 16, 5]:
        candidates.append(f"MiniGrid-Empty-{'Random-' if random_start else ''}{s}x{s}-v0")
        candidates.append(f"MiniGrid-Empty-{s}x{s}-v0")

    seen = set()
    for env_id in candidates:
        if env_id in seen:
            continue
        seen.add(env_id)
        try:
            env = gym.make(env_id, render_mode="rgb_array", **kwargs)
            if seed is not None:
                env.reset(seed=seed)
            return env
        except Exception:
            continue

    raise RuntimeError(f"No compatible MiniGrid Empty env found. Tried: {sorted(seen)}")


def get_goal_pos(env) -> Tuple[int, int]:
    grid = env.unwrapped.grid
    width, height = env.unwrapped.width, env.unwrapped.height
    for x in range(width):
        for y in range(height):
            obj = grid.get(x, y)
            if isinstance(obj, Goal):
                return x, y
    raise RuntimeError("Goal cell not found")


def shortest_turn(curr_dir: int, target_dir: int) -> list[int]:
    right_steps = (target_dir - curr_dir) % 4
    left_steps = (curr_dir - target_dir) % 4
    if right_steps <= left_steps:
        return [ACTION_RIGHT] * right_steps
    return [ACTION_LEFT] * left_steps


def desired_dir(agent_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> int:
    ax, ay = agent_pos
    gx, gy = goal_pos
    dx, dy = gx - ax, gy - ay
    if abs(dx) >= abs(dy):
        return 0 if dx > 0 else 2
    return 1 if dy > 0 else 3


def expert_next_action(env) -> int:
    unwrapped = env.unwrapped
    agent_pos = tuple(unwrapped.agent_pos)
    goal = get_goal_pos(env)

    if agent_pos == goal:
        return ACTION_FORWARD

    tdir = desired_dir(agent_pos, goal)
    turns = shortest_turn(unwrapped.agent_dir, tdir)
    if turns:
        return turns[0]
    return ACTION_FORWARD


def get_frame(env) -> np.ndarray:
    if hasattr(env, "get_frame"):
        return env.get_frame(highlight=False, tile_size=16)

    unwrapped = env.unwrapped
    if hasattr(unwrapped, "get_frame"):
        return unwrapped.get_frame(highlight=False, tile_size=16)

    frame = env.render()
    if frame is not None:
        return frame

    raise RuntimeError("Unable to get frame from environment")
