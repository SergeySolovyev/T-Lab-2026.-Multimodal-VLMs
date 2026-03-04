from __future__ import annotations

from src.envs.empty_env import ACTION_NAMES


def direct_action_user_prompt() -> str:
    return (
        "You control an agent in MiniGrid EmptyEnv. "
        "Choose exactly one action for the next step: left, right, or forward."
    )


def direct_action_target(action_id: int) -> str:
    return ACTION_NAMES[action_id]


def text_action_user_prompt() -> str:
    return (
        "You control an agent in MiniGrid EmptyEnv. "
        "Write 2-3 short sentences: (1) what you see and orientation, "
        "(2) immediate plan to reach the green goal, "
        "then final line exactly: ACTION: <left|right|forward>."
    )


def text_action_target(action_id: int) -> str:
    action = ACTION_NAMES[action_id]
    return (
        "I see the room and need to move toward the green goal. "
        "I will align my heading and reduce Manhattan distance to the goal.\n"
        f"ACTION: {action}"
    )
