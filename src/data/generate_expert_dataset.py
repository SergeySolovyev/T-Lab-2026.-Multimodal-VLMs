from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.prompt_formats import (
    direct_action_target,
    direct_action_user_prompt,
    text_action_target,
    text_action_user_prompt,
)
from src.envs.empty_env import expert_next_action, get_frame, make_empty_env
from src.utils import ensure_dir, save_json, set_seed


def collect_split(num_episodes: int, map_sizes: list[int], random_start_prob: float, max_steps: int, seed: int):
    rng = np.random.default_rng(seed)
    frames, actions, direct_prompts, direct_targets, ta_prompts, ta_targets = [], [], [], [], [], []

    for episode_idx in tqdm(range(num_episodes), desc="collect"):
        size = int(rng.choice(map_sizes))
        random_start = bool(rng.random() < random_start_prob)
        env = make_empty_env(size=size, random_start=random_start, max_steps=max_steps, seed=seed + episode_idx)
        _obs, _info = env.reset(seed=seed + episode_idx)

        done = False
        while not done:
            frame = get_frame(env)
            action = expert_next_action(env)

            frames.append(frame)
            actions.append(action)
            direct_prompts.append(direct_action_user_prompt())
            direct_targets.append(direct_action_target(action))
            ta_prompts.append(text_action_user_prompt())
            ta_targets.append(text_action_target(action))

            _obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

        env.close()

    return {
        "frames": np.array(frames, dtype=np.uint8),
        "actions": np.array(actions, dtype=np.int64),
        "direct_prompts": np.array(direct_prompts, dtype=object),
        "direct_targets": np.array(direct_targets, dtype=object),
        "text_action_prompts": np.array(ta_prompts, dtype=object),
        "text_action_targets": np.array(ta_targets, dtype=object),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="artifacts/datasets")
    parser.add_argument("--train_episodes", type=int, default=20000)
    parser.add_argument("--val_episodes", type=int, default=2000)
    parser.add_argument("--test_episodes", type=int, default=2000)
    parser.add_argument("--map_sizes", type=int, nargs="+", default=[5, 6, 8, 16])
    parser.add_argument("--random_start_prob", type=float, default=0.7)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    splits = {
        "train": args.train_episodes,
        "val": args.val_episodes,
        "test": args.test_episodes,
    }

    manifest = {
        "expert": "deterministic shortest-turn+forward heuristic in EmptyEnv",
        "why": "EmptyEnv has no obstacles, so shortest orientation + forward policy is near-optimal and perfectly reproducible.",
        "splits": {},
    }

    for split_name, n_eps in splits.items():
        split = collect_split(
            num_episodes=n_eps,
            map_sizes=args.map_sizes,
            random_start_prob=args.random_start_prob,
            max_steps=args.max_steps,
            seed=args.seed + hash(split_name) % 10000,
        )
        out_path = out_dir / f"{split_name}.npz"
        np.savez_compressed(out_path, **split)
        manifest["splits"][split_name] = {
            "episodes": n_eps,
            "samples": int(split["actions"].shape[0]),
            "path": str(out_path),
        }

    save_json(manifest, out_dir / "manifest.json")
    print(f"Saved dataset to {out_dir}")


if __name__ == "__main__":
    main()
