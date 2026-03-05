from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.prompt_formats import direct_action_user_prompt, text_action_user_prompt
from src.envs.empty_env import ACTION_NAMES, NAME_TO_ACTION, get_frame, make_empty_env
from src.eval.evaluate import evaluate_policy
from src.models.nanovlm_policy import NanoVLMPolicy, parse_action
from src.utils import ensure_dir, save_json, set_seed


def rollout_episode(policy: NanoVLMPolicy, mode: str, env_seed: int, max_steps: int = 100):
    env = make_empty_env(size=8, random_start=True, max_steps=max_steps, seed=env_seed)
    _obs, _info = env.reset(seed=env_seed)

    transitions = []
    done = False
    ep_return = 0.0

    while not done:
        frame = get_frame(env)
        prompt = direct_action_user_prompt() if mode == "action" else text_action_user_prompt()
        generated = policy.generate(frame, prompt, max_new_tokens=96 if mode == "text_action" else 4, temperature=0.8)
        action_name = parse_action(generated) or "forward"
        action = NAME_TO_ACTION[action_name]

        _obs, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        ep_return += float(reward)

        transitions.append(
            {
                "frame": frame,
                "prompt": prompt,
                "action_text": action_name if mode == "action" else generated,
                "reward": float(reward),
                "done": done,
            }
        )

    env.close()
    success = 1 if ep_return > 0 else 0
    return transitions, ep_return, success


def normalized_advantages(rewards: list[float]) -> np.ndarray:
    arr = np.array(rewards, dtype=np.float32)
    std = np.std(arr)
    if std < 1e-6:
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / (std + 1e-8)


def grpo_update(policy: NanoVLMPolicy, ref_policy: NanoVLMPolicy, grouped_rollouts, optimizer, kl_beta: float):
    losses = []
    for group in grouped_rollouts:
        group_returns = [item[1] for item in group]
        adv = normalized_advantages(group_returns)

        for gi, (traj, _ret, _succ) in enumerate(group):
            for step in traj:
                frame = step["frame"]
                prompt = step["prompt"]
                action_text = step["action_text"] if step["action_text"] else ACTION_NAMES[2]

                policy_loss = policy.sft_loss(frame, prompt, action_text)
                with torch.no_grad():
                    ref_loss = ref_policy.sft_loss(frame, prompt, action_text)

                logp_term = policy_loss * (-adv[gi])
                kl_term = torch.clamp(policy_loss - ref_loss, min=0.0)
                loss = logp_term + kl_beta * kl_term

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint", type=str, default="artifacts/sft/checkpoint_last")
    parser.add_argument("--nanovlm_repo", type=str, default="external/nanoVLM")
    parser.add_argument("--output_dir", type=str, default="artifacts/grpo_action")
    parser.add_argument("--mode", type=str, choices=["action", "text_action"], default="action")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes_per_update", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    policy = NanoVLMPolicy.load_from_checkpoint(args.init_checkpoint, args.nanovlm_repo, device="cuda")
    ref_policy = NanoVLMPolicy.load_from_checkpoint(args.init_checkpoint, args.nanovlm_repo, device="cuda")
    ref_policy.set_mode(False)

    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=args.lr)
    history = []

    print(
        f"Starting GRPO mode={args.mode}, updates={args.updates}, "
        f"episodes_per_update={args.episodes_per_update}, group_size={args.group_size}",
        flush=True,
    )

    for update in tqdm(range(args.updates), desc=f"grpo-{args.mode}"):
        grouped = []
        for episode_block in range(args.episodes_per_update):
            base_seed = args.seed + update * 1000 + episode_block * 10
            group_rollouts = []
            for g in range(args.group_size):
                traj, ep_ret, succ = rollout_episode(policy, args.mode, env_seed=base_seed + g)
                group_rollouts.append((traj, ep_ret, succ))
            grouped.append(group_rollouts)

        train_loss = grpo_update(policy, ref_policy, grouped, optimizer, kl_beta=args.kl_beta)

        print(
            f"update {update + 1}/{args.updates} done, train_loss={train_loss:.6f}",
            flush=True,
        )

        if (update + 1) % 10 == 0:
            eval_metrics = evaluate_policy(policy=policy, mode=args.mode, episodes=args.eval_episodes)
            print(
                f"eval@{update + 1}: success_rate={eval_metrics.get('success_rate', float('nan')):.4f}, "
                f"avg_return={eval_metrics.get('avg_return', float('nan')):.4f}",
                flush=True,
            )
            row = {
                "update": update + 1,
                "train_loss": train_loss,
                **eval_metrics,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(Path(args.output_dir) / "history.csv", index=False)

            ckpt_dir = Path(args.output_dir) / f"checkpoint_update_{update+1}"
            policy.save(str(ckpt_dir))

            ref_policy = NanoVLMPolicy.load_from_checkpoint(str(ckpt_dir), args.nanovlm_repo, device="cuda")
            ref_policy.set_mode(False)

    final_dir = Path(args.output_dir) / "checkpoint_last"
    policy.save(str(final_dir))
    save_json({"mode": args.mode, "history": history}, Path(args.output_dir) / "summary.json")
    print(f"GRPO finished. Last checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
