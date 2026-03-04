param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "."

& $Python -m src.data.generate_expert_dataset --out_dir artifacts/datasets --train_episodes 4000 --val_episodes 500 --test_episodes 500 --seed 42

& $Python -m src.train.train_sft --train_npz artifacts/datasets/train.npz --val_npz artifacts/datasets/val.npz --mode action --epochs 2 --output_dir artifacts/sft --model_source lusxvr/nanoVLM-222M --nanovlm_repo external/nanoVLM

& $Python -m src.train.train_grpo --init_checkpoint artifacts/sft/checkpoint_last --mode action --updates 100 --episodes_per_update 4 --group_size 4 --output_dir artifacts/grpo_action --nanovlm_repo external/nanoVLM

& $Python -m src.train.train_grpo --init_checkpoint artifacts/sft/checkpoint_last --mode text_action --updates 100 --episodes_per_update 4 --group_size 4 --output_dir artifacts/grpo_text_action --nanovlm_repo external/nanoVLM

& $Python -m src.eval.compare_runs --sft artifacts/sft/history.csv --grpo_action artifacts/grpo_action/history.csv --grpo_text_action artifacts/grpo_text_action/history.csv --out_dir artifacts/plots
