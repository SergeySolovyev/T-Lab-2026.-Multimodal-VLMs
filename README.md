# NanoVLM in MiniGrid EmptyEnv: SFT vs GRPO

This repository contains a compact, reproducible implementation for the assignment:

1. SFT (supervised fine-tuning) on expert trajectories,
2. GRPO with direct action output (`action`),
3. GRPO with text + action output (`text_action`).

## Project Goal

Adapt a vision-language model (NanoVLM) for control in MiniGrid EmptyEnv and compare training regimes by final quality and sample efficiency.

## Reports (Part 1 / Part 2)

- `NanoVLM MiniGrid Technical Report.pdf` — **Part 1** (problem statement, data, methodology).
- `NanoVLM MiniGrid Technical Report_Results.pdf` — **Part 2** (results, comparison, discussion).

Public Part 2 link (must resolve from repository root on `main`):

- https://github.com/SergeySolovyev/T-Lab-2026.-Multimodal-VLMs/blob/main/NanoVLM%20MiniGrid%20Technical%20Report_Results.pdf

## Mapping to Assignment Requirements

### 1) SFT baseline
- Expert dataset generation: `src/data/generate_expert_dataset.py`
- Prompt/target formats: `src/data/prompt_formats.py`
- SFT training: `src/train/train_sft.py`

### 2) GRPO: direct action output
- RL fine-tuning (`action`): `src/train/train_grpo.py`

### 3) GRPO: text + action output
- RL fine-tuning (`text_action`): `src/train/train_grpo.py`

### 4) Final comparison
- Run comparison: `src/eval/compare_runs.py`
- Evaluation utilities: `src/eval/evaluate.py`

## Repository Structure

- `src/` — core implementation (data, env, model, training, eval)
- `configs/` — SFT/GRPO configuration files
- `scripts/` — helper scripts for setup and runs
- `colab_launcher.ipynb` — main Colab pipeline
- `artifacts/` — output directories for generated results

## Reproducibility (Recommended: Colab)

Open `colab_launcher.ipynb` and run cells top-to-bottom.

Pipeline stages in notebook:
- environment and dependency setup,
- NanoVLM preparation,
- `SFT -> GRPO-action -> GRPO-text_action -> compare`,
- resume flow after runtime interruption.

## Local Run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/setup_nanovlm.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_local_pipeline.ps1
```

## Reference Links

- MiniGrid EmptyEnv: https://minigrid.farama.org/environments/minigrid/EmptyEnv/
- NanoVLM: https://github.com/huggingface/nanoVLM
- GRPO paper: https://arxiv.org/abs/2402.03300
