# MiniGrid EmptyEnv × NanoVLM: SFT vs GRPO

Локальный (VS Code) проект для задания:
- SFT на экспертных траекториях
- GRPO (direct action)
- GRPO (text + action)
- Сравнение по success rate / return / sample efficiency

## 1) Быстрый старт (Windows + VS Code)

1. Создайте venv и установите зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Клонируйте NanoVLM в `external/nanoVLM`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_nanovlm.ps1
```

3. Запустите весь MVP пайплайн:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local_pipeline.ps1
```

## 2) Режим запуска на Colab GPU (через тот же репозиторий)

- Откройте репозиторий в Colab runtime с GPU.
- Выполните те же шаги установки (`pip install -r requirements.txt`, `scripts/setup_nanovlm.ps1` эквивалент через `git clone`).
- Запускайте команды из `scripts/run_local_pipeline.ps1` как отдельные ячейки.

## 3) Формат эксперта

Эксперт — детерминированная policy для EmptyEnv:
- выбираем желаемое направление в сторону goal по Manhattan distance,
- минимальными поворотами выравниваем направление,
- делаем `forward`.

Это воспроизводимо и почти оптимально в пустой комнате без препятствий.

## 4) Что генерируется

- `artifacts/datasets/*.npz` — train/val/test экспертные переходы
- `artifacts/sft/*` — чекпоинты и история SFT
- `artifacts/grpo_action/*` — чекпоинты и история GRPO action
- `artifacts/grpo_text_action/*` — чекпоинты и история GRPO text+action
- `artifacts/plots/*` — итоговые графики и summary table

## 5) Команды по шагам

### 5.1 Сбор датасета

```powershell
python -m src.data.generate_expert_dataset --out_dir artifacts/datasets --train_episodes 20000 --val_episodes 2000 --test_episodes 2000
```

### 5.2 SFT baseline

```powershell
python -m src.train.train_sft --train_npz artifacts/datasets/train.npz --val_npz artifacts/datasets/val.npz --mode action --epochs 5 --output_dir artifacts/sft
```

### 5.3 GRPO direct action

```powershell
python -m src.train.train_grpo --init_checkpoint artifacts/sft/checkpoint_last --mode action --output_dir artifacts/grpo_action
```

### 5.4 GRPO text+action

```powershell
python -m src.train.train_grpo --init_checkpoint artifacts/sft/checkpoint_last --mode text_action --output_dir artifacts/grpo_text_action
```

### 5.5 Сравнение

```powershell
python -m src.eval.compare_runs --sft artifacts/sft/history.csv --grpo_action artifacts/grpo_action/history.csv --grpo_text_action artifacts/grpo_text_action/history.csv --out_dir artifacts/plots
```

## 6) Рекомендованный протокол отчёта

- 5+ seeds для финальных цифр.
- Для каждого метода: success rate curve, avg return curve.
- Sample efficiency: число env-steps до заданного порога success rate (например, 0.8).
- Сводная таблица: SFT vs GRPO-action vs GRPO-text+action.

## 7) Публикация на GitHub

После проверки локально:

```powershell
git init
git add .
git commit -m "MiniGrid EmptyEnv + NanoVLM SFT/GRPO pipeline"
git remote add origin https://github.com/SergeySolovyev/T-Lab-2026.-Multimodal-VLMs.git
git push -u origin main
```
