# NanoVLM in MiniGrid EmptyEnv: SFT vs GRPO

инимальный академический репозиторий для задания по сравнению трёх режимов дообучения:

1. SFT (supervised fine-tuning) на экспертных траекториях,
2. GRPO с прямым выводом действия (`action`),
3. GRPO в формате `text + action`.

## ель проекта

даптировать vision-and-language модель NanoVLM для управления агентом в среде MiniGrid EmptyEnv и провести сопоставимый эксперимент по качеству и sample efficiency для трёх режимов обучения.

## тчёты (Part 1 / Part 2)

- `NanoVLM MiniGrid Technical Report.pdf` — **Part 1** (постановка, данные, методика).
- `NanoVLM MiniGrid Technical Report_Results.pdf` — **Part 2** (результаты и обсуждение).

убличная ссылка на Part 2 (должна открываться именно из корня репозитория в ветке `main`):

- https://github.com/SergeySolovyev/T-Lab-2026.-Multimodal-VLMs/blob/main/NanoVLM%20MiniGrid%20Technical%20Report_Results.pdf

## Соответствие требованиям задания

### 1) SFT-бэйзлайн
- енерация экспертного датасета: `src/data/generate_expert_dataset.py`
- ормат обучающих таргетов: `src/data/prompt_formats.py`
- бучение SFT-политики: `src/train/train_sft.py`

### 2) GRPO: прямой вывод действия
- RL-дообучение в режиме `action`: `src/train/train_grpo.py`

### 3) GRPO: текст + действие
- RL-дообучение в режиме `text_action`: `src/train/train_grpo.py`

### 4) тоговое сравнение
- Скрипт сравнения запусков: `src/eval/compare_runs.py`
- Скрипт оценки: `src/eval/evaluate.py`

## Структура репозитория

- `src/` — основная реализация (данные, среда, модель, обучение, оценка)
- `configs/` — конфигурации SFT/GRPO
- `scripts/` — вспомогательные скрипты запуска
- `colab_launcher.ipynb` — основной воспроизводимый пайплайн для Colab
- `artifacts/` — директории для результатов

## оспроизводимость: рекомендованный запуск (Colab)

ткройте `colab_launcher.ipynb` и выполните ячейки последовательно сверху вниз.

оутбук покрывает:
- подготовку окружения и зависимостей,
- подключение/подготовку NanoVLM,
- поэтапный запуск `SFT → GRPO-action → GRPO-text_action → compare`,
- механизм восстановления после прерывания сессии.

## окальный запуск (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/setup_nanovlm.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_local_pipeline.ps1
```

## Ссылки на материалы задания

- MiniGrid EmptyEnv: https://minigrid.farama.org/environments/minigrid/EmptyEnv/
- NanoVLM: https://github.com/huggingface/nanoVLM
- GRPO: https://arxiv.org/abs/2402.03300
