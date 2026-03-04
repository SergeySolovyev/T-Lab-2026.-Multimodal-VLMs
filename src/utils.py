from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def save_json(data: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(data):
        data = asdict(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)
