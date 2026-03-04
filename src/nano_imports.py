from __future__ import annotations

import sys
from pathlib import Path


def add_nanovlm_to_path(nanovlm_repo: str | Path) -> None:
    repo_path = Path(nanovlm_repo).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(
            f"NanoVLM repo not found at {repo_path}. Clone it first to external/nanoVLM"
        )
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
