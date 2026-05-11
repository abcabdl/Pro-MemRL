from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def embedded_memrl_root() -> Path:
    return repo_root() / "ProactiveAgent-main-MemRL_KnowU"


def ensure_embedded_memrl_importable() -> Path:
    root = embedded_memrl_root()
    if not root.exists():
        raise FileNotFoundError(f"Embedded MemRL checkout not found: {root}")
    root_str = str(root)
    agent_str = str(root / "agent")
    for path in (root_str, agent_str):
        if path not in sys.path:
            sys.path.insert(0, path)
    return root
