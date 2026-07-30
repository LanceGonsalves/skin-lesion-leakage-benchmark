"""Configuration loading and shared utilities.

Every entry point reads config through here so that a single `config.yaml`
governs all experiments. This matters for this project in particular: the
naive-vs-grouped comparison is only valid if *everything except the split*
is held constant.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Repo root = parent of the directory containing this file.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Thin wrapper over the parsed YAML.

    Access nested values with dotted paths, e.g. ``cfg.get("audit.phash.hash_size")``.
    Paths are resolved relative to the repo root so scripts work from anywhere.
    """

    raw: dict[str, Any]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        """Resolve a configured path relative to the repo root."""
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"No path configured at '{dotted}'")
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    @property
    def seed(self) -> int:
        return int(self.get("seed", 42))

    @property
    def classes(self) -> list[str]:
        return list(self.get("dataset.classes", []))


def load_config(path: str | Path | None = None) -> Config:
    """Load `config.yaml` (or an explicit path) into a Config."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh) or {})


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if present) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch is optional for the audit phase, which is CPU/PIL only.
        pass
