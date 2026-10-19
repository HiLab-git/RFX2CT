from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
  with Path(path).open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
  return data or {}


def save_config(config: Dict[str, Any], path: str | Path) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)


def write_json(data: Dict[str, Any], path: str | Path) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

