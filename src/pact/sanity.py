"""Helpers for reading sanity-check outputs."""

from __future__ import annotations

import json
from pathlib import Path


def load_sanity(path: str | Path = "outputs/sanity_checks.json") -> dict[str, float]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

