from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_json_lenient(text: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        first_array = text.find("[")
        last_array = text.rfind("]")
        first_obj = text.find("{")
        last_obj = text.rfind("}")
        candidates = []
        if first_array != -1 and last_array > first_array:
            candidates.append(text[first_array : last_array + 1])
        if first_obj != -1 and last_obj > first_obj:
            candidates.append(text[first_obj : last_obj + 1])
        for candidate in candidates:
            try:
                return json.loads(candidate), None
            except json.JSONDecodeError:
                pass
        return None, str(exc)


def sample_deterministic(items: list[Any], n: int, seed: int) -> list[Any]:
    rng = random.Random(seed)
    copied = list(items)
    rng.shuffle(copied)
    return copied[: min(n, len(copied))]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").replace("_", " ").split())


def contains_phrase(text: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(text)

