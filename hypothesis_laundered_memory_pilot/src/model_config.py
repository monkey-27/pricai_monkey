from __future__ import annotations

from pathlib import Path


def load_model_config(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """Tiny YAML subset parser for configs/open_models.yaml.

    The config is intentionally simple: backend -> tier -> list of model ids.
    Avoiding a PyYAML dependency keeps the repo easy to run in minimal envs.
    """
    data: dict[str, dict[str, list[str]]] = {}
    section: str | None = None
    tier: str | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw.startswith(" ") and stripped.endswith(":"):
            section = stripped[:-1]
            data.setdefault(section, {})
            tier = None
            continue
        if section and raw.startswith("  ") and not raw.startswith("    ") and stripped.endswith(":"):
            tier = stripped[:-1]
            data[section].setdefault(tier, [])
            continue
        if section and tier and stripped.startswith("- "):
            data[section][tier].append(stripped[2:].strip())
    return data


def select_models(config: dict[str, dict[str, list[str]]], tier: str) -> list[str]:
    transformers = config.get("transformers", {})
    if tier == "all":
        models: list[str] = []
        for values in transformers.values():
            models.extend(values)
        return list(dict.fromkeys(models))
    return transformers.get(tier, [])
