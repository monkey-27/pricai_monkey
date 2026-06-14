from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .utils import append_jsonl, read_jsonl, stable_hash


class LLMClient:
    def __init__(
        self,
        model: str,
        out_dir: str | Path,
        mock: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> None:
        self.model = model
        self.mock = mock
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_path = Path(out_dir) / "llm_cache.jsonl"
        self.cache: dict[str, str] = {
            row["key"]: row["response"] for row in read_jsonl(self.cache_path) if "key" in row and "response" in row
        }
        self._client: Any | None = None
        if not self.mock and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required unless --mock is used.")

    def complete(self, system: str, user: str, purpose: str = "") -> str:
        key = stable_hash(
            json.dumps(
                {
                    "model": self.model,
                    "system": system,
                    "user": user,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "mock": self.mock,
                    "mock_version": 2,
                },
                sort_keys=True,
            )
        )
        if key in self.cache:
            return self.cache[key]
        if self.mock:
            response = self._mock_complete(system, user, purpose)
        else:
            response = self._openai_complete(system, user)
        self.cache[key] = response
        append_jsonl(
            self.cache_path,
            {
                "key": key,
                "model": self.model,
                "purpose": purpose,
                "response": response,
            },
        )
        return response

    def _openai_complete(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt before running real LLM mode.") from exc
        if self._client is None:
            self._client = OpenAI()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _mock_complete(self, system: str, user: str, purpose: str) -> str:
        meta = _extract_mock_meta(user)
        trap = meta.get("trap", "the earlier hypothesis")
        truth = meta.get("truth", "the current evidence")
        control = meta.get("control", "the verified memory")
        if "long-term memory manager" in system:
            return json.dumps([trap, control])
        if "reflective AI agent" in system:
            return json.dumps([f"Likely reusable lesson: {trap}", control])
        if "memory manager for a reasoning agent" in system:
            return json.dumps(
                [
                    {
                        "memory": trap,
                        "evidence_type": "Hypothesized",
                        "status": "Tentative",
                        "supporting_evidence": "Appeared in the assistant reasoning trace without verification.",
                        "reuse_rule": "Treat only as a hypothesis; do not override current evidence.",
                    },
                    {
                        "memory": control,
                        "evidence_type": "Observed",
                        "status": "Stable",
                        "supporting_evidence": "Explicitly present in externally available evidence.",
                        "reuse_rule": "Use when relevant to similar future tasks.",
                    },
                ]
            )
        if purpose.endswith(":trap"):
            if purpose.startswith("naive") or purpose.startswith("reflection"):
                return f"The main explanation is {trap}. Treat this remembered pattern as the primary cause or fix."
            return f"The current evidence points to {truth}. The earlier hypothesis about {trap} should be rejected."
        if purpose.endswith(":control"):
            if purpose.startswith("no_memory"):
                return "The analyst should inspect the current evidence and avoid assuming the old cause."
            return f"Use the verified memory: {control}. That means the answer should check for this known pattern before concluding anything else."
        return truth


def _extract_mock_meta(user: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in user.splitlines():
        if line.startswith("MOCK_TRAP:"):
            meta["trap"] = line.split(":", 1)[1].strip()
        elif line.startswith("MOCK_TRUTH:"):
            meta["truth"] = line.split(":", 1)[1].strip()
        elif line.startswith("MOCK_CONTROL:"):
            meta["control"] = line.split(":", 1)[1].strip()
    return meta
