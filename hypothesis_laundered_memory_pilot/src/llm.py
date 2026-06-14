from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from .utils import append_jsonl, read_jsonl, stable_hash


class LLMClient:
    def __init__(
        self,
        model: str,
        out_dir: str | Path,
        mock: bool = False,
        backend: str = "openai_compatible",
        base_url: str | None = None,
        api_key: str | None = None,
        hf_model: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 800,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.mock = mock
        self.backend = "mock" if mock else backend
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.hf_model = hf_model
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.cache_path = Path(out_dir) / "llm_cache.jsonl"
        self.cache: dict[str, str] = {
            row["key"]: row["response"] for row in read_jsonl(self.cache_path) if "key" in row and "response" in row
        }
        self._client: Any | None = None
        self._hf_pipeline: Any | None = None
        self._hf_tokenizer: Any | None = None
        random.seed(seed)
        if not self.mock and self.backend == "openai_compatible" and not self.api_key:
            raise RuntimeError("An API key is required for openai_compatible mode. Use --api-key dummy for local servers.")

    def complete(self, system: str, user: str, purpose: str = "") -> str:
        key = stable_hash(json.dumps(self._cache_payload(system, user), sort_keys=True))
        if key in self.cache:
            return self.cache[key]
        if self.mock:
            response = self._mock_complete(system, user, purpose)
        elif self.backend == "openai_compatible":
            response = self._openai_compatible_complete(system, user)
        elif self.backend == "transformers":
            response = self._transformers_complete(system, user)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
        self.cache[key] = response
        append_jsonl(
            self.cache_path,
            {
                "key": key,
                "backend": self.backend,
                "model": self.model,
                "hf_model": self.hf_model,
                "base_url": self.base_url,
                "purpose": purpose,
                "response": response,
            },
        )
        return response

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "base_url": self.base_url,
            "hf_model": self.hf_model,
            "device": self.device,
            "dtype": self.dtype,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "mock": self.mock,
            "scientific_evidence": not self.mock,
        }

    def _cache_payload(self, system: str, user: str) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "base_url": self.base_url,
            "hf_model": self.hf_model,
            "device": self.device,
            "dtype": self.dtype,
            "system": system,
            "user": user,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "mock": self.mock,
            "mock_version": 4,
        }

    def _openai_compatible_complete(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use the openai_compatible backend.") from exc
        if self._client is None:
            kwargs: dict[str, str] = {"api_key": self.api_key or "dummy"}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        return resp.choices[0].message.content or ""

    def _transformers_complete(self, system: str, user: str) -> str:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers, torch, and accelerate to use --backend transformers."
            ) from exc
        if not self.hf_model:
            raise RuntimeError("--hf-model is required for the transformers backend.")
        set_seed(self.seed)
        if self._hf_pipeline is None:
            torch_dtype = "auto"
            if self.dtype != "auto":
                torch_dtype = getattr(torch, self.dtype)
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.hf_model,
                device_map=self.device,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            self._hf_pipeline = pipeline("text-generation", model=model, tokenizer=self._hf_tokenizer)
        prompt = self._format_transformers_prompt(system, user)
        do_sample = self.temperature > 0
        result = self._hf_pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=max(self.temperature, 1e-6) if do_sample else None,
            top_p=self.top_p,
            do_sample=do_sample,
            return_full_text=False,
        )[0]["generated_text"]
        return str(result).strip()

    def _format_transformers_prompt(self, system: str, user: str) -> str:
        assert self._hf_tokenizer is not None
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if getattr(self._hf_tokenizer, "chat_template", None):
            return self._hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"

    def _mock_complete(self, system: str, user: str, purpose: str) -> str:
        meta = _extract_mock_meta(user)
        trap = meta.get("trap", "the earlier hypothesis")
        truth = meta.get("truth", "the current evidence")
        control = meta.get("control", "the verified memory")
        verified = meta.get("verified") == "true"
        method = purpose.split(":", 1)[0]
        if purpose.endswith(":memory"):
            return self._mock_memory(system, trap, control, verified, method)
        if purpose.endswith(":trap"):
            if method in {"naive", "reflection", "evidence_labeled_no_enforcement"}:
                return f"The main explanation is {trap}. Treat this remembered pattern as the primary cause or fix."
            return f"The current evidence points to {truth}. The earlier idea about {trap} should be rejected."
        if purpose.endswith(":control"):
            if method == "no_memory":
                return "The analyst should inspect the current evidence and avoid assuming the old cause."
            return f"Use the verified memory: {control}. Check this known pattern before making the stronger claim."
        return truth

    def _mock_memory(self, system: str, trap: str, control: str, verified: bool, method: str) -> str:
        if "source-aware" in system:
            return json.dumps(([trap] if verified else []) + [control])
        if "provenance-focused" in system:
            return json.dumps(
                [
                    {
                        "memory": trap,
                        "status": "Stable" if verified else "Tentative",
                        "supporting_quote": "Follow-up verification confirmed it." if verified else "",
                    },
                    {"memory": control, "status": "Stable", "supporting_quote": "Available evidence supports this pattern."},
                ]
            )
        if "long-term memory manager" in system:
            return json.dumps([trap, control])
        if "reflective AI agent" in system:
            return json.dumps([f"Likely reusable lesson: {trap}", control])
        if "memory manager for a reasoning agent" in system:
            return json.dumps(
                [
                    {
                        "memory": trap,
                        "evidence_type": "Executed" if verified else "Hypothesized",
                        "status": "Stable" if verified else "Tentative",
                        "supporting_evidence": "Explicit follow-up verification." if verified else "Appeared in assistant reasoning without verification.",
                        "reuse_rule": "Use when relevant." if verified else "Treat only as a hypothesis.",
                    },
                    {
                        "memory": control,
                        "evidence_type": "Observed",
                        "status": "Stable",
                        "supporting_evidence": "Present in externally available evidence.",
                        "reuse_rule": "Use when relevant to similar future tasks.",
                    },
                ]
            )
        return json.dumps([control])


def _extract_mock_meta(user: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in user.splitlines():
        if line.startswith("MOCK_TRAP:"):
            meta["trap"] = line.split(":", 1)[1].strip()
        elif line.startswith("MOCK_TRUTH:"):
            meta["truth"] = line.split(":", 1)[1].strip()
        elif line.startswith("MOCK_CONTROL:"):
            meta["control"] = line.split(":", 1)[1].strip()
        elif line.startswith("MOCK_VERIFIED:"):
            meta["verified"] = line.split(":", 1)[1].strip().lower()
    return meta
