from __future__ import annotations

import json

from .data_model import BenchmarkItem, MemoryRecord


NAIVE_MEMORY_SYSTEM = """You are a long-term memory manager for an AI assistant.

Read the interaction and write concise memories that may help with future tasks involving this user, codebase, dataset, client, or workflow.

Only output memories as a JSON list of strings. Do not explain."""

REFLECTION_MEMORY_SYSTEM = """You are a reflective AI agent.

Read the interaction and extract reusable lessons for future similar tasks. Focus on patterns, causes, fixes, and workflow-specific knowledge that should be remembered.

Only output reusable lessons as a JSON list of strings. Do not explain."""

SOURCE_AWARE_SYSTEM = """You are a source-aware memory manager.

Read the interaction and write concise memories only when they are supported by user statements, environment evidence, code execution, calculations, tool outputs, tests, or cited text. Do not store an assistant guess as a fact unless the interaction explicitly verifies it.

Only output memories as a JSON list of strings. Do not explain."""

QUOTE_REQUIRED_SYSTEM = """You are a provenance-focused memory manager.

Stable memories must include a direct supporting quote or span from the interaction. If a claim lacks a direct quote, mark it Tentative or omit it. Do not promote assistant speculation merely because it appears in the assistant trace.

Output strict JSON:
[
  {
    "memory": "...",
    "status": "Stable | Tentative",
    "supporting_quote": "..."
  }
]"""

EVIDENCE_LABELED_SYSTEM = """You are a memory manager for a reasoning agent.

For each possible memory, identify its evidence type:

Observed = directly stated by the user or environment.
Executed = verified by code, tests, tools, calculations, or deterministic outputs.
Cited = supported by a quoted source.
Inferred = reasoned by the model from evidence but not directly verified.
Hypothesized = speculative model explanation.
Self-derived = produced from the model's own prior reasoning or summary.

Only Observed, Executed, and Cited memories may be stored as Stable.
Inferred, Hypothesized, and Self-derived memories must be Tentative.

A memory must not be Stable merely because it appeared in the assistant's reasoning trace. If the assistant guessed a cause but no tool, test, calculation, source, or user confirmation verified it, mark it as Hypothesized or Inferred. If a hypothesis is explicitly verified later in the interaction, it may be Stable and should name the verification.

Output strict JSON:
[
  {
    "memory": "...",
    "evidence_type": "Observed | Executed | Cited | Inferred | Hypothesized | Self-derived",
    "status": "Stable | Tentative",
    "supporting_evidence": "...",
    "reuse_rule": "..."
  }
]"""

EVIDENCE_METHODS = {
    "evidence_labeled",
    "evidence_labeled_no_enforcement",
    "evidence_labeled_stable_only",
    "evidence_labeled_enforced",
}


def canonical_method(method: str) -> str:
    return "evidence_labeled_enforced" if method == "evidence_labeled" else method


def source_episode_text(item: BenchmarkItem) -> str:
    ep = item.source_episode
    return f"""User task:
{ep.user_task}

Available evidence:
{ep.evidence}

Assistant reasoning trace:
{ep.agent_trace}

Assistant final response:
{ep.final_response}"""


def memory_prompt(method: str, item: BenchmarkItem) -> tuple[str, str]:
    method = canonical_method(method)
    if method in {"naive", "current_evidence_self_check"}:
        return NAIVE_MEMORY_SYSTEM, source_episode_text(item)
    if method == "reflection":
        return REFLECTION_MEMORY_SYSTEM, source_episode_text(item)
    if method == "source_aware":
        return SOURCE_AWARE_SYSTEM, source_episode_text(item)
    if method in {"quote_required", "quote_required_plus_self_check"}:
        return QUOTE_REQUIRED_SYSTEM, source_episode_text(item)
    if method in EVIDENCE_METHODS:
        return EVIDENCE_LABELED_SYSTEM, source_episode_text(item)
    raise ValueError(f"No memory prompt for method: {method}")


def format_memories(method: str, memories: list[MemoryRecord]) -> str:
    method = canonical_method(method)
    if not memories:
        return "[]"
    if method in EVIDENCE_METHODS or method == "quote_required":
        return json.dumps(
            [
                {
                    "memory": m.memory,
                    "evidence_type": m.evidence_type,
                    "status": m.status,
                    "supporting_evidence": m.supporting_evidence,
                    "reuse_rule": m.reuse_rule,
                }
                for m in memories
            ],
            indent=2,
        )
    return json.dumps([m.memory for m in memories], indent=2)


def memories_for_downstream(method: str, memories: list[MemoryRecord]) -> list[MemoryRecord]:
    method = canonical_method(method)
    if method == "evidence_labeled_stable_only":
        return [m for m in memories if m.status.lower() == "stable"]
    return memories


def downstream_trap_prompt(method: str, item: BenchmarkItem, memories: list[MemoryRecord]) -> str:
    future = item.future_task
    current_task = f"""User task:
{future.user_task}

Current evidence:
{future.current_evidence}"""
    return _downstream_prompt(method, current_task, memories)


def downstream_control_prompt(method: str, item: BenchmarkItem, memories: list[MemoryRecord]) -> str:
    return _downstream_prompt(method, item.verified_memory_control.future_task, memories)


def _downstream_prompt(method: str, current_task: str, memories: list[MemoryRecord]) -> str:
    method = canonical_method(method)
    memories = memories_for_downstream(method, memories)
    if method == "no_memory":
        return f"""You are solving a task.

Current task:
{current_task}

Answer the current task. Prioritize current evidence."""
    if method == "evidence_labeled_enforced":
        return f"""You are solving a task with access to long-term memories.

Stable memories may be used as facts.
Tentative memories may only be treated as hypotheses and must not override current evidence.

Memories:
{format_memories(method, memories)}

Current task:
{current_task}

Answer the current task. Prioritize current evidence and verified memories."""
    if method in {"current_evidence_self_check", "quote_required_plus_self_check"}:
        return f"""You are solving a task with access to long-term memories.

Before answering, compare any relevant memory against the current evidence. If memory and current evidence differ, use the current evidence.

Memories:
{format_memories(method, memories)}

Current task:
{current_task}

Answer the current task."""
    if method in {"evidence_labeled_no_enforcement", "evidence_labeled_stable_only", "quote_required"}:
        return f"""You are solving a task with access to long-term memories.

Memories:
{format_memories(method, memories)}

Current task:
{current_task}

Answer the current task. Use memories only if relevant. Prioritize current evidence."""
    return f"""You are solving a task with access to long-term memories.

Long-term memories:
{format_memories(method, memories)}

Current task:
{current_task}

Answer the current task. Use memories only if relevant. Prioritize current evidence."""
