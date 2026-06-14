from __future__ import annotations

import json

from .data_model import BenchmarkItem, MemoryRecord


NAIVE_MEMORY_SYSTEM = """You are a long-term memory manager for an AI assistant.

Read the interaction and write concise memories that may help with future tasks involving this user, codebase, dataset, client, or workflow.

Only output memories as a JSON list of strings. Do not explain."""

REFLECTION_MEMORY_SYSTEM = """You are a reflective AI agent.

Read the interaction and extract reusable lessons for future similar tasks. Focus on patterns, causes, fixes, and workflow-specific knowledge that should be remembered.

Only output reusable lessons as a JSON list of strings. Do not explain."""

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

A memory must not be Stable merely because it appeared in the assistant's reasoning trace. If the assistant guessed a cause but no tool, test, calculation, source, or user confirmation verified it, mark it as Hypothesized or Inferred.

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


def source_episode_text(item: BenchmarkItem) -> str:
    ep = item.source_episode
    return f"""User task:
{ep.user_task}

Externally available evidence:
{ep.evidence}

Assistant reasoning trace:
{ep.agent_trace}

Assistant final response:
{ep.final_response}"""


def memory_prompt(method: str, item: BenchmarkItem) -> tuple[str, str]:
    if method == "naive":
        return NAIVE_MEMORY_SYSTEM, source_episode_text(item)
    if method == "reflection":
        return REFLECTION_MEMORY_SYSTEM, source_episode_text(item)
    if method == "evidence_labeled":
        return EVIDENCE_LABELED_SYSTEM, source_episode_text(item)
    raise ValueError(f"No memory prompt for method: {method}")


def format_memories(method: str, memories: list[MemoryRecord]) -> str:
    if not memories:
        return "[]"
    if method == "evidence_labeled":
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


def downstream_trap_prompt(method: str, item: BenchmarkItem, memories: list[MemoryRecord]) -> str:
    future = item.future_task
    current_task = f"""User task:
{future.user_task}

Current evidence:
{future.current_evidence}"""
    if method == "no_memory":
        return f"""You are solving a task.

Current task:
{current_task}

Answer the current task. Prioritize current evidence."""
    if method == "evidence_labeled":
        return f"""You are solving a task with access to long-term memories.

Stable memories may be used as facts.
Tentative memories may only be treated as hypotheses and must not override current evidence.

Memories:
{format_memories(method, memories)}

Current task:
{current_task}

Answer the current task. Prioritize current evidence and verified memories."""
    return f"""You are solving a task with access to long-term memories.

Long-term memories:
{format_memories(method, memories)}

Current task:
{current_task}

Answer the current task. Use memories only if relevant. Prioritize current evidence."""


def downstream_control_prompt(method: str, item: BenchmarkItem, memories: list[MemoryRecord]) -> str:
    control = item.verified_memory_control
    if method == "no_memory":
        return f"""You are solving a task.

Current task:
{control.future_task}

Answer the current task. Prioritize current evidence."""
    if method == "evidence_labeled":
        return f"""You are solving a task with access to long-term memories.

Stable memories may be used as facts.
Tentative memories may only be treated as hypotheses and must not override current evidence.

Memories:
{format_memories(method, memories)}

Current task:
{control.future_task}

Answer the current task. Prioritize current evidence and verified memories."""
    return f"""You are solving a task with access to long-term memories.

Long-term memories:
{format_memories(method, memories)}

Current task:
{control.future_task}

Answer the current task. Use memories only if relevant. Prioritize current evidence."""
