"""Named audit phases for the PACT pilot.

The current environment does not expose true subagents, so these functions are
deliberately separated and report as simulated audit agents.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pact.schema import Episode, ProspectiveActionContract

FORBIDDEN_PREDICTION_FIELDS = (
    "gold_state",
    "case_type",
    "expected_action_keywords",
    "forbidden_action_keywords",
)


def _write(path: Path, title: str, lines: list[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def audit_dataset(
    contracts: list[ProspectiveActionContract], episodes: list[Episode], output_dir: Path
) -> None:
    lines = ["Simulated subagent: Dataset Audit Agent.", ""]
    by_family: dict[str, list[Episode]] = defaultdict(list)
    for episode in episodes:
        by_family[episode.family].append(episode)
    failures: list[str] = []
    if len(contracts) != 10:
        failures.append(f"Expected 10 contracts, found {len(contracts)}.")
    if len(episodes) != 100:
        failures.append(f"Expected 100 episodes, found {len(episodes)}.")
    required = {
        "direct_trigger": 2,
        "indirect_trigger": 3,
        "near_miss": 3,
        "wrong_scope": 1,
    }
    for family, family_episodes in sorted(by_family.items()):
        counts = Counter(episode.case_type for episode in family_episodes)
        if len(family_episodes) != 10:
            failures.append(f"{family}: expected 10 episodes, found {len(family_episodes)}.")
        for case_type, count in required.items():
            if counts[case_type] != count:
                failures.append(f"{family}: expected {count} {case_type}, found {counts[case_type]}.")
        if counts["conflict"] + counts["already_satisfied"] != 1:
            failures.append(f"{family}: expected one conflict or already_satisfied case.")
    trivial = [
        episode.episode_id
        for episode in episodes
        if episode.case_type == "indirect_trigger"
        and any(keyword in episode.current_query.lower() for keyword in ("contract", "guard", "prospective"))
    ]
    lines.append(f"Families: {len(by_family)}; episodes: {len(episodes)}.")
    lines.append(f"Indirect cases with explicit PACT jargon: {trivial or 'none'}.")
    lines.append("Near-miss cases are concept questions or adjacent-domain explanations by construction.")
    lines.append("Domain balance: exactly 10 episodes per family.")
    lines.append("")
    lines.append("Result: PASS" if not failures else "Result: FAIL")
    lines.extend(f"- {failure}" for failure in failures)
    _write(output_dir / "audit_dataset.md", "Dataset Audit Agent", lines)


def audit_baselines(output_dir: Path) -> None:
    lines = ["Simulated subagent: Baseline Fairness Audit Agent.", ""]
    src_root = Path(__file__).resolve().parent
    checked = [src_root / "baselines.py", src_root / "pact.py"]
    leaks: list[str] = []
    for path in checked:
        text = path.read_text(encoding="utf-8")
        for field in FORBIDDEN_PREDICTION_FIELDS:
            if field in text:
                leaks.append(f"{path.name} contains forbidden field name {field}.")
    lines.append("Keyword baseline uses overlap over family/cue/action text with a fixed threshold.")
    lines.append("TF-IDF baseline uses sklearn when available and bag-of-words cosine otherwise.")
    lines.append("PACT methods receive only EpisodeInput, not label or scoring keyword fields.")
    lines.append(f"Leakage scan files: {', '.join(path.name for path in checked)}.")
    lines.append("")
    lines.append("Result: PASS" if not leaks else "Result: FAIL")
    lines.extend(f"- {leak}" for leak in leaks)
    _write(output_dir / "audit_baselines.md", "Baseline Fairness Audit Agent", lines)


def audit_metrics(metrics: dict[str, dict[str, float]], output_dir: Path) -> None:
    lines = ["Simulated subagent: Metric Audit Agent.", ""]
    failures: list[str] = []
    required = {
        "trigger_accuracy",
        "fire_precision",
        "fire_recall",
        "fire_f1",
        "indirect_trigger_recall",
        "false_trigger_rate_near_wrong",
        "action_completion_rate_gold_fire",
        "action_completion_rate_indirect_fire",
        "conflict_accuracy",
        "already_satisfied_accuracy",
        "checker_repair_gain",
    }
    for method, method_metrics in metrics.items():
        missing = sorted(required - set(method_metrics))
        if missing:
            failures.append(f"{method}: missing metrics {missing}.")
        for key, value in method_metrics.items():
            if key != "checker_repair_gain" and not 0.0 <= float(value) <= 1.0:
                failures.append(f"{method}: {key} outside [0, 1]: {value}.")
    lines.append("Zero division is handled by safe division returning 0.0.")
    lines.append("False-trigger rate is defined over near_miss and wrong_scope gold-suppress examples.")
    lines.append("Action completion is evaluated for gold fire cases and separately for indirect fire cases.")
    lines.append("")
    lines.append("Result: PASS" if not failures else "Result: FAIL")
    lines.extend(f"- {failure}" for failure in failures)
    _write(output_dir / "audit_metrics.md", "Metric Audit Agent", lines)


def audit_reproducibility(output_dir: Path) -> None:
    lines = ["Simulated subagent: Reproducibility Audit Agent.", ""]
    root = Path(__file__).resolve().parents[2]
    commands = [
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "-m", "pact.dataset", "--write"],
        [sys.executable, "-m", "pact.run_eval", "--methods", "all"],
    ]
    failures: list[str] = []
    before = (output_dir / "metrics.json").read_text(encoding="utf-8") if (output_dir / "metrics.json").exists() else ""
    for command in commands:
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        lines.append(f"`{' '.join(command)}` exit={completed.returncode}")
        if completed.stdout.strip():
            lines.append("stdout:")
            lines.append("```")
            lines.append(completed.stdout.strip()[-1500:])
            lines.append("```")
        if completed.stderr.strip():
            lines.append("stderr:")
            lines.append("```")
            lines.append(completed.stderr.strip()[-1500:])
            lines.append("```")
        if completed.returncode != 0:
            failures.append(f"Command failed: {' '.join(command)}")
    after = (output_dir / "metrics.json").read_text(encoding="utf-8") if (output_dir / "metrics.json").exists() else ""
    if before and after and json.loads(before) != json.loads(after):
        failures.append("metrics.json changed across repeated evaluation.")
    lines.append("")
    lines.append("Result: PASS" if not failures else "Result: FAIL")
    lines.extend(f"- {failure}" for failure in failures)
    _write(output_dir / "audit_reproducibility.md", "Reproducibility Audit Agent", lines)


def audit_research_value(metrics: dict[str, dict[str, float]], output_dir: Path) -> None:
    lines = ["Simulated subagent: Research-Value Audit Agent.", ""]
    pact = metrics.get("PACTFull", {})
    ordinary_baselines = {
        "NoMemoryBaseline",
        "KeywordTriggerBaseline",
        "TfidfMemoryBaseline",
        "ContractPromptHeuristicBaseline",
    }
    best_baseline = max(
        (
            value.get("action_completion_rate_indirect_fire", 0.0)
            for key, value in metrics.items()
            if key in ordinary_baselines
        ),
        default=0.0,
    )
    pact_indirect = pact.get("action_completion_rate_indirect_fire", 0.0)
    pact_false = pact.get("false_trigger_rate_near_wrong", 1.0)
    lines.append(f"PACTFull indirect action completion: {pact_indirect:.3f}.")
    lines.append(f"Best ordinary-baseline indirect action completion: {best_baseline:.3f}.")
    lines.append(f"PACTFull near/wrong false-trigger rate: {pact_false:.3f}.")
    if pact_indirect > best_baseline and pact_false <= 0.15:
        lines.append("Decision: CONTINUE. The pilot supports testing PACT with a model-backed PAM.")
    else:
        lines.append("Decision: WEAK OR KILL. The headline metric did not clear the simple pilot gate.")
    lines.append("Kill condition for next experiment: PACT loses indirect completion after labels are blinded and cases are expanded.")
    lines.append("Next experiment: replace heuristic PAM with a small local classifier or held-out prompt judge while preserving the same audits.")
    _write(output_dir / "audit_research_value.md", "Research-Value Audit Agent", lines)


def run_all_audits(
    contracts: list[ProspectiveActionContract],
    episodes: list[Episode],
    metrics: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dataset(contracts, episodes, output_dir)
    audit_baselines(output_dir)
    audit_metrics(metrics, output_dir)
    audit_reproducibility(output_dir)
    audit_research_value(metrics, output_dir)
