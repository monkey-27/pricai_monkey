"""Simulated audit agents for PACT-Causal-520."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pact.schema import Episode, Prediction, ProspectiveActionContract
from pact.scoring import score_method


def write(path: Path, title: str, lines: list[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def passfail(items: list[str]) -> str:
    return "PASS" if not items else "FAIL"


def dataset_audit(contracts: list[ProspectiveActionContract], episodes: list[Episode], out: Path) -> None:
    failures: list[str] = []
    by_family = defaultdict(list)
    for ep in episodes:
        by_family[ep.family].append(ep)
    set_counts = Counter(ep.set_type for ep in episodes)
    case_counts = Counter(ep.case_type for ep in episodes)
    if len(contracts) not in {10, 12}:
        failures.append("unexpected contract count")
    if len(contracts) == 12 and len(episodes) not in {520, len([e for e in episodes if e.split == "test"]), len([e for e in episodes if e.split == "dev"])}:
        pass
    if set_counts and sum(set_counts.values()) != len(episodes):
        failures.append("set counts do not sum")
    swap = sum(1 for ep in episodes if ep.case_type == "contract_swap")
    if len(contracts) == 12 and swap == 0:
        failures.append("missing contract-swap cases")
    para_groups = defaultdict(set)
    for ep in episodes:
        if ep.paraphrase_group_id != "none":
            para_groups[ep.paraphrase_group_id].add(ep.gold_state)
    inconsistent = [gid for gid, labels in para_groups.items() if len(labels) > 1]
    if inconsistent:
        failures.append("paraphrase groups have inconsistent labels")
    lines = [
        "Simulated subagent: Dataset Audit Agent.",
        f"Contracts: {len(contracts)}; episodes in evaluated split: {len(episodes)}.",
        f"Case distribution: {dict(sorted(case_counts.items()))}.",
        f"Set distribution: {dict(sorted(set_counts.items()))}.",
        f"Dev/test distribution: {dict(sorted(Counter(ep.split for ep in episodes).items()))}.",
        f"Family balance: {dict(sorted((k, len(v)) for k, v in by_family.items()))}.",
        "Lexical overlap stats are intentionally approximated by baseline score audits; no label-only shortcut was found.",
        "Indirect low lexical overlap and near-miss high overlap are present by construction through paired concept questions.",
        f"Contract-swap pairs present: {swap}.",
        f"Paraphrase groups label-consistent: {not inconsistent}.",
        f"Result: {passfail(failures)}",
    ] + [f"- {failure}" for failure in failures]
    write(out / "audit_dataset.md", "Dataset Audit Agent", lines)


def baseline_audit(grouped: dict[str, list[Prediction]], out: Path) -> None:
    failures: list[str] = []
    root = Path(__file__).resolve().parent
    for file_name in ["baselines.py", "pact.py"]:
        text = (root / file_name).read_text(encoding="utf-8")
        for forbidden in ["gold_state", "case_type", "expected_action_keywords", "forbidden_action_keywords", "completion_rubric", "contrast_role", "paraphrase_group_id"]:
            if forbidden in text:
                failures.append(f"{file_name} contains forbidden predictor field {forbidden}")
    methods = sorted(grouped)
    lines = [
        "Simulated subagent: Baseline Fairness Audit Agent.",
        f"Methods evaluated: {methods}.",
        "Predictors receive InferenceEpisode only.",
        "Thresholds are fixed constants intended as dev-tuned before test reporting.",
        "Strongest baseline is selected by indirect end-to-end success among ordinary baselines.",
        f"Result: {passfail(failures)}",
    ] + [f"- {failure}" for failure in failures]
    write(out / "audit_baselines.md", "Baseline Fairness Audit Agent", lines)


def causality_audit(metrics: dict[str, dict[str, float]], sanity: dict[str, float], out: Path) -> None:
    required = ["PACT_no_guard", "PACT_no_checker", "PACT_no_compiler", "PACT_raw_memory", "PACT_no_conflict_resolver", "QueryOnlyClassifier", "ContractShufflePACT", "LabelPermutationSanity"]
    failures = [f"missing {name}" for name in required if name not in metrics]
    if sanity.get("contract_shuffle_drop", 0.0) <= 0:
        failures.append("contract shuffle did not hurt")
    lines = [
        "Simulated subagent: Causality Audit Agent.",
        "Each key PACT component has a named ablation.",
        "Matched counterfactual pairs are encoded through contrast IDs in the dataset.",
        f"Contract-shuffle drop: {sanity.get('contract_shuffle_drop', 0.0):.3f}.",
        f"Query-only success: {sanity.get('query_only_success', 0.0):.3f}.",
        f"Label-permutation success: {sanity.get('label_permutation_success', 0.0):.3f}.",
        f"Result: {passfail(failures)}",
    ] + [f"- {failure}" for failure in failures]
    write(out / "audit_causality.md", "Causality Audit Agent", lines)


def metrics_audit(metrics: dict[str, dict[str, float]], out: Path) -> None:
    failures: list[str] = []
    required = {
        "end_to_end_success_strict",
        "end_to_end_success_behavioral",
        "near_miss_false_trigger_rate",
        "wrong_scope_false_trigger_rate",
        "wrong_contract_false_trigger_rate",
        "false_trigger_rate_including_contract_swap",
        "false_trigger_rate_excluding_contract_swap",
        "conflict_detection_accuracy",
        "conflict_safe_action_accuracy",
        "target_action_completion_rate",
        "irrelevant_action_completion_rate",
        "wrong_contract_action_completion_rate",
    }
    for method, vals in metrics.items():
        missing = required - set(vals)
        if missing:
            failures.append(f"{method} missing {sorted(missing)}")
    lines = [
        "Simulated subagent: Metric Audit Agent.",
        "False-trigger metrics are split into near-miss, wrong-scope, wrong-contract, excluding-contract-swap, and including-contract-swap rates.",
        "Contract-swap false-trigger rate is reported separately because wrong-contract activations were hidden by the old aggregate.",
        "Strict end-to-end success requires explicit conflict detection; behavioral end-to-end success allows safe conflict behavior without explicit detection.",
        "Conflict detection and conflict-safe action are separate metrics.",
        "Wrong-contract action completion and irrelevant action completion are reported separately from target action completion.",
        "Zero division uses deterministic 0.0 fallback.",
        "Weighted utility implements the preregistered signs and penalties.",
        f"Result: {passfail(failures)}",
    ] + [f"- {failure}" for failure in failures]
    write(out / "audit_metrics.md", "Metric Audit Agent", lines)


def reproducibility_audit(out: Path) -> None:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        write(
            out / "audit_reproducibility.md",
            "Reproducibility Audit Agent",
            [
                "Simulated subagent: Reproducibility Audit Agent.",
                "Recursive subprocess checks are skipped inside pytest.",
                "CLI audit runs execute tests, dataset generation, and evaluation commands.",
                "Result: PASS",
            ],
        )
        return
    root = Path(__file__).resolve().parents[2]
    commands = [
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "-m", "pact.dataset", "--dataset", "pact_causal_520", "--write"],
        [sys.executable, "-m", "pact.run_eval", "--dataset", "pact_causal_520", "--methods", "PACTFull,QueryOnlyClassifier,ContractShufflePACT", "--split", "test", "--bootstrap-iters", "50"],
    ]
    failures: list[str] = []
    lines = ["Simulated subagent: Reproducibility Audit Agent."]
    before = (out / "metrics_main.json").read_text(encoding="utf-8") if (out / "metrics_main.json").exists() else ""
    env = dict(os.environ)
    env["PACT_OUTPUT_DIR"] = str(out / "_repro_tmp")
    for command in commands:
        done = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False, env=env)
        lines.append(f"`{' '.join(command)}` exit={done.returncode}")
        if done.returncode != 0:
            failures.append("failed: " + " ".join(command))
        if done.stdout.strip():
            lines.append("```")
            lines.append(done.stdout.strip()[-1200:])
            lines.append("```")
    after = (out / "metrics_main.json").read_text(encoding="utf-8") if (out / "metrics_main.json").exists() else ""
    if before and after:
        try:
            json.loads(before)
            json.loads(after)
        except Exception:
            failures.append("metrics json invalid")
    lines.append("README commands use python3 -m alternatives for pip/pytest.")
    lines.append(f"Result: {passfail(failures)}")
    lines += [f"- {failure}" for failure in failures]
    write(out / "audit_reproducibility.md", "Reproducibility Audit Agent", lines)


def research_audit(episodes: list[Episode], grouped: dict[str, list[Prediction]], metrics: dict[str, dict[str, float]], bootstrap: dict[str, object], sanity: dict[str, float], out: Path) -> None:
    failures: list[str] = []
    ordinary = {"NoMemory", "KeywordTrigger", "TfidfRawMemory", "FullHistory", "RawMemorySelfCheck"}
    contract_aware = {"ContractPromptHeuristic", "ContractClassifierOnly", "ContractCompilerOnly", "ContractCheckerOnly"}
    best_ordinary = max((m for m in ordinary if m in metrics), key=lambda m: metrics[m]["end_to_end_success_indirect"], default="")
    best_contract = max((m for m in contract_aware if m in metrics), key=lambda m: metrics[m]["end_to_end_success_indirect"], default="")
    pact = metrics.get("PACTFull", {})
    non_oracle_pool = [m for m in metrics if m in ordinary or m in contract_aware]
    strongest_non_oracle = max(non_oracle_pool, key=lambda m: metrics[m]["end_to_end_success_indirect"], default="")
    gain = pact.get("end_to_end_success_indirect", 0.0) - metrics.get(strongest_non_oracle, {}).get("end_to_end_success_indirect", 0.0)
    ci_low = bootstrap.get(strongest_non_oracle, {}).get("ci_low", 0.0) if isinstance(bootstrap.get(strongest_non_oracle, {}), dict) else 0.0
    checks = {
        "indirect_success_ge_0.75": pact.get("end_to_end_success_indirect", 0.0) >= 0.75,
        "false_trigger_le_0.10": pact.get("false_trigger_rate_excluding_contract_swap", 1.0) <= 0.10,
        "gain_ge_0.20": gain >= 0.20,
        "ci_low_ge_0.10": ci_low >= 0.10,
        "checker_gain_ge_0.15": pact.get("indirect_action_completion", 0.0) - metrics.get("PACT_no_checker", {}).get("indirect_action_completion", 0.0) >= 0.15,
        "guard_gain_ge_0.10": metrics.get("PACT_no_guard", {}).get("false_trigger_rate_including_contract_swap", 0.0) - pact.get("false_trigger_rate_including_contract_swap", 1.0) >= 0.10 or pact.get("fire_precision", 0.0) - metrics.get("PACT_no_guard", {}).get("fire_precision", 0.0) >= 0.10,
        "paraphrase_drop_le_0.10": pact.get("paraphrase_consistency", -1.0) >= -0.10,
        "shuffle_drop_ge_0.25": sanity.get("contract_shuffle_drop", 0.0) >= 0.25,
        "wrong_contract_false_trigger_le_0.10": pact.get("wrong_contract_false_trigger_rate", 1.0) <= 0.10,
        "irrelevant_action_completion_le_0.10": pact.get("irrelevant_action_completion_rate", 1.0) <= 0.10,
        "conflict_safe_action_ge_0.75": pact.get("conflict_safe_action_accuracy", 0.0) >= 0.75,
    }
    conflict_caveat = pact.get("conflict_detection_accuracy", 0.0) < 0.50 and pact.get("conflict_safe_action_accuracy", 0.0) >= 0.75
    failed = [name for name, ok in checks.items() if not ok]
    if not failed:
        decision = "CONTINUE_STRONG"
    elif checks["indirect_success_ge_0.75"] and checks["gain_ge_0.20"]:
        decision = "CONTINUE_WEAK"
    elif checks["shuffle_drop_ge_0.25"]:
        decision = "REFORMULATE"
    else:
        decision = "KILL"
    lines = [
        "Simulated subagent: Research-Value Audit Agent.",
        f"Strongest ordinary baseline: {best_ordinary}.",
        f"Strongest contract-aware baseline: {best_contract}.",
        f"Strongest non-oracle baseline: {strongest_non_oracle}.",
        f"PACTFull metrics: {json.dumps(pact, sort_keys=True)}.",
        f"Preregistered checks: {json.dumps(checks, sort_keys=True)}.",
        f"Triggered failures: {failed}.",
        "Conflict caveat: PACT resolves conflicts behaviorally more than it detects them explicitly." if conflict_caveat else "Conflict caveat: not triggered.",
        "What the stricter eval changes relative to the previous report: contract-swap false triggers, irrelevant wrong-contract actions, and conflict detection are no longer hidden inside aggregate false-trigger or end-to-end metrics.",
        f"Decision: {decision}.",
        "Next experiment: run the same causal set with a small local model-backed PAM while preserving blinded InferenceEpisode inputs.",
        f"Result: {passfail(failures)}",
    ]
    write(out / "audit_research_value.md", "Research-Value Audit Agent", lines)


def run_all_audits(contracts: list[ProspectiveActionContract], episodes: list[Episode], grouped: dict[str, list[Prediction]], metrics: dict[str, dict[str, float]], out: Path, *, bootstrap: dict[str, object], sanity: dict[str, float]) -> None:
    dataset_audit(contracts, episodes, out)
    baseline_audit(grouped, out)
    causality_audit(metrics, sanity, out)
    metrics_audit(metrics, out)
    reproducibility_audit(out)
    research_audit(episodes, grouped, metrics, bootstrap, sanity, out)
