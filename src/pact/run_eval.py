"""Evaluation runner for PACT-Causal-520."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from pact.audits import run_all_audits
from pact.baselines import (
    FullHistory,
    KeywordTrigger,
    LLMStub,
    LabelPermutationSanity,
    NoMemory,
    QueryOnlyClassifier,
    RawMemorySelfCheck,
    TfidfRawMemory,
)
from pact.dataset import DEFAULT_DATASET, ROOT, load_contracts, load_episodes, write_dataset
from pact.pact import METHOD_NAMES, R2Config, get_method
from pact.schema import Episode, Prediction
from pact.scoring import activated, binary_successes, episode_success, episode_success_behavioral, format_summary, group_predictions, score_method
from pact.stats import cluster_bootstrap_diff, holm, mcnemar, permutation_test

OUTPUT_DIR = Path(os.environ.get("PACT_OUTPUT_DIR", ROOT / "outputs"))


R2_PRIMARY_METRICS = [
    "wrong_contract_false_trigger_rate",
    "indirect_end_to_end_success_strict",
    "indirect_action_completion",
    "false_trigger_rate_including_contract_swap",
    "false_trigger_rate_excluding_contract_swap",
    "conflict_detection_accuracy",
    "conflict_safe_action_accuracy",
    "target_action_completion_rate",
    "irrelevant_action_completion_rate",
    "end_to_end_success_strict",
    "end_to_end_success_behavioral",
    "near_miss_false_trigger_rate",
    "wrong_scope_false_trigger_rate",
    "paraphrase_consistency",
    "naturalistic_success",
    "scheduling_family_success",
]


def build_methods(methods: str, r2_config: R2Config | None = None):
    names = METHOD_NAMES if methods == "all" else [m.strip() for m in methods.split(",") if m.strip()]
    if methods == "r2":
        names = [
            "PACTFull_current",
            "PACT_specificity_gate",
            "PACT_conditional_bonus",
            "PACT_intent_family_gate",
            "PACT_state_action_split",
            "PACT_R2_full",
            "PACT_no_guard",
            "PACT_no_checker",
            "PACT_no_compiler",
            "PACT_raw_memory",
            "PACT_no_conflict_resolver",
            "ContractCompilerOnly",
            "TfidfRawMemory",
            "QueryOnlyClassifier",
            "ContractShufflePACT",
        ]
    out = []
    for name in names:
        if name == "NoMemory":
            out.append(NoMemory())
        elif name == "KeywordTrigger":
            out.append(KeywordTrigger())
        elif name == "TfidfRawMemory":
            out.append(TfidfRawMemory())
        elif name == "FullHistory":
            out.append(FullHistory())
        elif name == "RawMemorySelfCheck":
            out.append(RawMemorySelfCheck())
        elif name == "QueryOnlyClassifier":
            out.append(QueryOnlyClassifier())
        elif name == "LabelPermutationSanity":
            out.append(LabelPermutationSanity())
        elif name.startswith("LLM"):
            out.append(LLMStub(name))
        else:
            out.append(get_method(name, r2_config))
    return out


def filter_split(episodes: list[Episode], split: str) -> list[Episode]:
    if split == "all":
        return episodes
    return [ep for ep in episodes if ep.split == split]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def needs_r2(methods: str) -> bool:
    if methods in {"all", "r2"}:
        return True
    return any(name.strip().startswith("PACT_") or name.strip() == "PACTFull_current" for name in methods.split(","))


def tune_r2_config(contracts, all_episodes: list[Episode], seed: int) -> R2Config:
    dev = [ep for ep in all_episodes if ep.split == "dev"]
    rows = []
    best: tuple[float, dict[str, object], R2Config] | None = None
    for specificity_floor in [0.08, 0.10, 0.12, 0.15, 0.18]:
        for base_floor in [0.10, 0.15, 0.20, 0.25]:
            for bonus_multiplier in [0.15, 0.25, 0.35]:
                for intent_threshold in [0.40, 0.50, 0.60]:
                    config = R2Config(specificity_floor, base_floor, bonus_multiplier, intent_threshold)
                    method = get_method("PACT_R2_full", config)
                    preds = [method.predict(contracts, ep.to_inference()) for ep in dev]
                    metrics = score_method(dev, preds)
                    objective = (
                        metrics["indirect_end_to_end_success_strict"]
                        - 2.0 * metrics["wrong_contract_false_trigger_rate"]
                        - metrics["false_trigger_rate_including_contract_swap"]
                        + 0.5 * metrics["conflict_detection_accuracy"]
                    )
                    constraints = (
                        metrics["wrong_contract_false_trigger_rate"] <= 0.12
                        and metrics["false_trigger_rate_including_contract_swap"] <= 0.08
                        and metrics["indirect_end_to_end_success_strict"] >= 0.80
                    )
                    row = {
                        "specificity_floor": specificity_floor,
                        "base_floor": base_floor,
                        "bonus_multiplier": bonus_multiplier,
                        "intent_family_confidence_threshold": intent_threshold,
                        "objective": objective,
                        "constraints_pass": constraints,
                        **{metric: metrics[metric] for metric in R2_PRIMARY_METRICS if metric in metrics},
                    }
                    rows.append(row)
                    rank = objective + (100.0 if constraints else 0.0)
                    if best is None or rank > best[0]:
                        best = (rank, row, config)
    write_csv(OUTPUT_DIR / "r2_threshold_search.csv", rows)
    assert best is not None
    best_row = best[1]
    payload = {
        "config": {
            "specificity_floor": best[2].specificity_floor,
            "base_floor": best[2].base_floor,
            "bonus_multiplier": best[2].bonus_multiplier,
            "intent_family_confidence_threshold": best[2].intent_family_confidence_threshold,
        },
        "dev_result": best_row,
        "seed": seed,
        "note": "Selected using dev split only; reused unchanged for test/all reporting.",
    }
    (OUTPUT_DIR / "r2_best_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return best[2]


def metric_rows(episodes: list[Episode], grouped: dict[str, list[Prediction]], key: str) -> list[dict[str, object]]:
    rows = []
    values = sorted({getattr(ep, key) for ep in episodes})
    for method, preds in grouped.items():
        by_id = {p.episode_id: p for p in preds}
        for value in values:
            eps = [ep for ep in episodes if getattr(ep, key) == value]
            rows.append({"method": method, key: value, **score_method(eps, [by_id[ep.episode_id] for ep in eps])})
    return rows


def paired_outputs(episodes: list[Episode], grouped: dict[str, list[Prediction]], bootstrap_iters: int, seed: int) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object], dict[str, object]]:
    pact = grouped.get("PACTFull", [])
    if not pact:
        return [], {}, {}, {}
    pact_bits = binary_successes(episodes, pact)
    comparisons = [name for name in grouped if name != "PACTFull"]
    pair_rows = []
    boot: dict[str, object] = {}
    mc: dict[str, object] = {}
    perm: dict[str, object] = {}
    pvals = {}
    for name in comparisons:
        other = grouped[name]
        bits = binary_successes(episodes, other)
        diff = sum(a - b for a, b in zip(pact_bits, bits)) / len(pact_bits)
        pair_rows.append({"comparison": f"PACTFull_vs_{name}", "absolute_gain": diff})
        boot[name] = cluster_bootstrap_diff(episodes, pact, other, iters=bootstrap_iters, seed=seed)
        mc[name] = mcnemar(pact_bits, bits)
        perm[name] = permutation_test(pact_bits, bits, iters=min(bootstrap_iters, 2000), seed=seed)
        pvals[name] = mc[name]["p"]
    adjusted = holm(pvals)
    for row in pair_rows:
        name = row["comparison"].replace("PACTFull_vs_", "")
        row["mcnemar_holm_p"] = adjusted[name]
        row["bootstrap_ci_low"] = boot[name]["ci_low"]
        row["bootstrap_ci_high"] = boot[name]["ci_high"]
    return pair_rows, boot, mc, perm


def write_errors(episodes: list[Episode], predictions: list[Prediction], path: Path) -> None:
    by_id = {p.episode_id: p for p in predictions}
    rows = []
    for ep in episodes:
        pred = by_id[ep.episode_id]
        if not episode_success(ep, pred):
            rows.append({"episode_id": ep.episode_id, "family": ep.family, "set_type": ep.set_type, "case_type": ep.case_type, "current_query": ep.current_query, "gold_state": ep.gold_state, **pred.to_dict()})
    write_csv(path, rows)


DIFFERENCE_METRICS = [
    "end_to_end_success_indirect",
    "indirect_action_completion",
    "false_trigger_rate_including_contract_swap",
    "wrong_contract_false_trigger_rate",
    "conflict_detection_accuracy",
    "conflict_safe_action_accuracy",
    "target_action_completion_rate",
    "irrelevant_action_completion_rate",
]


def strongest_contract_baseline_name(grouped: dict[str, list[Prediction]], episodes: list[Episode]) -> str:
    names = {"ContractPromptHeuristic", "ContractClassifierOnly", "ContractCompilerOnly", "ContractCheckerOnly"}
    candidates = [name for name in grouped if name in names]
    if not candidates:
        return ""
    return max(candidates, key=lambda name: score_method(episodes, grouped[name])["end_to_end_success_indirect"])


def write_method_differences(episodes: list[Episode], grouped: dict[str, list[Prediction]], metrics: dict[str, dict[str, float]]) -> None:
    targets = [
        ("strongest_ordinary", strongest_baseline_name(grouped, episodes, ordinary=True)),
        ("strongest_contract_aware", strongest_contract_baseline_name(grouped, episodes)),
        ("ContractCompilerOnly", "ContractCompilerOnly"),
        ("PACT_no_checker", "PACT_no_checker"),
        ("PACT_no_guard", "PACT_no_guard"),
        ("PACT_raw_memory", "PACT_raw_memory"),
        ("ContractShufflePACT", "ContractShufflePACT"),
        ("QueryOnlyClassifier", "QueryOnlyClassifier"),
    ]
    rows = []
    pact = metrics.get("PACTFull", {})
    for label, method in targets:
        if not method or method not in metrics:
            continue
        row: dict[str, object] = {"comparison": f"PACTFull_vs_{label}", "baseline_method": method}
        for metric in DIFFERENCE_METRICS:
            row[f"{metric}_pact"] = pact.get(metric, 0.0)
            row[f"{metric}_baseline"] = metrics[method].get(metric, 0.0)
            row[f"{metric}_diff"] = pact.get(metric, 0.0) - metrics[method].get(metric, 0.0)
        rows.append(row)
    write_csv(OUTPUT_DIR / "method_differences.csv", rows)


def failure_reason(ep: Episode, pred: Prediction) -> str:
    if episode_success(ep, pred):
        return "success"
    if ep.case_type == "contract_swap" and activated(pred):
        return "wrong_contract_false_fire"
    if ep.gold_state == "conflict" and pred.predicted_state != "conflict":
        if episode_success_behavioral(ep, pred):
            return "behavioral_conflict_without_detection"
        return "conflict_failure"
    if ep.gold_state == "fire" and pred.predicted_contract_id != ep.gold_contract_id:
        return "wrong_contract_action"
    if ep.gold_state == "suppress" and activated(pred):
        return "false_trigger"
    return "missed_or_incomplete_action"


def write_r2_reports(episodes: list[Episode], grouped: dict[str, list[Prediction]], metrics: dict[str, dict[str, float]], sanity: dict[str, float]) -> None:
    variant_rows = []
    for method, vals in metrics.items():
        row = {"method": method, **{metric: vals.get(metric, 0.0) for metric in R2_PRIMARY_METRICS}}
        row["contract_shuffle_drop"] = sanity.get("contract_shuffle_drop", 0.0) if method == "PACT_R2_full" else ""
        variant_rows.append(row)
    write_csv(OUTPUT_DIR / "r2_variant_metrics.csv", variant_rows)
    cur = grouped.get("PACTFull_current") or grouped.get("PACTFull", [])
    r2 = grouped.get("PACT_R2_full", [])
    if not cur or not r2:
        for name in ["r2_error_transition.csv", "r2_fixed_errors.csv", "r2_new_errors.csv", "r2_contract_swap_errors.csv", "r2_conflict_errors.csv", "r2_family_metrics.csv", "manual_audit_r2_template.csv"]:
            write_csv(OUTPUT_DIR / name, [])
        return
    cur_by = {p.episode_id: p for p in cur}
    r2_by = {p.episode_id: p for p in r2}
    transitions = []
    for ep in episodes:
        cp = cur_by[ep.episode_id]
        rp = r2_by[ep.episode_id]
        cur_ok = episode_success(ep, cp)
        r2_ok = episode_success(ep, rp)
        if cur_ok and r2_ok:
            transition = "unchanged_correct"
        elif (not cur_ok) and r2_ok:
            transition = "fixed"
        elif cur_ok and not r2_ok:
            transition = "regressed"
        else:
            transition = "unchanged_wrong"
        if cp.predicted_state != rp.predicted_state or cp.predicted_contract_id != rp.predicted_contract_id or transition in {"fixed", "regressed"}:
            transitions.append({
                "episode_id": ep.episode_id,
                "family": ep.family,
                "case_type": ep.case_type,
                "set_type": ep.set_type,
                "gold_state": ep.gold_state,
                "gold_contract_id": ep.gold_contract_id,
                "current_pred_state": cp.predicted_state,
                "current_pred_contract_id": cp.predicted_contract_id,
                "r2_pred_state": rp.predicted_state,
                "r2_pred_contract_id": rp.predicted_contract_id,
                "current_success_strict": cur_ok,
                "r2_success_strict": r2_ok,
                "transition": transition,
                "failure_reason_current": failure_reason(ep, cp),
                "failure_reason_r2": failure_reason(ep, rp),
                "current_response": cp.response,
                "r2_response": rp.response,
            })
    write_csv(OUTPUT_DIR / "r2_error_transition.csv", transitions)
    write_csv(OUTPUT_DIR / "r2_fixed_errors.csv", [row for row in transitions if row["transition"] == "fixed"])
    write_csv(OUTPUT_DIR / "r2_new_errors.csv", [row for row in transitions if row["transition"] == "regressed"])
    write_csv(OUTPUT_DIR / "r2_contract_swap_errors.csv", [row for row in transitions if row["case_type"] == "contract_swap" and not row["r2_success_strict"]])
    conflict_rows = []
    for ep in episodes:
        if ep.gold_state == "conflict":
            rp = r2_by[ep.episode_id]
            conflict_rows.append({
                "episode_id": ep.episode_id,
                "family": ep.family,
                "case_type": ep.case_type,
                "r2_pred_state": rp.predicted_state,
                "r2_pred_contract_id": rp.predicted_contract_id,
                "conflict_detected": rp.predicted_state == "conflict",
                "behavioral_success": episode_success_behavioral(ep, rp),
                "strict_success": episode_success(ep, rp),
                "response": rp.response,
            })
    write_csv(OUTPUT_DIR / "r2_conflict_errors.csv", [row for row in conflict_rows if not row["strict_success"]])
    family_rows = []
    for family in sorted({ep.family for ep in episodes}):
        eps = [ep for ep in episodes if ep.family == family]
        c_preds = [cur_by[ep.episode_id] for ep in eps]
        r_preds = [r2_by[ep.episode_id] for ep in eps]
        cm = score_method(eps, c_preds)
        rm = score_method(eps, r_preds)
        family_rows.append({
            "family": family,
            "PACTFull_current_strict_success": cm["end_to_end_success_strict"],
            "PACT_R2_full_strict_success": rm["end_to_end_success_strict"],
            "delta": rm["end_to_end_success_strict"] - cm["end_to_end_success_strict"],
            "wrong_contract_false_triggers_before": cm["wrong_contract_false_trigger_rate"],
            "wrong_contract_false_triggers_after": rm["wrong_contract_false_trigger_rate"],
            "conflict_detection_before": cm["conflict_detection_accuracy"],
            "conflict_detection_after": rm["conflict_detection_accuracy"],
            "indirect_success_before": cm["indirect_end_to_end_success_strict"],
            "indirect_success_after": rm["indirect_end_to_end_success_strict"],
        })
    write_csv(OUTPUT_DIR / "r2_family_metrics.csv", family_rows)
    write_manual_r2_template(episodes, grouped, transitions)


def write_manual_r2_template(episodes: list[Episode], grouped: dict[str, list[Prediction]], transitions: list[dict[str, object]]) -> None:
    cur = grouped.get("PACTFull_current") or grouped.get("PACTFull", [])
    r2 = grouped.get("PACT_R2_full", [])
    baseline_name = strongest_baseline_name(grouped, episodes, ordinary=True)
    baseline = grouped.get(baseline_name, [])
    cur_by = {p.episode_id: p for p in cur}
    r2_by = {p.episode_id: p for p in r2}
    base_by = {p.episode_id: p for p in baseline}
    by_id = {ep.episode_id: ep for ep in episodes}
    chosen: list[tuple[str, Episode]] = []
    for row in transitions:
        if row["transition"] == "fixed":
            chosen.append(("current_error_fixed_by_r2", by_id[row["episode_id"]]))
        if row["transition"] == "regressed":
            chosen.append(("r2_new_regression", by_id[row["episode_id"]]))
    for ep in episodes:
        if ep.episode_id in r2_by and ep.case_type == "contract_swap" and not episode_success(ep, r2_by[ep.episode_id]):
            chosen.append(("remaining_r2_contract_swap_failure", ep))
        if ep.episode_id in r2_by and ep.gold_state == "conflict" and not episode_success(ep, r2_by[ep.episode_id]):
            chosen.append(("remaining_r2_conflict_failure", ep))
    baseline_fail = [ep for ep in episodes if ep.episode_id in base_by and ep.episode_id in r2_by and not episode_success(ep, base_by[ep.episode_id]) and episode_success(ep, r2_by[ep.episode_id])][:30]
    successes = [ep for ep in episodes if ep.episode_id in r2_by and episode_success(ep, r2_by[ep.episode_id])][:30]
    for ep in baseline_fail:
        chosen.append(("baseline_failure_r2_success", ep))
    for ep in successes:
        chosen.append(("r2_success_sample", ep))
    rows = []
    seen = set()
    for group, ep in chosen:
        key = (group, ep.episode_id)
        if key in seen:
            continue
        seen.add(key)
        cp = cur_by.get(ep.episode_id)
        rp = r2_by.get(ep.episode_id)
        bp = base_by.get(ep.episode_id)
        rows.append({
            "audit_group": group,
            "episode_id": ep.episode_id,
            "family": ep.family,
            "set_type": ep.set_type,
            "case_type": ep.case_type,
            "current_query": ep.current_query,
            "history_summary": ep.history_summary,
            "gold_state": ep.gold_state,
            "gold_contract_id": ep.gold_contract_id,
            "current_pred_state": cp.predicted_state if cp else "",
            "current_pred_contract_id": cp.predicted_contract_id if cp else "",
            "r2_pred_state": rp.predicted_state if rp else "",
            "r2_pred_contract_id": rp.predicted_contract_id if rp else "",
            "current_response": cp.response if cp else "",
            "r2_response": rp.response if rp else "",
            "strongest_baseline_method": baseline_name,
            "strongest_baseline_predicted_state": bp.predicted_state if bp else "",
            "strongest_baseline_predicted_contract_id": bp.predicted_contract_id if bp else "",
            "strongest_baseline_response": bp.response if bp else "",
            "audit_question_gold_label_valid": "",
            "audit_question_r2_fixed_for_right_reason": "",
            "audit_question_r2_regression_over_suppression": "",
            "audit_question_wrong_contract_issue": "",
            "audit_question_conflict_detected_or_behavior_only": "",
            "audit_question_action_completion_meaningful": "",
            "audit_question_baseline_unfairly_disadvantaged": "",
            "manual_notes": "",
        })
    write_csv(OUTPUT_DIR / "manual_audit_r2_template.csv", rows)


def write_manual_sample(episodes: list[Episode], grouped: dict[str, list[Prediction]]) -> None:
    pact = grouped.get("PACTFull", grouped.get("PACTFull_current", []))
    baseline_name = strongest_baseline_name(grouped, episodes, ordinary=True)
    baseline = grouped.get(baseline_name, [])
    pact_by = {p.episode_id: p for p in pact}
    base_by = {p.episode_id: p for p in baseline}
    chosen: list[tuple[str, Episode, Prediction]] = []
    pact_errors = [ep for ep in episodes if ep.episode_id in pact_by and not episode_success(ep, pact_by[ep.episode_id])]
    swap_errors = [ep for ep in pact_errors if ep.case_type == "contract_swap"]
    conflict_cases = [ep for ep in episodes if ep.gold_state == "conflict" and ep.episode_id in pact_by]
    successes = [ep for ep in episodes if ep.episode_id in pact_by and episode_success(ep, pact_by[ep.episode_id])][:30]
    base_fail_pact_success = [ep for ep in episodes if ep.episode_id in base_by and ep.episode_id in pact_by and not episode_success(ep, base_by[ep.episode_id]) and episode_success(ep, pact_by[ep.episode_id])][:30]
    false_triggers_all = [ep for ep in episodes if ep.episode_id in pact_by and ep.gold_state == "suppress" and activated(pact_by[ep.episode_id])]
    false_triggers = false_triggers_all if len(false_triggers_all) < 50 else false_triggers_all[:50]
    safety = [ep for ep in episodes if ep.priority_expectation == "safety" and ep.episode_id in pact_by and not episode_success(ep, pact_by[ep.episode_id])]
    buckets = [
        ("pact_error", pact_errors),
        ("pact_contract_swap_error", swap_errors),
        ("pact_conflict_case", conflict_cases),
        ("pact_success_sample", successes),
        ("baseline_failure_pact_success", base_fail_pact_success),
        ("safety_critical_failure", safety),
        ("false_trigger", false_triggers),
    ]
    for group, bucket in buckets:
        for ep in bucket:
            chosen.append((group, ep, pact_by[ep.episode_id]))
    seen = set()
    rows = []
    for group, ep, pred in chosen:
        key = (group, ep.episode_id, pred.method)
        if key in seen:
            continue
        seen.add(key)
        base = base_by.get(ep.episode_id)
        rows.append({
            "audit_group": group,
            "episode_id": ep.episode_id,
            "family": ep.family,
            "set_type": ep.set_type,
            "case_type": ep.case_type,
            "current_query": ep.current_query,
            "history_summary": ep.history_summary,
            "gold_state": ep.gold_state,
            "gold_contract_id": ep.gold_contract_id,
            "method": pred.method,
            "predicted_state": pred.predicted_state,
            "predicted_contract_id": pred.predicted_contract_id,
            "response": pred.response,
            "strongest_baseline_method": baseline_name,
            "strongest_baseline_predicted_state": base.predicted_state if base else "",
            "strongest_baseline_predicted_contract_id": base.predicted_contract_id if base else "",
            "strongest_baseline_response": base.response if base else "",
            "audit_question_gold_label_valid": "",
            "audit_question_solved_for_right_reason": "",
            "audit_question_action_completion_meaningful": "",
            "audit_question_baseline_unfairly_disadvantaged": "",
            "audit_question_wrong_contract_issue": "",
            "audit_question_conflict_detection_vs_safe_action": "",
            "manual_notes": "",
        })
    write_csv(OUTPUT_DIR / "manual_audit_sample.csv", rows)
    write_csv(OUTPUT_DIR / "manual_audit_completed_template.csv", rows)


def strongest_baseline_name(grouped: dict[str, list[Prediction]], episodes: list[Episode], *, ordinary: bool) -> str:
    ordinary_names = {"NoMemory", "KeywordTrigger", "TfidfRawMemory", "FullHistory", "RawMemorySelfCheck"}
    excluded = {"PACTFull", "ContractShufflePACT", "LabelPermutationSanity"} | {name for name in grouped if name.startswith("LLM")}
    candidates = [name for name in grouped if (name in ordinary_names if ordinary else name not in ordinary_names and name not in excluded and not name.startswith("PACT_"))]
    if not candidates:
        return ""
    return max(candidates, key=lambda name: score_method(episodes, grouped[name])["end_to_end_success_indirect"])


def run(dataset: str = DEFAULT_DATASET, methods: str = "all", split: str = "test", audit: bool = False, bootstrap_iters: int = 1000, seed: int = 0) -> dict[str, dict[str, float]]:
    write_dataset(dataset)
    contracts = load_contracts(dataset)
    all_episodes = load_episodes(dataset)
    episodes = filter_split(all_episodes, split)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    r2_config = tune_r2_config(contracts, all_episodes, seed) if needs_r2(methods) and dataset == "pact_causal_520" else R2Config()
    predictions: list[Prediction] = []
    for method in build_methods(methods, r2_config):
        for ep in episodes:
            predictions.append(method.predict(contracts, ep.to_inference()))
    grouped = group_predictions(predictions)
    metrics = {name: score_method(episodes, preds) for name, preds in grouped.items()}
    write_csv(OUTPUT_DIR / "predictions.csv", [p.to_dict() for p in predictions])
    (OUTPUT_DIR / "metrics_main.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Legacy compatibility for older scripts.
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(OUTPUT_DIR / "metrics_by_family.csv", metric_rows(episodes, grouped, "family"))
    write_csv(OUTPUT_DIR / "metrics_by_case_type.csv", metric_rows(episodes, grouped, "case_type"))
    write_csv(OUTPUT_DIR / "metrics_by_set_type.csv", metric_rows(episodes, grouped, "set_type"))
    write_method_differences(episodes, grouped, metrics)
    pair_rows, boot, mc, perm = paired_outputs(episodes, grouped, bootstrap_iters, seed)
    write_csv(OUTPUT_DIR / "paired_comparisons.csv", pair_rows)
    (OUTPUT_DIR / "bootstrap_ci.json").write_text(json.dumps(boot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "mcnemar_tests.json").write_text(json.dumps(mc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "permutation_tests.json").write_text(json.dumps(perm, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sanity = {
        "query_only_success": metrics.get("QueryOnlyClassifier", {}).get("end_to_end_success", 0.0),
        "contract_shuffle_drop": metrics.get("PACTFull", {}).get("end_to_end_success", 0.0) - metrics.get("ContractShufflePACT", {}).get("end_to_end_success", 0.0),
        "r2_contract_shuffle_drop": metrics.get("PACT_R2_full", {}).get("end_to_end_success", 0.0) - metrics.get("ContractShufflePACT", {}).get("end_to_end_success", 0.0),
        "label_permutation_success": metrics.get("LabelPermutationSanity", {}).get("end_to_end_success", 0.0),
    }
    (OUTPUT_DIR / "sanity_checks.json").write_text(json.dumps(sanity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if "PACTFull" in grouped:
        write_errors(episodes, grouped["PACTFull"], OUTPUT_DIR / "errors_pact.csv")
    sb = strongest_baseline_name(grouped, episodes, ordinary=True)
    if sb:
        write_errors(episodes, grouped[sb], OUTPUT_DIR / "errors_strongest_baseline.csv")
    else:
        write_csv(OUTPUT_DIR / "errors_strongest_baseline.csv", [])
    write_manual_sample(episodes, grouped)
    write_r2_reports(episodes, grouped, metrics, sanity)
    summary = format_summary(metrics)
    (OUTPUT_DIR / "audit_report.md").write_text("# PACT-Causal-520 Evaluation\n\n" + summary + "\n", encoding="utf-8")
    if audit:
        run_all_audits(contracts, episodes, grouped, metrics, OUTPUT_DIR, bootstrap=boot, sanity=sanity)
    print(summary)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=["pact100_legacy", "pact_causal_520"])
    parser.add_argument("--methods", default="all")
    parser.add_argument("--split", default="test", choices=["dev", "test", "all"])
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.dataset, args.methods, args.split, args.audit, args.bootstrap_iters, args.seed)


if __name__ == "__main__":
    main()
