"""Evaluation runner for PACT-Causal-520."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
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
from pact.baselines import allowed_contracts, best_contract, compiled_plan, contains, overlap
from pact.pact import CONFLICT_OPPOSITION, FAMILY_COMPILER, NEAR, WRONG, METHOD_NAMES, R2Config, detect_intent_family, get_method
from pact.schema import Episode, InferenceEpisode, Prediction, ProspectiveActionContract
from pact.scoring import action_completed, activated, binary_successes, correct_contract, episode_success, episode_success_behavioral, format_summary, group_predictions, score_method, target_action_completed
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

D3_METHODS = [
    "PACTFull_current",
    "PACT_R2_full",
    "PACT_intent_plus_state",
    "PACT_intent_plus_state_checker",
    "PACT_intent_plus_state_family_compiler",
    "LearnedPAM",
    "LearnedPAM_plus_checker",
    "LearnedPAM_plus_family_compiler",
    "QueryOnlyClassifier",
    "ContractOnlyClassifier",
    "QueryPlusFamilyClassifier",
    "QueryPlusContractClassifier",
    "QueryPlusWrongContractOnly",
    "ContractShufflePACT",
]


def build_methods(methods: str, r2_config: R2Config | None = None):
    names = METHOD_NAMES if methods == "all" else [m.strip() for m in methods.split(",") if m.strip()]
    if methods == "d3":
        names = D3_METHODS
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
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def needs_r2(methods: str) -> bool:
    if methods in {"all", "r2", "d3"}:
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


def _pred_by(grouped: dict[str, list[Prediction]], method: str) -> dict[str, Prediction]:
    return {p.episode_id: p for p in grouped.get(method, [])}


def _fixed_regressed(episodes: list[Episode], base: list[Prediction], other: list[Prediction]) -> tuple[int, int]:
    base_by = {p.episode_id: p for p in base}
    other_by = {p.episode_id: p for p in other}
    fixed = regressed = 0
    for ep in episodes:
        if ep.episode_id not in base_by or ep.episode_id not in other_by:
            continue
        base_ok = episode_success(ep, base_by[ep.episode_id])
        other_ok = episode_success(ep, other_by[ep.episode_id])
        fixed += int((not base_ok) and other_ok)
        regressed += int(base_ok and (not other_ok))
    return fixed, regressed


def _d3_metrics_row(name: str, vals: dict[str, float], episodes: list[Episode], preds: list[Prediction], current: list[Prediction], sanity: dict[str, float]) -> dict[str, object]:
    fixed, regressed = _fixed_regressed(episodes, current, preds) if current and preds else (0, 0)
    return {
        "method": name,
        "indirect_strict_success": vals.get("indirect_end_to_end_success_strict", 0.0),
        "wrong_contract_false_trigger_rate": vals.get("wrong_contract_false_trigger_rate", 0.0),
        "conflict_detection_accuracy": vals.get("conflict_detection_accuracy", 0.0),
        "conflict_safe_action_accuracy": vals.get("conflict_safe_action_accuracy", 0.0),
        "target_completion": vals.get("target_action_completion_rate", 0.0),
        "strict_e2e": vals.get("end_to_end_success_strict", 0.0),
        "behavioral_e2e": vals.get("end_to_end_success_behavioral", 0.0),
        "naturalistic_strict_success": vals.get("naturalistic_success", 0.0),
        "contract_shuffle_drop": sanity.get("contract_shuffle_drop", 0.0) if name in {"PACTFull_current", "PACT_R2_full"} else "",
        "fixed_vs_current": fixed,
        "regressed_vs_current": regressed,
    }


def _oracle_response(ep: Episode, contract: ProspectiveActionContract, *, compiler: bool = False) -> str:
    bits = list(dict.fromkeys(ep.expected_action_keywords))
    if compiler:
        return f"{compiled_plan(contract)} " + " ".join(bits)
    return "Oracle diagnostic action: " + " ".join(bits)


def oracle_predictions(contracts: list[ProspectiveActionContract], episodes: list[Episode], grouped: dict[str, list[Prediction]]) -> dict[str, list[Prediction]]:
    by_contract = {c.contract_id: c for c in contracts}
    current = _pred_by(grouped, "PACTFull_current") or _pred_by(grouped, "PACTFull")
    r2 = _pred_by(grouped, "PACT_R2_full")
    out: dict[str, list[Prediction]] = defaultdict(list)
    for ep in episodes:
        cur = current.get(ep.episode_id)
        rp = r2.get(ep.episode_id, cur)
        gold_contract = by_contract.get(ep.gold_contract_id) or by_contract.get(ep.target_contract_id) or allowed_contracts(contracts, ep.to_inference())[0]
        selected_contract = by_contract.get(cur.predicted_contract_id, gold_contract) if cur else gold_contract
        if ep.gold_state in {"fire", "conflict", "already_satisfied"}:
            state = ep.gold_state
            contract_id = gold_contract.contract_id
            response = _oracle_response(ep, gold_contract)
            completed = ep.gold_state in {"fire", "conflict"}
        else:
            state = "suppress"
            contract_id = "none"
            response = "Oracle diagnostic suppresses: no target contract should fire."
            completed = False
        out["OracleContractSelection"].append(Prediction("OracleContractSelection", ep.episode_id, contract_id, state, 1.0, response, completed, False, "ORACLE_UNFAIR: gold contract selection"))

        state = ep.gold_state
        response = _oracle_response(ep, selected_contract) if state in {"fire", "conflict"} else "Oracle diagnostic state suppresses."
        out["OracleState"].append(Prediction("OracleState", ep.episode_id, selected_contract.contract_id if state != "suppress" else "none", state, 1.0, response, state in {"fire", "conflict"}, False, "ORACLE_UNFAIR: gold state"))

        if ep.gold_state == "conflict":
            out["OracleConflictState"].append(Prediction("OracleConflictState", ep.episode_id, selected_contract.contract_id, "conflict", 1.0, _oracle_response(ep, selected_contract), True, False, "ORACLE_UNFAIR: conflict state only"))
        elif cur:
            out["OracleConflictState"].append(Prediction("OracleConflictState", ep.episode_id, cur.predicted_contract_id, cur.predicted_state, cur.confidence, cur.response, cur.action_completed, cur.repaired, "ORACLE_UNFAIR: conflict state only"))

        source = rp or cur
        if source and ep.gold_state in {"fire", "conflict"}:
            response = _oracle_response(ep, by_contract.get(source.predicted_contract_id, gold_contract))
            completed = True
        else:
            response = source.response if source else "Oracle target completion noop."
            completed = source.action_completed if source else False
        if source:
            out["OracleTargetCompletion"].append(Prediction("OracleTargetCompletion", ep.episode_id, source.predicted_contract_id, source.predicted_state, source.confidence, response, completed, source.repaired, "ORACLE_UNFAIR: perfect target-action completion"))

        comp_contract = by_contract.get(source.predicted_contract_id, gold_contract) if source else gold_contract
        comp_response = _oracle_response(ep, comp_contract, compiler=True) if source and source.predicted_state in {"fire", "conflict"} else "Oracle compiler suppresses."
        if source:
            out["OracleCompiler"].append(Prediction("OracleCompiler", ep.episode_id, source.predicted_contract_id, source.predicted_state, source.confidence, comp_response, source.predicted_state in {"fire", "conflict"}, True, "ORACLE_UNFAIR: rubric-derived compiler"))
    return dict(out)


def _largest_gain_area(base: dict[str, float], vals: dict[str, float]) -> tuple[str, str]:
    gains = {
        "contract_selection": base.get("wrong_contract_false_trigger_rate", 0.0) - vals.get("wrong_contract_false_trigger_rate", 0.0),
        "state_conflict": vals.get("conflict_detection_accuracy", 0.0) - base.get("conflict_detection_accuracy", 0.0),
        "compiler_checker": vals.get("target_action_completion_rate", 0.0) - base.get("target_action_completion_rate", 0.0),
        "overall_strict": vals.get("end_to_end_success_strict", 0.0) - base.get("end_to_end_success_strict", 0.0),
    }
    area = max(gains, key=gains.get)
    bottleneck = {
        "contract_selection": "wrong contract selection / applicability",
        "state_conflict": "state and conflict prediction",
        "compiler_checker": "action compiler, checker, or rubric alignment",
        "overall_strict": "mixed end-to-end ceiling",
    }[area]
    return area, bottleneck


def conflict_subtype(ep: Episode) -> str:
    text = f"{ep.history_summary} {ep.current_query}".lower()
    if any(term in text for term in ("current", "latest", "verify", "source", "memory only", "without checking")):
        return "current_fact_verification_conflict"
    if any(term in text for term in ("diagnosis", "medical", "legal", "certain", "guarantee", "advice")):
        return "medical_or_legal_certainty_conflict"
    if any(term in text for term in ("unsafe", "allergy", "allergic", "peanut", "nut", "security", "secret", "cookie")):
        return "unsafe_recommendation_conflict"
    if any(term in text for term in ("style", "rewrite", "grandiose", "legalistic", "tone")):
        return "style_conflict"
    if any(term in text for term in ("ignore", "skip", "do not mention", "don't mention", "do not check")):
        return "skip_safety_or_verification"
    if len(ep.available_contract_ids) > 1 and ep.gold_state == "conflict":
        return "multi_contract_conflict"
    return "other"


def _score_anatomy_row(ep: Episode, pred: Prediction, contracts: list[ProspectiveActionContract]) -> dict[str, object]:
    by_contract = {c.contract_id: c for c in contracts}
    contract = by_contract.get(pred.predicted_contract_id) or allowed_contracts(contracts, ep.to_inference())[0]
    text = f"{ep.history_summary} {ep.current_query}"
    retrieval = max(overlap(contract.cue, text), overlap(contract.guard, text), overlap(contract.action, text))
    guard = overlap(contract.guard, text)
    action = overlap(contract.action + " " + contract.check, text)
    intent = detect_intent_family(ep.current_query, ep.history_summary)
    intent_score = 1.0 if intent.family == contract.family and intent.family != "unknown" else 0.0
    specificity = max(retrieval, guard, action)
    base = retrieval + 0.40 * guard + 0.25 * action
    bonus = 0.22 if "bonus=0.22" in pred.rationale or "bonus=0.2" in pred.rationale else 0.0
    final = pred.confidence
    if ep.gold_state == "fire" and pred.predicted_state != "fire":
        bucket = "missed_indirect_trigger" if ep.case_type == "indirect_trigger" else "missed_fire"
    elif ep.gold_state == "conflict" and pred.predicted_state == "fire":
        bucket = "conflict_fire"
    elif ep.case_type == "contract_swap" and activated(pred):
        bucket = "false_fire_contract_swap"
    elif ep.case_type == "near_miss" and activated(pred):
        bucket = "false_fire_near_miss"
    elif ep.case_type == "wrong_scope" and activated(pred):
        bucket = "false_fire_wrong_scope"
    elif ep.gold_state == "fire":
        bucket = "true_fire"
    else:
        bucket = "other"
    return {
        "method": pred.method,
        "episode_id": ep.episode_id,
        "family": ep.family,
        "case_type": ep.case_type,
        "error_bucket": bucket,
        "retrieval_score": retrieval,
        "guard_score": guard,
        "action_score": action,
        "intent_family_score": intent_score,
        "bonus": bonus,
        "base_score": base,
        "specificity": specificity,
        "final_score": final,
        "fire_threshold": 0.14,
        "predicted_state": pred.predicted_state,
        "predicted_contract_id": pred.predicted_contract_id,
        "success": episode_success(ep, pred),
        "failure_type": failure_reason(ep, pred),
    }


def _average_rows(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    fields = ["retrieval_score", "guard_score", "action_score", "intent_family_score", "bonus", "base_score", "specificity", "final_score"]
    out = []
    for value in sorted({row[key] for row in rows}):
        bucket = [row for row in rows if row[key] == value]
        agg: dict[str, object] = {key: value, "n": len(bucket)}
        for field in fields:
            agg[field] = sum(float(row[field]) for row in bucket) / len(bucket)
        agg["success_rate"] = sum(1 for row in bucket if row["success"]) / len(bucket)
        out.append(agg)
    return out


def write_d3_reports(contracts: list[ProspectiveActionContract], episodes: list[Episode], grouped: dict[str, list[Prediction]], metrics: dict[str, dict[str, float]], sanity: dict[str, float]) -> None:
    current = grouped.get("PACTFull_current", grouped.get("PACTFull", []))
    r2 = grouped.get("PACT_R2_full", [])
    current_by = {p.episode_id: p for p in current}
    r2_by = {p.episode_id: p for p in r2}

    composition_methods = ["PACTFull_current", "PACT_R2_full", "PACT_intent_plus_state", "PACT_intent_plus_state_checker", "PACT_intent_plus_state_family_compiler"]
    comp_rows = []
    for name in composition_methods:
        if name not in grouped:
            continue
        fixed_cur, reg_cur = _fixed_regressed(episodes, current, grouped[name])
        fixed_r2, reg_r2 = _fixed_regressed(episodes, r2, grouped[name]) if r2 else (0, 0)
        vals = metrics[name]
        comp_rows.append({
            "method": name,
            "indirect_strict_success": vals.get("indirect_end_to_end_success_strict", 0.0),
            "wrong_contract_false_trigger_rate": vals.get("wrong_contract_false_trigger_rate", 0.0),
            "conflict_detection_accuracy": vals.get("conflict_detection_accuracy", 0.0),
            "conflict_safe_action_accuracy": vals.get("conflict_safe_action_accuracy", 0.0),
            "target_action_completion_rate": vals.get("target_action_completion_rate", 0.0),
            "strict_e2e": vals.get("end_to_end_success_strict", 0.0),
            "behavioral_e2e": vals.get("end_to_end_success_behavioral", 0.0),
            "fixed_vs_PACTFull_current": fixed_cur,
            "regressed_vs_PACTFull_current": reg_cur,
            "fixed_vs_PACT_R2_full": fixed_r2,
            "regressed_vs_PACT_R2_full": reg_r2,
        })
    write_csv(OUTPUT_DIR / "composition_ablation.csv", comp_rows)

    oracles = oracle_predictions(contracts, episodes, grouped)
    oracle_metrics = {name: score_method(episodes, preds) for name, preds in oracles.items()}
    oracle_rows = []
    current_metrics = metrics.get("PACTFull_current", metrics.get("PACTFull", {}))
    r2_metrics = metrics.get("PACT_R2_full", {})
    for name, vals in oracle_metrics.items():
        area, bottleneck = _largest_gain_area(current_metrics, vals)
        oracle_rows.append({
            "method": name,
            "oracle_unfair": True,
            "strict_e2e": vals.get("end_to_end_success_strict", 0.0),
            "behavioral_e2e": vals.get("end_to_end_success_behavioral", 0.0),
            "indirect_success": vals.get("indirect_end_to_end_success_strict", 0.0),
            "conflict_detection": vals.get("conflict_detection_accuracy", 0.0),
            "conflict_safe_action": vals.get("conflict_safe_action_accuracy", 0.0),
            "target_completion": vals.get("target_action_completion_rate", 0.0),
            "wrong_contract_false_trigger": vals.get("wrong_contract_false_trigger_rate", 0.0),
            "improvement_over_PACTFull_current": vals.get("end_to_end_success_strict", 0.0) - current_metrics.get("end_to_end_success_strict", 0.0),
            "improvement_over_PACT_R2_full": vals.get("end_to_end_success_strict", 0.0) - r2_metrics.get("end_to_end_success_strict", 0.0),
            "largest_gain_area": area,
            "diagnosed_bottleneck": bottleneck,
        })
    write_csv(OUTPUT_DIR / "oracle_ceiling.csv", oracle_rows)

    learned = ["LearnedPAM", "LearnedPAM_plus_checker", "LearnedPAM_plus_family_compiler"]
    write_csv(OUTPUT_DIR / "learned_pam_results.csv", [_d3_metrics_row(name, metrics[name], episodes, grouped[name], current, sanity) for name in learned if name in grouped])
    feature_report = {
        "backend": "deterministic_fallback",
        "training_split": "dev_only",
        "test_labels_seen_at_prediction_time": False,
        "features": ["query_history_ngrams", "contract_text_overlap", "intent_family_match", "family_indicator", "conflict_phrase_indicators", "already_satisfied_indicators", "wrong_scope_indicators"],
        "forbidden_prediction_fields": ["case_type", "gold_state", "expected_action_keywords", "forbidden_action_keywords", "completion_rubric", "notes"],
        "note": "Fallback mini-probe uses frozen deterministic feature thresholds rather than an external API or non-deterministic model.",
    }
    (OUTPUT_DIR / "learned_pam_feature_report.json").write_text(json.dumps(feature_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    learned_errors = []
    for name in learned:
        for ep in episodes:
            pred = _pred_by(grouped, name).get(ep.episode_id)
            if pred and not episode_success(ep, pred):
                learned_errors.append({"method": name, "episode_id": ep.episode_id, "family": ep.family, "case_type": ep.case_type, "failure_reason": failure_reason(ep, pred), **pred.to_dict()})
    write_csv(OUTPUT_DIR / "learned_pam_errors.csv", learned_errors)

    control_methods = ["QueryOnlyClassifier", "ContractOnlyClassifier", "QueryPlusFamilyClassifier", "QueryPlusContractClassifier", "QueryPlusWrongContractOnly"]
    control_rows = []
    for name in control_methods:
        if name not in metrics:
            continue
        row = _d3_metrics_row(name, metrics[name], episodes, grouped[name], current, sanity)
        row["validity_flag"] = ""
        if name == "QueryPlusFamilyClassifier" and row["strict_e2e"] >= metrics.get("PACT_R2_full", {}).get("end_to_end_success_strict", 1.0) - 0.05:
            row["validity_flag"] = "family_control_nearly_matches_full_contract"
        control_rows.append(row)
    write_csv(OUTPUT_DIR / "query_contract_controls.csv", control_rows)

    stress_rows = []
    stress_method = get_method("PACT_R2_full")
    contract_ids = [c.contract_id for c in contracts]
    for size in [1, 3, 6, 12]:
        preds = []
        confusion = Counter()
        for ep in episodes:
            desired = ep.gold_contract_id if ep.gold_state in {"fire", "conflict", "already_satisfied"} else ""
            distractors = [cid for cid in contract_ids if cid != desired]
            offset = sum(ord(ch) for ch in ep.episode_id) % len(distractors)
            rotated = distractors[offset:] + distractors[:offset]
            pool = ([desired] if desired else []) + rotated
            pool = [cid for cid in pool if cid][:size]
            if not pool:
                pool = rotated[:size]
            if ep.case_type == "contract_swap":
                pool = (ep.available_contract_ids + rotated)[:size]
            inf = InferenceEpisode(ep.episode_id, ep.history_summary, ep.current_query, pool)
            pred = stress_method.predict(contracts, inf)
            preds.append(pred)
            if pred.predicted_contract_id and pred.predicted_contract_id != "none":
                confusion[f"{ep.family}->{pred.predicted_contract_id}"] += 1
        vals = score_method(episodes, preds)
        correct_selection = sum(1 for ep, pred in zip(episodes, preds) if ep.gold_state in {"fire", "conflict", "already_satisfied"} and correct_contract(ep, pred)) / max(1, sum(1 for ep in episodes if ep.gold_state in {"fire", "conflict", "already_satisfied"}))
        stress_rows.append({
            "pool_size": size,
            "method": "PACT_R2_full",
            "correct_contract_selection_rate": correct_selection,
            "wrong_contract_false_trigger_rate": vals.get("wrong_contract_false_trigger_rate", 0.0),
            "indirect_success": vals.get("indirect_end_to_end_success_strict", 0.0),
            "strict_e2e": vals.get("end_to_end_success_strict", 0.0),
            "selection_confusion_by_family": json.dumps(dict(confusion.most_common(12)), sort_keys=True),
        })
    write_csv(OUTPUT_DIR / "multi_contract_stress.csv", stress_rows)

    anatomy = []
    for name in ["PACTFull_current", "PACT_R2_full", "PACT_intent_plus_state"]:
        for ep in episodes:
            pred = _pred_by(grouped, name).get(ep.episode_id)
            if pred:
                anatomy.append(_score_anatomy_row(ep, pred, contracts))
    write_csv(OUTPUT_DIR / "score_anatomy.csv", anatomy)
    write_csv(OUTPUT_DIR / "score_anatomy_by_error_type.csv", _average_rows(anatomy, "error_bucket") if anatomy else [])
    write_csv(OUTPUT_DIR / "score_anatomy_by_family.csv", _average_rows(anatomy, "family") if anatomy else [])
    write_csv(OUTPUT_DIR / "score_anatomy_false_fire_cases.csv", [row for row in anatomy if str(row["error_bucket"]).startswith("false_fire")])

    conflict_rows = []
    for subtype in sorted({conflict_subtype(ep) for ep in episodes if ep.gold_state == "conflict"}):
        eps = [ep for ep in episodes if ep.gold_state == "conflict" and conflict_subtype(ep) == subtype]
        if not eps:
            continue
        for name in ["PACTFull_current", "PACT_R2_full", "PACT_intent_plus_state", "LearnedPAM"]:
            preds_by = _pred_by(grouped, name)
            preds = [preds_by[ep.episode_id] for ep in eps if ep.episode_id in preds_by]
            if not preds:
                continue
            vals = score_method(eps, preds)
            conflict_rows.append({
                "method": name,
                "conflict_subtype": subtype,
                "n": len(eps),
                "conflict_detection_accuracy_by_subtype": vals.get("conflict_detection_accuracy", 0.0),
                "conflict_safe_action_accuracy_by_subtype": vals.get("conflict_safe_action_accuracy", 0.0),
                "conflict_as_fire_rate_by_subtype": vals.get("conflict_as_fire_rate", 0.0),
                "conflict_as_suppress_rate_by_subtype": vals.get("conflict_as_suppress_rate", 0.0),
            })
    write_csv(OUTPUT_DIR / "conflict_taxonomy.csv", conflict_rows)

    mismatch_rows = []
    for name in ["PACTFull_current", "PACT_R2_full", "PACT_intent_plus_state", "LearnedPAM"]:
        for ep in episodes:
            pred = _pred_by(grouped, name).get(ep.episode_id)
            if pred and pred.predicted_state == "fire" and pred.predicted_contract_id == ep.gold_contract_id and pred.action_completed and not target_action_completed(ep, pred):
                mismatch_rows.append({
                    "method": name,
                    "episode_id": ep.episode_id,
                    "family": ep.family,
                    "case_type": ep.case_type,
                    "query": ep.current_query,
                    "response": pred.response,
                    "expected_action_keywords": "|".join(ep.expected_action_keywords),
                    "completion_rubric": ep.completion_rubric,
                    "checker_result": pred.action_completed,
                    "target_completion_result": target_action_completed(ep, pred),
                    "failure_reason": failure_reason(ep, pred),
                })
    write_csv(OUTPUT_DIR / "target_completion_mismatches.csv", mismatch_rows)

    nat_eps = [ep for ep in episodes if ep.set_type == "naturalistic"]
    nat_rows = []
    nat_failures = []
    for name, preds in grouped.items():
        by = {p.episode_id: p for p in preds}
        subset = [by[ep.episode_id] for ep in nat_eps if ep.episode_id in by]
        if not subset:
            continue
        vals = score_method(nat_eps, subset)
        nat_rows.append({
            "method": name,
            "strict_success": vals.get("end_to_end_success_strict", 0.0),
            "behavioral_success": vals.get("end_to_end_success_behavioral", 0.0),
            "wrong_contract_false_trigger_rate": vals.get("wrong_contract_false_trigger_rate", 0.0),
            "conflict_detection": vals.get("conflict_detection_accuracy", 0.0),
            "target_completion": vals.get("target_action_completion_rate", 0.0),
            "indirect_success": vals.get("indirect_end_to_end_success_strict", 0.0),
            "false_trigger_rate_including_contract_swap": vals.get("false_trigger_rate_including_contract_swap", 0.0),
        })
        for ep in nat_eps:
            pred = by.get(ep.episode_id)
            if pred and not episode_success(ep, pred):
                nat_failures.append({"method": name, "episode_id": ep.episode_id, "family": ep.family, "case_type": ep.case_type, "failure_reason": failure_reason(ep, pred), **pred.to_dict()})
    write_csv(OUTPUT_DIR / "naturalistic_metrics.csv", nat_rows)
    write_csv(OUTPUT_DIR / "naturalistic_failures.csv", nat_failures)

    candidate_methods = [name for name in ["PACT_intent_plus_state", "PACT_intent_plus_state_family_compiler", "LearnedPAM", "LearnedPAM_plus_family_compiler", "QueryPlusContractClassifier"] if name in metrics]
    best_method = max(candidate_methods, key=lambda name: metrics[name]["end_to_end_success_strict"], default="PACT_R2_full")
    best_by = _pred_by(grouped, best_method)
    transition_rows = []
    for ep in episodes:
        cp = current_by.get(ep.episode_id)
        rp = r2_by.get(ep.episode_id)
        dp = best_by.get(ep.episode_id)
        if not cp or not dp:
            continue
        cur_ok = episode_success(ep, cp)
        d3_ok = episode_success(ep, dp)
        transition = "fixed" if (not cur_ok and d3_ok) else "regressed" if (cur_ok and not d3_ok) else "unchanged_correct" if cur_ok and d3_ok else "unchanged_wrong"
        if cp.predicted_state != dp.predicted_state or cp.predicted_contract_id != dp.predicted_contract_id or (rp and (rp.predicted_state != dp.predicted_state or rp.predicted_contract_id != dp.predicted_contract_id)) or transition in {"fixed", "regressed"}:
            transition_rows.append({
                "episode_id": ep.episode_id,
                "family": ep.family,
                "case_type": ep.case_type,
                "set_type": ep.set_type,
                "gold_state": ep.gold_state,
                "current_prediction": f"{cp.predicted_state}:{cp.predicted_contract_id}",
                "r2_prediction": f"{rp.predicted_state}:{rp.predicted_contract_id}" if rp else "",
                "d3_method": best_method,
                "d3_prediction": f"{dp.predicted_state}:{dp.predicted_contract_id}",
                "current_success_strict": cur_ok,
                "r2_success_strict": episode_success(ep, rp) if rp else "",
                "d3_success_strict": d3_ok,
                "transition": transition,
                "failure_reason_current": failure_reason(ep, cp),
                "failure_reason_r2": failure_reason(ep, rp) if rp else "",
                "failure_reason_d3": failure_reason(ep, dp),
            })
    write_csv(OUTPUT_DIR / "d3_error_transition.csv", transition_rows)
    write_csv(OUTPUT_DIR / "d3_fixed_errors.csv", [row for row in transition_rows if row["transition"] == "fixed"])
    write_csv(OUTPUT_DIR / "d3_new_errors.csv", [row for row in transition_rows if row["transition"] == "regressed"])

    diag_rows = []
    for name in ["PACTFull_current", "PACT_R2_full", "PACT_intent_plus_state", "PACT_intent_plus_state_family_compiler", "LearnedPAM", "LearnedPAM_plus_checker", "LearnedPAM_plus_family_compiler", "QueryOnlyClassifier", "QueryPlusFamilyClassifier", "QueryPlusContractClassifier"]:
        if name in metrics:
            diag_rows.append(_d3_metrics_row(name, metrics[name], episodes, grouped[name], current, sanity))
    for row in oracle_rows:
        diag_rows.append({
            "method": row["method"],
            "indirect_strict_success": row["indirect_success"],
            "wrong_contract_false_trigger_rate": row["wrong_contract_false_trigger"],
            "conflict_detection_accuracy": row["conflict_detection"],
            "conflict_safe_action_accuracy": row["conflict_safe_action"],
            "target_completion": row["target_completion"],
            "strict_e2e": row["strict_e2e"],
            "behavioral_e2e": row["behavioral_e2e"],
            "naturalistic_strict_success": "",
            "contract_shuffle_drop": "",
            "fixed_vs_current": "",
            "regressed_vs_current": "",
            "oracle_unfair": True,
        })
    write_csv(OUTPUT_DIR / "diagnostic_summary.csv", diag_rows)

    write_manual_d3_template(episodes, grouped, transition_rows, mismatch_rows, best_method)
    write_audit_d3(comp_rows, oracle_rows, metrics, sanity, best_method, transition_rows, mismatch_rows, stress_rows, nat_rows, control_rows)


def write_manual_d3_template(episodes: list[Episode], grouped: dict[str, list[Prediction]], transitions: list[dict[str, object]], mismatches: list[dict[str, object]], best_method: str) -> None:
    cur_by = _pred_by(grouped, "PACTFull_current") or _pred_by(grouped, "PACTFull")
    r2_by = _pred_by(grouped, "PACT_R2_full")
    d3_by = _pred_by(grouped, best_method)
    baseline_name = strongest_baseline_name(grouped, episodes, ordinary=True)
    base_by = _pred_by(grouped, baseline_name)
    by_id = {ep.episode_id: ep for ep in episodes}
    chosen: list[tuple[str, Episode]] = []
    for row in transitions:
        ep = by_id[row["episode_id"]]
        if row["transition"] == "fixed":
            chosen.append(("d3_fix_over_current", ep))
        elif row["transition"] == "regressed":
            chosen.append(("d3_regression_from_current", ep))
    for ep in episodes:
        pred = d3_by.get(ep.episode_id)
        if pred and ep.case_type == "contract_swap" and not episode_success(ep, pred):
            chosen.append(("remaining_wrong_contract_failure", ep))
        if pred and ep.gold_state == "conflict" and not episode_success(ep, pred):
            chosen.append(("remaining_conflict_failure", ep))
    for row in mismatches:
        chosen.append(("target_completion_mismatch", by_id[row["episode_id"]]))
    for ep in episodes:
        bp = base_by.get(ep.episode_id)
        dp = d3_by.get(ep.episode_id)
        if bp and dp and not episode_success(ep, bp) and episode_success(ep, dp):
            chosen.append(("baseline_failure_d3_success", ep))
            if sum(1 for group, _ in chosen if group == "baseline_failure_d3_success") >= 30:
                break
    for ep in episodes:
        dp = d3_by.get(ep.episode_id)
        if dp and episode_success(ep, dp):
            chosen.append(("d3_success_sample", ep))
            if sum(1 for group, _ in chosen if group == "d3_success_sample") >= 30:
                break
    rows = []
    seen = set()
    for group, ep in chosen:
        key = (group, ep.episode_id)
        if key in seen:
            continue
        seen.add(key)
        cp = cur_by.get(ep.episode_id)
        rp = r2_by.get(ep.episode_id)
        dp = d3_by.get(ep.episode_id)
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
            "pact_current_pred_state": cp.predicted_state if cp else "",
            "pact_current_pred_contract_id": cp.predicted_contract_id if cp else "",
            "r2_pred_state": rp.predicted_state if rp else "",
            "r2_pred_contract_id": rp.predicted_contract_id if rp else "",
            "d3_pred_state": dp.predicted_state if dp else "",
            "d3_pred_contract_id": dp.predicted_contract_id if dp else "",
            "pact_current_response": cp.response if cp else "",
            "r2_response": rp.response if rp else "",
            "d3_response": dp.response if dp else "",
            "strongest_baseline_method": baseline_name,
            "strongest_baseline_predicted_state": bp.predicted_state if bp else "",
            "strongest_baseline_predicted_contract_id": bp.predicted_contract_id if bp else "",
            "strongest_baseline_response": bp.response if bp else "",
            "audit_question_gold_label_valid": "",
            "audit_question_d3_fixed_for_right_reason": "",
            "audit_question_d3_regression_over_suppression": "",
            "audit_question_wrong_contract_issue": "",
            "audit_question_conflict_detected_or_behavior_only": "",
            "audit_question_action_completion_meaningful": "",
            "audit_question_baseline_unfairly_disadvantaged": "",
            "manual_notes": "",
        })
    write_csv(OUTPUT_DIR / "manual_audit_d3_template.csv", rows)


def write_audit_d3(comp_rows: list[dict[str, object]], oracle_rows: list[dict[str, object]], metrics: dict[str, dict[str, float]], sanity: dict[str, float], best_method: str, transitions: list[dict[str, object]], mismatches: list[dict[str, object]], stress_rows: list[dict[str, object]], nat_rows: list[dict[str, object]], control_rows: list[dict[str, object]]) -> None:
    fixed = sum(1 for row in transitions if row["transition"] == "fixed")
    regressed = sum(1 for row in transitions if row["transition"] == "regressed")
    best_vals = metrics.get(best_method, {})
    deterministic_candidates = [row for row in comp_rows if row["method"] in {"PACT_intent_plus_state", "PACT_intent_plus_state_family_compiler"}]
    deterministic_ready = any(
        row["wrong_contract_false_trigger_rate"] <= 0.10
        and row["indirect_strict_success"] >= 0.84
        and row["conflict_detection_accuracy"] >= 0.58
        and row["conflict_safe_action_accuracy"] >= 0.85
        and row["target_action_completion_rate"] >= 0.85
        and row["strict_e2e"] >= 0.88
        and row["fixed_vs_PACTFull_current"] > row["regressed_vs_PACTFull_current"]
        for row in deterministic_candidates
    )
    learned_names = ["LearnedPAM", "LearnedPAM_plus_checker", "LearnedPAM_plus_family_compiler"]
    learned_best = max((name for name in learned_names if name in metrics), key=lambda name: metrics[name]["end_to_end_success_strict"], default="")
    det_best = max((row["method"] for row in deterministic_candidates), key=lambda name: metrics.get(name, {}).get("end_to_end_success_strict", 0.0), default="")
    learned_beats = bool(learned_best and det_best and metrics[learned_best]["wrong_contract_false_trigger_rate"] <= metrics[det_best]["wrong_contract_false_trigger_rate"] and metrics[learned_best]["indirect_end_to_end_success_strict"] >= metrics[det_best]["indirect_end_to_end_success_strict"] and metrics[learned_best]["conflict_detection_accuracy"] >= metrics[det_best]["conflict_detection_accuracy"] and metrics[learned_best]["end_to_end_success_strict"] > metrics[det_best]["end_to_end_success_strict"])
    largest_oracle = max(oracle_rows, key=lambda row: row["improvement_over_PACTFull_current"], default={})
    naturalistic_best = max((row for row in nat_rows if row["method"] == best_method), key=lambda row: row["strict_success"], default={})
    family_threat = any(row.get("validity_flag") for row in control_rows)
    if deterministic_ready:
        decision = "DETERMINISTIC_PACT_READY"
    elif learned_beats:
        decision = "USE_LEARNED_PAM"
    elif family_threat:
        decision = "NARROW_CLAIM"
    elif largest_oracle.get("largest_gain_area") == "contract_selection":
        decision = "FIX_CONTRACT_SELECTION"
    elif largest_oracle.get("largest_gain_area") == "compiler_checker":
        decision = "FIX_COMPILER_CHECKER"
    elif naturalistic_best and naturalistic_best.get("strict_success", 1.0) < best_vals.get("end_to_end_success_strict", 0.0) - 0.05:
        decision = "NARROW_CLAIM"
    else:
        decision = "KILL"
    lines = [
        "Simulated subagent: D3 Diagnostic Audit Agent.",
        "Dataset unchanged: D3 reuses frozen pact_causal_520 and does not edit labels, case types, splits, or rubrics.",
        "Dev-only tuning: R2 thresholds remain dev-selected and frozen; D3 diagnostics do not tune on test.",
        "Oracle rows are diagnostic/unfair ceilings and are not fair baselines.",
        "LearnedPAM status: deterministic offline fallback mini-probe; no external LLM/API calls.",
        f"Best diagnostic method by strict E2E: {best_method} ({best_vals.get('end_to_end_success_strict', 0.0):.3f}).",
        f"D3 fixed vs current: {fixed}; regressed vs current: {regressed}.",
        f"Largest oracle gain: {largest_oracle.get('method', '')} area={largest_oracle.get('largest_gain_area', '')} bottleneck={largest_oracle.get('diagnosed_bottleneck', '')}.",
        f"Target-completion mismatch count: {len(mismatches)}.",
        f"Multi-contract stress summary: {json.dumps(stress_rows, sort_keys=True)[:3000]}.",
        f"Naturalistic row for best method: {json.dumps(naturalistic_best, sort_keys=True)}.",
        f"Query/family control threat: {family_threat}.",
        f"D3 research decision: {decision}.",
        "Manual audit status: manual_audit_d3_template.csv is a template, not completed human evidence.",
        "Caveats: inspect remaining wrong-contract failures, conflict subtype concentrations, target-completion mismatches, and D3 regressions before making any positive claim.",
    ]
    (OUTPUT_DIR / "audit_d3.md").write_text("# D3 Diagnostic Audit Agent\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


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
    write_d3_reports(contracts, episodes, grouped, metrics, sanity)
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
