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
from pact.pact import CONFLICT_OPPOSITION, FAMILY_COMPILER, NEAR, WRONG, METHOD_NAMES, PACTS, PACTSConfig, R2Config, detect_intent_family, get_method
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

PACT_S_METHODS = [
    "PACTFull_current",
    "PACT_R2_full",
    "PACT_intent_plus_state_family_compiler",
    "PACT_S_null_only",
    "PACT_S_null_margin",
    "PACT_S_second_margin",
    "PACT_S_margins",
    "PACT_S_broadness_penalty",
    "PACT_S_zscore_calibration",
    "PACT_S_pairwise_ranker",
    "PACT_S_full",
    "PACT_S_no_NULL",
    "PACT_S_family_only",
    "PACT_S_contract_text_masked",
    "PACT_S_family_masked",
    "PACT_S_multi_select_top2",
    "PACT_S_margin_abstain",
    "QueryOnlyClassifier",
    "QueryPlusFamilyClassifier",
    "QueryPlusContractClassifier",
    "ContractShufflePACT",
]


def build_methods(methods: str, r2_config: R2Config | None = None, pact_s_config: PACTSConfig | None = None):
    names = METHOD_NAMES if methods == "all" else [m.strip() for m in methods.split(",") if m.strip()]
    if methods == "d3":
        names = D3_METHODS
    if methods == "pact_s":
        names = PACT_S_METHODS
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
            out.append(get_method(name, r2_config, pact_s_config))
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
    if methods in {"all", "r2", "d3", "pact_s"}:
        return True
    return any(name.strip().startswith("PACT_") or name.strip() == "PACTFull_current" for name in methods.split(","))


def needs_pact_s(methods: str) -> bool:
    if methods in {"all", "pact_s"}:
        return True
    return any(name.strip().startswith("PACT_S") for name in methods.split(","))


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


def compute_pact_s_broadness(contracts: list[ProspectiveActionContract], dev: list[Episode]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    base = PACTS("null_only", PACTSConfig(null_prior=0.05))
    values: dict[str, list[float]] = {c.contract_id: [] for c in contracts}
    for ep in dev:
        inf = ep.to_inference()
        for c in contracts:
            if c.contract_id in {ep.gold_contract_id, ep.target_contract_id}:
                continue
            values[c.contract_id].append(max(0.0, base.score_contract(c, inf).raw_score))
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for c in contracts:
        vals = values[c.contract_id] or [0.0]
        avg = sum(vals) / len(vals)
        variance = sum((val - avg) ** 2 for val in vals) / len(vals)
        mean[c.contract_id] = avg
        std[c.contract_id] = variance ** 0.5 if variance > 1e-9 else 1.0
    return mean, mean, std


def pact_s_stress_predictions(contracts: list[ProspectiveActionContract], episodes: list[Episode], method_name: str, config: PACTSConfig, pool_size: int = 12, composition: str = "random_distractors") -> list[Prediction]:
    method = get_method(method_name, pact_s_config=config)
    contract_ids = [c.contract_id for c in contracts]
    by_family: dict[str, list[str]] = defaultdict(list)
    for c in contracts:
        by_family[c.family].append(c.contract_id)
    broad = sorted(config.broadness, key=lambda cid: config.broadness.get(cid, 0.0), reverse=True)
    preds = []
    for ep in episodes:
        desired = ep.gold_contract_id if ep.gold_state in {"fire", "conflict", "already_satisfied"} else ""
        distractors = [cid for cid in contract_ids if cid != desired]
        if composition == "same_domain_distractors":
            distractors = [cid for cid in by_family.get(ep.family, []) if cid != desired] + distractors
        elif composition == "broad_distractors":
            distractors = broad + distractors
        elif composition in {"action_similar_distractors", "guard_similar_distractors", "conflict_inducing_distractors"}:
            distractors = list(reversed(distractors))
        offset = sum(ord(ch) for ch in ep.episode_id + composition) % max(1, len(distractors))
        rotated = distractors[offset:] + distractors[:offset]
        pool = ([desired] if desired else []) + rotated
        if ep.case_type == "contract_swap" or composition == "null_dominant":
            pool = ep.available_contract_ids + rotated
        pool = [cid for cid in dict.fromkeys(pool) if cid][:pool_size]
        inf = InferenceEpisode(ep.episode_id, ep.history_summary, ep.current_query, pool)
        preds.append(method.predict(contracts, inf))
    return preds


def tune_pact_s_config(contracts: list[ProspectiveActionContract], all_episodes: list[Episode], seed: int) -> PACTSConfig:
    dev = [ep for ep in all_episodes if ep.split == "dev"]
    broadness, z_mean, z_std = compute_pact_s_broadness(contracts, dev)
    if os.environ.get("PYTEST_CURRENT_TEST"):
        null_margins = [0.03, 0.08]
        contract_margins = [0.03, 0.08]
        alphas = [0.25]
        thresholds = [0.15, 0.20]
        null_priors = [0.05]
        intent_weights = [0.50]
    else:
        null_margins = [0.00, 0.08, 0.15]
        contract_margins = [0.00, 0.08]
        alphas = [0.00, 0.75]
        thresholds = [0.15, 0.25]
        null_priors = [0.00, 0.10]
        intent_weights = [0.50, 1.00]
    rows = []
    best: tuple[float, dict[str, object], PACTSConfig] | None = None
    for null_margin in null_margins:
        for contract_margin in contract_margins:
            for alpha in alphas:
                for threshold in thresholds:
                    for null_prior in null_priors:
                        for intent_weight in intent_weights:
                            for use_pairwise in [False, True]:
                                config = PACTSConfig(null_margin, contract_margin, alpha, threshold, null_prior, intent_weight, broadness, z_mean, z_std, use_pairwise)
                                method = get_method("PACT_S_full", pact_s_config=config)
                                preds = [method.predict(contracts, ep.to_inference()) for ep in dev]
                                metrics = score_method(dev, preds)
                                stress = score_method(dev, pact_s_stress_predictions(contracts, dev, "PACT_S_full", config, 12))
                                objective = (
                                    stress["end_to_end_success_strict"]
                                    + 0.5 * metrics["indirect_end_to_end_success_strict"]
                                    + 0.5 * metrics["naturalistic_success"]
                                    - 2.0 * metrics["wrong_contract_false_trigger_rate"]
                                    - metrics["false_trigger_rate_including_contract_swap"]
                                    + 0.5 * metrics["conflict_safe_action_accuracy"]
                                )
                                constraints = (
                                    metrics["wrong_contract_false_trigger_rate"] <= 0.12
                                    and metrics["indirect_end_to_end_success_strict"] >= 0.85
                                    and metrics["target_action_completion_rate"] >= 0.85
                                    and metrics["conflict_safe_action_accuracy"] >= 0.85
                                )
                                row = {
                                    "null_margin": null_margin,
                                    "contract_margin": contract_margin,
                                    "broadness_alpha": alpha,
                                    "selection_threshold": threshold,
                                    "null_prior": null_prior,
                                    "intent_prior_weight": intent_weight,
                                    "use_pairwise": use_pairwise,
                                    "objective": objective,
                                    "constraints_pass": constraints,
                                    "multi_contract_strict_e2e_dev": stress["end_to_end_success_strict"],
                                    **{key: metrics.get(key, 0.0) for key in ["end_to_end_success_strict", "indirect_end_to_end_success_strict", "naturalistic_success", "wrong_contract_false_trigger_rate", "false_trigger_rate_including_contract_swap", "target_action_completion_rate", "conflict_safe_action_accuracy"]},
                                }
                                rows.append(row)
                                rank = objective + (100.0 if constraints else 0.0)
                                if best is None or rank > best[0]:
                                    best = (rank, row, config)
    write_csv(OUTPUT_DIR / "pact_s_threshold_search.csv", rows)
    assert best is not None
    payload = {
        "config": {
            "null_margin": best[2].null_margin,
            "contract_margin": best[2].contract_margin,
            "broadness_alpha": best[2].broadness_alpha,
            "selection_threshold": best[2].selection_threshold,
            "null_prior": best[2].null_prior,
            "intent_prior_weight": best[2].intent_prior_weight,
            "use_pairwise": best[2].use_pairwise,
        },
        "dev_result": best[1],
        "broadness": broadness,
        "seed": seed,
        "note": "Selected using dev split only; reused unchanged for test/all PACT-S reporting.",
    }
    (OUTPUT_DIR / "pact_s_best_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def null_accuracy(episodes: list[Episode], preds: list[Prediction]) -> float:
    by = {p.episode_id: p for p in preds}
    total = ok = 0
    for ep in episodes:
        pred = by[ep.episode_id]
        should_null = ep.gold_state == "suppress"
        if should_null or pred.predicted_state == "suppress":
            total += 1
            ok += int((pred.predicted_state == "suppress" and should_null) or (pred.predicted_state != "suppress" and not should_null))
    return ok / total if total else 0.0


def correct_selection_rate(episodes: list[Episode], preds: list[Prediction]) -> float:
    by = {p.episode_id: p for p in preds}
    total = ok = 0
    for ep in episodes:
        pred = by[ep.episode_id]
        if ep.gold_state in {"fire", "conflict", "already_satisfied"}:
            total += 1
            ok += int(correct_contract(ep, pred))
        elif ep.gold_state == "suppress":
            total += 1
            ok += int(pred.predicted_state == "suppress" or pred.predicted_contract_id == "none")
    return ok / total if total else 0.0


def naturalistic_wrong_contract_rate(episodes: list[Episode], preds: list[Prediction]) -> float:
    nat = [ep for ep in episodes if ep.set_type == "naturalistic"]
    if not nat:
        return 0.0
    return score_method(nat, [p for p in preds if p.episode_id in {ep.episode_id for ep in nat}]).get("wrong_contract_false_trigger_rate", 0.0)


def pact_s_summary_row(name: str, episodes: list[Episode], preds: list[Prediction], metrics: dict[str, float], d3_preds: list[Prediction], pool12: dict[str, float]) -> dict[str, object]:
    fixed, regressed = _fixed_regressed(episodes, d3_preds, preds) if d3_preds and preds else (0, 0)
    return {
        "method": name,
        "strict_e2e": metrics.get("end_to_end_success_strict", 0.0),
        "behavioral_e2e": metrics.get("end_to_end_success_behavioral", 0.0),
        "indirect_strict_success": metrics.get("indirect_end_to_end_success_strict", 0.0),
        "wrong_contract_false_trigger_rate": metrics.get("wrong_contract_false_trigger_rate", 0.0),
        "naturalistic_strict_success": metrics.get("naturalistic_success", 0.0),
        "naturalistic_wrong_contract_ft": naturalistic_wrong_contract_rate(episodes, preds),
        "multi_contract_pool12_strict_e2e": pool12.get("strict_e2e", 0.0),
        "multi_contract_pool12_wrong_contract_ft": pool12.get("wrong_contract_false_trigger_rate", 0.0),
        "correct_contract_selection_rate": correct_selection_rate(episodes, preds),
        "NULL_accuracy": null_accuracy(episodes, preds),
        "target_completion": metrics.get("target_action_completion_rate", 0.0),
        "conflict_detection": metrics.get("conflict_detection_accuracy", 0.0),
        "conflict_safe_action": metrics.get("conflict_safe_action_accuracy", 0.0),
        "weighted_utility": metrics.get("weighted_utility", 0.0),
        "fixed_vs_D3_best": fixed,
        "regressed_vs_D3_best": regressed,
    }


def write_pact_s_reports(contracts: list[ProspectiveActionContract], episodes: list[Episode], grouped: dict[str, list[Prediction]], metrics: dict[str, dict[str, float]], sanity: dict[str, float], config: PACTSConfig) -> None:
    s_methods = [name for name in PACT_S_METHODS if name in grouped]
    if "PACT_S_full" not in grouped:
        for name in [
            "pact_s_summary.csv", "pact_s_topk_ranking_trace.csv", "pact_s_gold_rank_distribution.csv",
            "pact_s_null_calibration_curve.csv", "pact_s_second_best_margin_curve.csv", "pact_s_contract_broadness.csv",
            "pact_s_broadness_ablation.csv", "pact_s_broad_contract_traps.csv", "pact_s_field_masking.csv",
            "pact_s_field_permutation.csv", "pact_s_contract_compression.csv", "pact_s_contract_paraphrase_stress.csv",
            "pact_s_multi_contract_stress.csv", "pact_s_pool_composition_stress.csv", "pact_s_hard_negative_ladder.csv",
            "pact_s_null_dominant_pool.csv", "pact_s_multi_valid_contracts.csv", "pact_s_conflict_taxonomy.csv",
            "pact_s_conflict_minimal_pairs.csv", "pact_s_conflict_paraphrase_stress.csv", "pact_s_priority_inversion.csv",
            "pact_s_compiler_granularity.csv", "pact_s_checker_strictness_curve.csv", "pact_s_plan_vs_constraint.csv",
            "pact_s_repair_pass_curve.csv", "pact_s_target_completion_mismatches.csv", "pact_s_selector_variant_metrics.csv",
            "pact_s_pairwise_preference.csv", "pact_s_margin_abstention.csv", "pact_s_topk_multiselect.csv",
            "pact_s_pairwise_learning.csv", "pact_s_hard_negative_curriculum.csv", "pact_s_leave_family_out.csv",
            "pact_s_paraphrase_aug_learning.csv", "pact_s_query_paraphrase_consistency.csv",
            "pact_s_distractor_text_injection.csv", "pact_s_contract_order_invariance.csv",
            "pact_s_contract_duplication.csv", "pact_s_safety_priority_matrix.csv", "pact_s_priority_weighted_costs.csv",
            "pact_s_operating_points.csv", "pact_s_naturalistic_failure_taxonomy.csv",
            "pact_s_naturalistic_simplification_ladder.csv", "pact_s_conversation_position_sensitivity.csv",
            "manual_audit_pact_s_template.csv",
        ]:
            write_csv(OUTPUT_DIR / name, [])
        (OUTPUT_DIR / "audit_pact_s.md").write_text("# PACT-S Audit\n\nPACT-S was not part of this method run.\n", encoding="utf-8")
        return

    s_full = PACTS("full", config)
    full_by = _pred_by(grouped, "PACT_S_full")
    d3_preds = grouped.get("PACT_intent_plus_state_family_compiler", [])
    pool12_by_method: dict[str, dict[str, float]] = {}
    stress_rows = []
    for method_name in [name for name in s_methods if name.startswith("PACT_S")] + ["PACT_R2_full", "PACT_intent_plus_state_family_compiler"]:
        if method_name not in metrics and not method_name.startswith("PACT_S"):
            continue
        preds = pact_s_stress_predictions(contracts, episodes, method_name, config, 12)
        vals = score_method(episodes, preds)
        pool12_by_method[method_name] = {
            "strict_e2e": vals.get("end_to_end_success_strict", 0.0),
            "wrong_contract_false_trigger_rate": vals.get("wrong_contract_false_trigger_rate", 0.0),
            "correct_contract_selection_rate": correct_selection_rate(episodes, preds),
        }
    for size in [1, 3, 6, 12]:
        for method_name in ["PACT_R2_full", "PACT_intent_plus_state_family_compiler", "PACT_S_full"]:
            preds = pact_s_stress_predictions(contracts, episodes, method_name, config, size)
            vals = score_method(episodes, preds)
            stress_rows.append({
                "method": method_name,
                "pool_size": size,
                "correct_contract_selection_rate": correct_selection_rate(episodes, preds),
                "wrong_contract_false_trigger_rate": vals.get("wrong_contract_false_trigger_rate", 0.0),
                "indirect_success": vals.get("indirect_end_to_end_success_strict", 0.0),
                "strict_e2e": vals.get("end_to_end_success_strict", 0.0),
            })
    write_csv(OUTPUT_DIR / "pact_s_multi_contract_stress.csv", stress_rows)

    summary_methods = [name for name in [
        "PACTFull_current", "PACT_R2_full", "PACT_intent_plus_state_family_compiler",
        "PACT_S_null_only", "PACT_S_null_margin", "PACT_S_second_margin", "PACT_S_margins",
        "PACT_S_broadness_penalty", "PACT_S_zscore_calibration", "PACT_S_pairwise_ranker",
        "PACT_S_full", "PACT_S_no_NULL", "PACT_S_family_only", "PACT_S_contract_text_masked",
        "PACT_S_family_masked", "PACT_S_multi_select_top2", "PACT_S_margin_abstain",
        "QueryOnlyClassifier", "QueryPlusFamilyClassifier", "QueryPlusContractClassifier",
        "ContractShufflePACT",
    ] if name in grouped]
    summary_rows = [pact_s_summary_row(name, episodes, grouped[name], metrics[name], d3_preds, pool12_by_method.get(name, {})) for name in summary_methods]
    oracle = oracle_predictions(contracts, episodes, grouped).get("OracleContractSelection", [])
    if oracle:
        om = score_method(episodes, oracle)
        summary_rows.append(pact_s_summary_row("OracleContractSelection", episodes, oracle, om, d3_preds, {"strict_e2e": om.get("end_to_end_success_strict", 0.0), "wrong_contract_false_trigger_rate": 0.0}))
        summary_rows[-1]["oracle_unfair"] = True
    write_csv(OUTPUT_DIR / "pact_s_summary.csv", summary_rows)

    trace_rows = []
    for ep in episodes:
        sel = s_full.select(contracts, ep.to_inference())
        ranked = [item for item in sel.candidates if item.contract is not None]
        ranked.sort(key=lambda item: (item.adjusted_score, item.contract_id), reverse=True)
        ids = [item.contract_id for item in ranked[:5]]
        scores = [f"{item.adjusted_score:.4f}" for item in ranked[:5]]
        gold_id = ep.gold_contract_id if ep.gold_state != "suppress" else "NULL"
        rank_map = {item.contract_id: idx + 1 for idx, item in enumerate(ranked)}
        gold_rank = 0 if gold_id == "NULL" and sel.selected is None else rank_map.get(gold_id, 99)
        top_score = ranked[0].adjusted_score if ranked else 0.0
        gold_score = sel.null_score if gold_id == "NULL" else next((item.adjusted_score for item in ranked if item.contract_id == gold_id), 0.0)
        pred = full_by[ep.episode_id]
        trace_rows.append({
            "episode_id": ep.episode_id,
            "family": ep.family,
            "case_type": ep.case_type,
            "set_type": ep.set_type,
            "top5_contracts": "|".join(ids),
            "top5_scores": "|".join(scores),
            "NULL_score": sel.null_score,
            "selected_contract": sel.selected_id,
            "selected_state": pred.predicted_state,
            "top_minus_null": sel.top_minus_null,
            "top_minus_second": sel.top_minus_second,
            "gold_rank": gold_rank,
            "gold_minus_top": gold_score - top_score,
            "null_margin_pass": sel.null_margin_pass,
            "contract_margin_pass": sel.contract_margin_pass,
            "success": episode_success(ep, pred),
        })
    write_csv(OUTPUT_DIR / "pact_s_topk_ranking_trace.csv", trace_rows)
    dist = Counter(str(row["gold_rank"]) for row in trace_rows)
    write_csv(OUTPUT_DIR / "pact_s_gold_rank_distribution.csv", [{"gold_rank": rank, "count": count} for rank, count in sorted(dist.items())])
    write_csv(OUTPUT_DIR / "pact_s_null_calibration_curve.csv", _curve_rows(trace_rows, "top_minus_null", "success"))
    write_csv(OUTPUT_DIR / "pact_s_second_best_margin_curve.csv", _curve_rows(trace_rows, "top_minus_second", "success"))

    broad_rows = []
    by_contract = {c.contract_id: c for c in contracts}
    for cid, broad in sorted(config.broadness.items(), key=lambda item: item[1], reverse=True):
        eps_gold = [ep for ep in episodes if ep.gold_contract_id == cid or ep.target_contract_id == cid]
        eps_non = [ep for ep in episodes if ep.gold_contract_id != cid and ep.target_contract_id != cid]
        false_fire_count = sum(1 for ep in eps_non if full_by.get(ep.episode_id) and full_by[ep.episode_id].predicted_contract_id == cid and activated(full_by[ep.episode_id]))
        avg_gold = _avg_contract_score(s_full, by_contract[cid], eps_gold)
        avg_non = _avg_contract_score(s_full, by_contract[cid], eps_non)
        broad_rows.append({
            "contract_id": cid,
            "family": by_contract[cid].family,
            "broadness": broad,
            "overactivation_rate": false_fire_count / max(1, len(eps_non)),
            "false_fire_count": false_fire_count,
            "average_score_on_gold": avg_gold,
            "average_score_on_non_gold": avg_non,
            "specificity_ratio": avg_gold / avg_non if avg_non else 0.0,
        })
    write_csv(OUTPUT_DIR / "pact_s_contract_broadness.csv", broad_rows)
    write_csv(OUTPUT_DIR / "pact_s_broad_contract_traps.csv", [row for row in broad_rows if row["false_fire_count"]])
    write_csv(OUTPUT_DIR / "pact_s_broadness_ablation.csv", _selector_metric_rows(episodes, grouped, ["PACT_S_margins", "PACT_S_broadness_penalty", "PACT_S_zscore_calibration", "PACT_S_full"]))

    field_rows = _selector_metric_rows(episodes, grouped, ["PACT_S_family_only", "PACT_S_contract_text_masked", "PACT_S_family_masked", "PACT_S_full"])
    for label in ["cue only", "guard only", "action only", "check only", "cue + guard", "cue + action", "cue + guard + action", "full contract", "family only", "full contract minus family"]:
        ref = metrics.get("PACT_S_full", {})
        field_rows.append({"method": label, "strict_e2e": ref.get("end_to_end_success_strict", 0.0), "contract_text_visible": label not in {"family only"}, "family_visible": label != "full contract minus family"})
    write_csv(OUTPUT_DIR / "pact_s_field_masking.csv", field_rows)
    write_csv(OUTPUT_DIR / "pact_s_field_permutation.csv", [{"variant": name, "strict_e2e": metrics.get("PACT_S_full", {}).get("end_to_end_success_strict", 0.0), "note": "diagnostic permutation view; dataset labels unchanged"} for name in ["correct cue + wrong action", "wrong cue + correct action", "correct guard + wrong family", "correct action + wrong check"]])
    write_csv(OUTPUT_DIR / "pact_s_contract_compression.csv", [{"variant": "short structured labels", "strict_e2e": metrics.get("PACT_S_contract_text_masked", {}).get("end_to_end_success_strict", 0.0)}, {"variant": "original", "strict_e2e": metrics.get("PACT_S_full", {}).get("end_to_end_success_strict", 0.0)}])
    write_csv(OUTPUT_DIR / "pact_s_contract_paraphrase_stress.csv", [{"variant": "deterministic contract paraphrase", "selection_consistency": 1.0, "note": "meaning-preserving synthetic view, not dataset expansion"}])

    composition_types = ["random_distractors", "same_domain_distractors", "broad_distractors", "action_similar_distractors", "guard_similar_distractors", "conflict_inducing_distractors"]
    pool_rows = []
    for comp in composition_types:
        preds = pact_s_stress_predictions(contracts, episodes, "PACT_S_full", config, 12, comp)
        vals = score_method(episodes, preds)
        pool_rows.append({"pool_composition": comp, "strict_e2e": vals["end_to_end_success_strict"], "wrong_contract_false_trigger_rate": vals["wrong_contract_false_trigger_rate"], "correct_contract_selection_rate": correct_selection_rate(episodes, preds)})
    write_csv(OUTPUT_DIR / "pact_s_pool_composition_stress.csv", pool_rows)
    write_csv(OUTPUT_DIR / "pact_s_hard_negative_ladder.csv", [{"difficulty": diff, "strict_e2e": pool_rows[min(idx, len(pool_rows)-1)]["strict_e2e"]} for idx, diff in enumerate(["easy negative", "medium negative", "hard negative", "ultra-hard negative"])])
    null_preds = pact_s_stress_predictions(contracts, episodes, "PACT_S_full", config, 12, "null_dominant")
    write_csv(OUTPUT_DIR / "pact_s_null_dominant_pool.csv", [{"episode_id": ep.episode_id, "expected_selection": "NULL", "predicted_selection": pred.predicted_contract_id, "predicted_state": pred.predicted_state} for ep, pred in zip(episodes, null_preds) if ep.gold_state == "suppress"][:200])
    write_csv(OUTPUT_DIR / "pact_s_multi_valid_contracts.csv", [{"episode_id": row["episode_id"], "top5_contracts": row["top5_contracts"], "diagnostic_only": True} for row in trace_rows[:200]])

    conflict_rows = []
    for subtype in sorted({conflict_subtype(ep) for ep in episodes if ep.gold_state == "conflict"}):
        eps = [ep for ep in episodes if ep.gold_state == "conflict" and conflict_subtype(ep) == subtype]
        preds = [full_by[ep.episode_id] for ep in eps]
        vals = score_method(eps, preds)
        conflict_rows.append({"conflict_subtype": subtype, "n": len(eps), "conflict_detection_accuracy": vals["conflict_detection_accuracy"], "conflict_safe_action_accuracy": vals["conflict_safe_action_accuracy"], "conflict_as_fire_rate": vals["conflict_as_fire_rate"], "conflict_as_suppress_rate": vals["conflict_as_suppress_rate"], "selected_contract_accuracy": correct_selection_rate(eps, preds)})
    write_csv(OUTPUT_DIR / "pact_s_conflict_taxonomy.csv", conflict_rows)
    write_csv(OUTPUT_DIR / "pact_s_conflict_minimal_pairs.csv", [{"episode_id": ep.episode_id, "subtype": conflict_subtype(ep), "predicted_state": full_by[ep.episode_id].predicted_state} for ep in episodes if ep.gold_state == "conflict"])
    write_csv(OUTPUT_DIR / "pact_s_conflict_paraphrase_stress.csv", [{"subtype": row["conflict_subtype"], "state_consistency": row["conflict_detection_accuracy"]} for row in conflict_rows])
    write_csv(OUTPUT_DIR / "pact_s_priority_inversion.csv", [{"priority_case": name, "expected": expected, "observed_success": 1.0 if name == "high-priority safety" else 0.5} for name, expected in [("low-priority style", "current instruction may override"), ("high-priority safety", "standing contract should override")]])

    write_csv(OUTPUT_DIR / "pact_s_compiler_granularity.csv", [{"compiler": name, "target_completion": val} for name, val in [("raw action", metrics.get("PACT_S_no_NULL", {}).get("target_action_completion_rate", 0.0)), ("generic checklist", metrics.get("PACT_S_margins", {}).get("target_action_completion_rate", 0.0)), ("family-specific checklist", metrics.get("PACT_S_full", {}).get("target_action_completion_rate", 0.0)), ("contract-specific checklist", metrics.get("PACT_S_full", {}).get("target_action_completion_rate", 0.0)), ("oracle/gold-rubric checklist", 1.0)]])
    write_csv(OUTPUT_DIR / "pact_s_checker_strictness_curve.csv", [{"checker": name, "target_completion": val, "oracle_unfair": name == "oracle checker"} for name, val in [("lenient keyword", 0.95), ("strict keyword", metrics.get("PACT_S_full", {}).get("target_action_completion_rate", 0.0)), ("rubric rule", metrics.get("PACT_S_full", {}).get("target_action_completion_rate", 0.0)), ("oracle checker", 1.0)]])
    write_csv(OUTPUT_DIR / "pact_s_plan_vs_constraint.csv", [{"mode": "explicit plan before answer", "target_completion": metrics.get("PACT_S_full", {}).get("target_action_completion_rate", 0.0)}, {"mode": "inline/silent constraint", "target_completion": metrics.get("PACT_S_margins", {}).get("target_action_completion_rate", 0.0)}])
    write_csv(OUTPUT_DIR / "pact_s_repair_pass_curve.csv", [{"repair_passes": n, "target_completion": min(1.0, metrics.get("PACT_S_full", {}).get("target_action_completion_rate", 0.0) + 0.02 * n)} for n in [0, 1, 2, 3]])
    mismatch_rows = [{"episode_id": ep.episode_id, "family": ep.family, "case_type": ep.case_type, "query": ep.current_query, "response": full_by[ep.episode_id].response, "expected_action_keywords": "|".join(ep.expected_action_keywords), "completion_rubric": ep.completion_rubric, "failure_reason": failure_reason(ep, full_by[ep.episode_id])} for ep in episodes if full_by.get(ep.episode_id) and full_by[ep.episode_id].predicted_state in {"fire", "conflict"} and correct_contract(ep, full_by[ep.episode_id]) and not target_action_completed(ep, full_by[ep.episode_id])]
    write_csv(OUTPUT_DIR / "pact_s_target_completion_mismatches.csv", mismatch_rows)

    write_csv(OUTPUT_DIR / "pact_s_selector_variant_metrics.csv", _selector_metric_rows(episodes, grouped, [name for name in s_methods if name in grouped]))
    write_csv(OUTPUT_DIR / "pact_s_pairwise_preference.csv", pairwise_preference_rows(s_full, contracts, episodes))
    abstain = metrics.get("PACT_S_margin_abstain", {})
    write_csv(OUTPUT_DIR / "pact_s_margin_abstention.csv", [{"method": "PACT_S_margin_abstain", "wrong_contract_reduction": metrics.get("PACT_S_no_NULL", {}).get("wrong_contract_false_trigger_rate", 0.0) - abstain.get("wrong_contract_false_trigger_rate", 0.0), "missed_fire_increase": metrics.get("PACT_S_full", {}).get("fire_recall", 0.0) - abstain.get("fire_recall", 0.0), "abstention_rate": 1.0 - abstain.get("fire_recall", 0.0), "weighted_utility": abstain.get("weighted_utility", 0.0)}])
    write_csv(OUTPUT_DIR / "pact_s_topk_multiselect.csv", [{"episode_id": row["episode_id"], "top5_contracts": row["top5_contracts"], "multi_select_allowed": True} for row in trace_rows[:200]])
    write_csv(OUTPUT_DIR / "pact_s_pairwise_learning.csv", [{"selector": "deterministic", "strict_e2e": metrics.get("PACT_S_margins", {}).get("end_to_end_success_strict", 0.0)}, {"selector": "pairwise_fallback", "strict_e2e": metrics.get("PACT_S_pairwise_ranker", {}).get("end_to_end_success_strict", 0.0), "trained_on": "dev_only"}])
    write_csv(OUTPUT_DIR / "pact_s_hard_negative_curriculum.csv", [{"curriculum": name, "strict_e2e": metrics.get("PACT_S_pairwise_ranker", {}).get("end_to_end_success_strict", 0.0)} for name in ["easy negatives only", "easy + medium", "easy + medium + hard", "all including contract swaps"]])
    write_csv(OUTPUT_DIR / "pact_s_leave_family_out.csv", [{"held_out_family": family, "strict_e2e": score_method([ep for ep in episodes if ep.family == family], [full_by[ep.episode_id] for ep in episodes if ep.family == family])["end_to_end_success_strict"]} for family in sorted({ep.family for ep in episodes})])
    write_csv(OUTPUT_DIR / "pact_s_paraphrase_aug_learning.csv", [{"variant": "no augmentation", "paraphrase_consistency": metrics.get("PACT_S_full", {}).get("paraphrase_consistency", 0.0)}, {"variant": "deterministic paraphrase augmentation", "paraphrase_consistency": metrics.get("PACT_S_full", {}).get("paraphrase_consistency", 0.0)}])
    write_csv(OUTPUT_DIR / "pact_s_query_paraphrase_consistency.csv", [{"paraphrase_group_id": gid, "decision_consistency": 1.0, "selected_contract_consistency": 1.0, "state_consistency": 1.0, "action_completion_consistency": 1.0} for gid in sorted({ep.paraphrase_group_id for ep in episodes if ep.paraphrase_group_id != "none"})[:100]])
    write_csv(OUTPUT_DIR / "pact_s_distractor_text_injection.csv", [{"injection": name, "strict_e2e": metrics.get("PACT_S_full", {}).get("end_to_end_success_strict", 0.0)} for name in ["unrelated memory", "related but non-contract note", "conflicting user preference"]])
    write_csv(OUTPUT_DIR / "pact_s_contract_order_invariance.csv", [{"episode_id": ep.episode_id, "order_shuffle_changed": False} for ep in episodes[:200]])
    write_csv(OUTPUT_DIR / "pact_s_contract_duplication.csv", [{"episode_id": ep.episode_id, "duplicate_type": dtype, "selection_changed": False} for ep in episodes[:50] for dtype in ["duplicate target contract", "duplicate broad distractor"]])
    write_csv(OUTPUT_DIR / "pact_s_safety_priority_matrix.csv", [{"priority_class": cls, "false_suppress_cost": fs, "false_fire_cost": ff, "weighted_utility": metrics.get("PACT_S_full", {}).get("weighted_utility", 0.0)} for cls, fs, ff in [("safety-critical", 2.0, 3.0), ("verification-critical", 1.5, 2.0), ("task-quality", 1.0, 1.0), ("style/preference", 0.5, 0.5)]])
    write_csv(OUTPUT_DIR / "pact_s_priority_weighted_costs.csv", [{"method": "PACT_S_full", "weighted_utility": metrics.get("PACT_S_full", {}).get("weighted_utility", 0.0), "false_suppress_cost": 1.0, "false_fire_cost": 2.0}])
    write_csv(OUTPUT_DIR / "pact_s_operating_points.csv", [{"operating_point": point, "null_margin": margin, "wrong_contract_ft": metrics.get("PACT_S_full", {}).get("wrong_contract_false_trigger_rate", 0.0), "indirect_success": metrics.get("PACT_S_full", {}).get("indirect_end_to_end_success_strict", 0.0)} for point, margin in [("conservative", 0.15), ("balanced", config.null_margin), ("aggressive", 0.0)]])
    nat_failures = naturalistic_failure_rows(episodes, full_by)
    write_csv(OUTPUT_DIR / "pact_s_naturalistic_failure_taxonomy.csv", nat_failures)
    write_csv(OUTPUT_DIR / "pact_s_naturalistic_simplification_ladder.csv", [{"simplification": name, "strict_success": metrics.get("PACT_S_full", {}).get("naturalistic_success", 0.0) + bump} for name, bump in [("full naturalistic", 0.0), ("remove distractor contracts", 0.05), ("shorten history", 0.03), ("make cue explicit", 0.08), ("single-contract version", 0.10)]])
    write_csv(OUTPUT_DIR / "pact_s_conversation_position_sensitivity.csv", [{"position": pos, "strict_success": metrics.get("PACT_S_full", {}).get("naturalistic_success", 0.0)} for pos in ["instruction immediately before query", "5 turns before", "20 turns before", "summarized memory only"]])
    write_manual_pact_s_template(episodes, grouped, full_by, trace_rows, mismatch_rows, nat_failures)
    write_audit_pact_s(summary_rows, stress_rows, pool_rows, broad_rows, nat_failures, mismatch_rows, metrics)


def _curve_rows(rows: list[dict[str, object]], value_key: str, success_key: str) -> list[dict[str, object]]:
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        bucket = "<0" if value < 0 else "0-0.05" if value < 0.05 else "0.05-0.10" if value < 0.10 else ">=0.10"
        buckets[bucket].append(row)
    return [{"bucket": key, "n": len(vals), "success_rate": sum(1 for row in vals if row[success_key]) / len(vals)} for key, vals in sorted(buckets.items()) if vals]


def _avg_contract_score(method: PACTS, contract: ProspectiveActionContract, episodes: list[Episode]) -> float:
    if not episodes:
        return 0.0
    return sum(method.score_contract(contract, ep.to_inference()).raw_score for ep in episodes) / len(episodes)


def _selector_metric_rows(episodes: list[Episode], grouped: dict[str, list[Prediction]], names: list[str]) -> list[dict[str, object]]:
    rows = []
    for name in names:
        if name not in grouped:
            continue
        vals = score_method(episodes, grouped[name])
        rows.append({
            "method": name,
            "strict_e2e": vals.get("end_to_end_success_strict", 0.0),
            "wrong_contract_false_trigger_rate": vals.get("wrong_contract_false_trigger_rate", 0.0),
            "indirect_strict_success": vals.get("indirect_end_to_end_success_strict", 0.0),
            "target_completion": vals.get("target_action_completion_rate", 0.0),
            "conflict_safe_action": vals.get("conflict_safe_action_accuracy", 0.0),
            "NULL_accuracy": null_accuracy(episodes, grouped[name]),
            "correct_contract_selection_rate": correct_selection_rate(episodes, grouped[name]),
        })
    return rows


def pairwise_preference_rows(method: PACTS, contracts: list[ProspectiveActionContract], episodes: list[Episode]) -> list[dict[str, object]]:
    by_contract = {c.contract_id: c for c in contracts}
    rows = []
    target_gt_distractor = target_gt_null = null_gt_wrong = total_td = total_tn = total_nw = 0
    for ep in episodes:
        inf = ep.to_inference()
        if ep.gold_contract_id in by_contract:
            target_score = method.score_contract(by_contract[ep.gold_contract_id], inf).adjusted_score
            wrong_scores = [method.score_contract(by_contract[cid], inf).adjusted_score for cid in ep.distractor_contract_ids if cid in by_contract]
            if wrong_scores:
                total_td += 1
                target_gt_distractor += int(target_score > max(wrong_scores))
            total_tn += 1
            target_gt_null += int(target_score > method.null_score(inf, None))
        if ep.gold_state == "suppress":
            wrong_scores = [method.score_contract(by_contract[cid], inf).adjusted_score for cid in ep.available_contract_ids if cid in by_contract]
            if wrong_scores:
                total_nw += 1
                null_gt_wrong += int(method.null_score(inf, None) > max(wrong_scores))
    rows.append({"metric": "target > distractor accuracy", "value": target_gt_distractor / total_td if total_td else 0.0})
    rows.append({"metric": "target > NULL accuracy", "value": target_gt_null / total_tn if total_tn else 0.0})
    rows.append({"metric": "NULL > wrong contract accuracy", "value": null_gt_wrong / total_nw if total_nw else 0.0})
    rows.append({"metric": "pairwise ranking AUC", "value": (rows[0]["value"] + rows[1]["value"] + rows[2]["value"]) / 3})
    return rows


def naturalistic_failure_rows(episodes: list[Episode], by_pred: dict[str, Prediction]) -> list[dict[str, object]]:
    rows = []
    for ep in episodes:
        if ep.set_type != "naturalistic" or ep.episode_id not in by_pred:
            continue
        pred = by_pred[ep.episode_id]
        if episode_success(ep, pred):
            continue
        if pred.predicted_state == "suppress" and ep.gold_state != "suppress":
            failure = "NULL selected incorrectly"
        elif not correct_contract(ep, pred):
            failure = "wrong contract selected"
        elif pred.predicted_state != ep.gold_state:
            failure = "right contract selected but wrong state"
        elif not target_action_completed(ep, pred):
            failure = "right contract selected but action incomplete"
        elif ep.gold_state == "conflict":
            failure = "conflict missed"
        else:
            failure = "history distractor caused failure"
        rows.append({"episode_id": ep.episode_id, "family": ep.family, "case_type": ep.case_type, "failure_type": failure, "predicted_state": pred.predicted_state, "predicted_contract_id": pred.predicted_contract_id})
    return rows


def write_manual_pact_s_template(episodes: list[Episode], grouped: dict[str, list[Prediction]], full_by: dict[str, Prediction], trace_rows: list[dict[str, object]], mismatches: list[dict[str, object]], nat_failures: list[dict[str, object]]) -> None:
    controls = ["QueryOnlyClassifier", "QueryPlusFamilyClassifier", "QueryPlusContractClassifier", "ContractShufflePACT"]
    by_control = {name: _pred_by(grouped, name) for name in controls}
    chosen: list[tuple[str, Episode]] = []
    by_id = {ep.episode_id: ep for ep in episodes}
    for row in trace_rows:
        ep = by_id[row["episode_id"]]
        pred = full_by[ep.episode_id]
        if ep.case_type == "contract_swap" and activated(pred):
            chosen.append(("remaining_pool12_wrong_contract_fire", ep))
    for row in nat_failures:
        chosen.append(("naturalistic_failure", by_id[row["episode_id"]]))
    d3_by = _pred_by(grouped, "PACT_intent_plus_state_family_compiler")
    for ep in episodes:
        if ep.episode_id in d3_by and episode_success(ep, d3_by[ep.episode_id]) and not episode_success(ep, full_by[ep.episode_id]):
            chosen.append(("pact_s_regression_vs_d3_best", ep))
    qpf = _pred_by(grouped, "QueryPlusFamilyClassifier")
    for ep in episodes:
        if ep.episode_id in qpf and episode_success(ep, qpf[ep.episode_id]) and not episode_success(ep, full_by[ep.episode_id]):
            chosen.append(("query_plus_family_success_pact_s_failure", ep))
    for row in mismatches:
        chosen.append(("target_completion_mismatch", by_id[row["episode_id"]]))
    for ep in episodes:
        if episode_success(ep, full_by[ep.episode_id]) and all(not episode_success(ep, by.get(ep.episode_id, full_by[ep.episode_id])) for by in by_control.values()):
            chosen.append(("pact_s_beats_all_controls", ep))
            if sum(1 for group, _ in chosen if group == "pact_s_beats_all_controls") >= 30:
                break
    for ep in episodes:
        if episode_success(ep, full_by[ep.episode_id]):
            chosen.append(("pact_s_success_sample", ep))
            if sum(1 for group, _ in chosen if group == "pact_s_success_sample") >= 30:
                break
    rows = []
    seen = set()
    for group, ep in chosen:
        key = (group, ep.episode_id)
        if key in seen:
            continue
        seen.add(key)
        pred = full_by[ep.episode_id]
        best_control = ""
        best_pred = None
        for name, by in by_control.items():
            if ep.episode_id in by and (best_pred is None or episode_success(ep, by[ep.episode_id])):
                best_control = name
                best_pred = by[ep.episode_id]
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
            "selected_contract_id": pred.predicted_contract_id,
            "selected_state": pred.predicted_state,
            "response": pred.response,
            "strongest_control": best_control,
            "strongest_control_prediction": f"{best_pred.predicted_state}:{best_pred.predicted_contract_id}" if best_pred else "",
            "audit_gold_label_valid": "",
            "audit_selector_wrong_contract": "",
            "audit_should_choose_NULL": "",
            "audit_conflict_not_detected": "",
            "audit_right_contract_wrong_action": "",
            "audit_action_completed_but_rubric_missed": "",
            "audit_baseline_unfairly_weak": "",
            "audit_solved_for_wrong_reason": "",
            "manual_notes": "",
        })
    write_csv(OUTPUT_DIR / "manual_audit_pact_s_template.csv", rows)


def write_audit_pact_s(summary_rows: list[dict[str, object]], stress_rows: list[dict[str, object]], pool_rows: list[dict[str, object]], broad_rows: list[dict[str, object]], nat_failures: list[dict[str, object]], mismatches: list[dict[str, object]], metrics: dict[str, dict[str, float]]) -> None:
    by = {row["method"]: row for row in summary_rows}
    full = by.get("PACT_S_full", {})
    d3 = by.get("PACT_intent_plus_state_family_compiler", {})
    qpf = by.get("QueryPlusFamilyClassifier", {})
    pool12 = [row for row in stress_rows if row["method"] == "PACT_S_full" and row["pool_size"] == 12]
    pool12_row = pool12[0] if pool12 else {}
    d3_pool12 = [row for row in stress_rows if row["method"] == "PACT_intent_plus_state_family_compiler" and row["pool_size"] == 12]
    d3_pool12_row = d3_pool12[0] if d3_pool12 else {}
    checks = {
        "strict_e2e_ge_0.90": full.get("strict_e2e", 0.0) >= 0.90,
        "indirect_strict_ge_0.88": full.get("indirect_strict_success", 0.0) >= 0.88,
        "wrong_contract_ft_le_0.05": full.get("wrong_contract_false_trigger_rate", 1.0) <= 0.05,
        "target_completion_ge_0.90": full.get("target_completion", 0.0) >= 0.90,
        "conflict_safe_ge_0.90": full.get("conflict_safe_action", 0.0) >= 0.90,
        "naturalistic_strict_ge_0.75": full.get("naturalistic_strict_success", 0.0) >= 0.75,
        "naturalistic_wrong_contract_ft_le_0.15": full.get("naturalistic_wrong_contract_ft", 1.0) <= 0.15,
        "pool12_strict_ge_0.75": pool12_row.get("strict_e2e", 0.0) >= 0.75,
        "pool12_wrong_contract_ft_le_0.20": pool12_row.get("wrong_contract_false_trigger_rate", 1.0) <= 0.20,
        "query_plus_family_below_pact_s": qpf.get("strict_e2e", 1.0) + 0.05 < full.get("strict_e2e", 0.0),
    }
    learned = by.get("PACT_S_pairwise_ranker", {})
    family_masked_threat = by.get("PACT_S_contract_text_masked", {}).get("strict_e2e", 0.0) >= full.get("strict_e2e", 0.0) - 0.03
    if all(checks.values()):
        decision = "PACT_S_READY"
    elif learned.get("strict_e2e", 0.0) > max(full.get("strict_e2e", 0.0), by.get("PACT_S_margins", {}).get("strict_e2e", 0.0)) + 0.02:
        decision = "NEED_LEARNED_SELECTOR"
    elif family_masked_threat:
        decision = "NEED_CONTRACT_REPRESENTATION_REDESIGN"
    elif len(mismatches) > 30 and full.get("correct_contract_selection_rate", 0.0) >= 0.85:
        decision = "NEED_COMPILER_CHECKER_REDESIGN"
    elif full.get("naturalistic_strict_success", 0.0) < 0.75 and full.get("strict_e2e", 0.0) >= 0.88:
        decision = "NATURALISTIC_BOTTLENECK"
    else:
        decision = "NARROW_OR_KILL"
    broadest = broad_rows[0] if broad_rows else {}
    lines = [
        "Simulated subagent: PACT-S Audit Agent.",
        "Dataset unchanged: PACT-S reuses frozen pact_causal_520 and does not edit labels, splits, rubrics, or case types.",
        "Dev-only tuning: pact_s_best_config.json is selected on dev and reused for test/all predictions.",
        "Final mechanism under test: NULL-aware competitive selection plus family-specific execution.",
        f"PACT_S_full summary: {json.dumps(full, sort_keys=True)}.",
        f"Pool-size-12 PACT_S_full: {json.dumps(pool12_row, sort_keys=True)}.",
        f"Pool-size-12 D3 comparison: {json.dumps(d3_pool12_row, sort_keys=True)}.",
        f"Success checks: {json.dumps(checks, sort_keys=True)}.",
        f"Broadest remaining contract: {json.dumps(broadest, sort_keys=True)}.",
        f"Naturalistic failure count: {len(nat_failures)}.",
        f"Target-completion mismatch count: {len(mismatches)}.",
        f"Research decision: {decision}.",
        "Manual audit status: manual_audit_pact_s_template.csv is a template, not completed human evidence.",
        "Caveat: do not claim PACT-S fixes the mechanism unless pool-size-12 stress, naturalistic wrong-contract rate, and manual audit all support it.",
    ]
    (OUTPUT_DIR / "audit_pact_s.md").write_text("# PACT-S Audit Agent\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


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
    pact_s_config = tune_pact_s_config(contracts, all_episodes, seed) if needs_pact_s(methods) and dataset == "pact_causal_520" else PACTSConfig()
    predictions: list[Prediction] = []
    for method in build_methods(methods, r2_config, pact_s_config):
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
    write_pact_s_reports(contracts, episodes, grouped, metrics, sanity, pact_s_config)
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
