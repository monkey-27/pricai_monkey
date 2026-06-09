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
from pact.pact import METHOD_NAMES, get_method
from pact.schema import Episode, Prediction
from pact.scoring import activated, binary_successes, episode_success, format_summary, group_predictions, score_method
from pact.stats import cluster_bootstrap_diff, holm, mcnemar, permutation_test

OUTPUT_DIR = Path(os.environ.get("PACT_OUTPUT_DIR", ROOT / "outputs"))


def build_methods(methods: str):
    names = METHOD_NAMES if methods == "all" else [m.strip() for m in methods.split(",") if m.strip()]
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
            out.append(get_method(name))
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


def write_manual_sample(episodes: list[Episode], grouped: dict[str, list[Prediction]]) -> None:
    pact = grouped.get("PACTFull", [])
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
    episodes = filter_split(load_episodes(dataset), split)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions: list[Prediction] = []
    for method in build_methods(methods):
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
