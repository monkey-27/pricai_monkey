"""Metrics for PACT-Causal-520."""

from __future__ import annotations

from collections import defaultdict
from statistics import pstdev

from pact.schema import Episode, Prediction


def div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def action_completed(episode: Episode, prediction: Prediction) -> bool:
    if prediction.predicted_state not in {"fire", "conflict"}:
        return False
    text = prediction.response.lower()
    required = all(item.lower() in text for item in episode.expected_action_keywords)
    forbidden = any(item.lower() in text for item in episode.forbidden_action_keywords)
    return required and not forbidden


def episode_success(episode: Episode, prediction: Prediction) -> bool:
    completed = action_completed(episode, prediction)
    if episode.gold_state == "fire":
        return prediction.predicted_state == "fire" and completed
    if episode.gold_state == "suppress":
        return prediction.predicted_state == "suppress"
    if episode.gold_state == "already_satisfied":
        return prediction.predicted_state in {"already_satisfied", "suppress"} and not completed
    if episode.gold_state == "conflict":
        return prediction.predicted_state == "conflict" and completed
    return False


def utility(episode: Episode, prediction: Prediction) -> float:
    completed = action_completed(episode, prediction)
    if episode.gold_state == "fire" and prediction.predicted_state == "fire" and completed:
        return 1.0
    if episode.gold_state == "suppress" and prediction.predicted_state == "suppress":
        return 1.0
    if episode.gold_state == "already_satisfied" and prediction.predicted_state == "fire":
        return -0.5
    if episode.gold_state == "suppress" and prediction.predicted_state == "fire":
        return -1.0
    if episode.gold_state == "conflict" and prediction.predicted_state != "conflict" and episode.priority_expectation == "safety":
        return -1.5
    return 0.0


def score_method(episodes: list[Episode], predictions: list[Prediction]) -> dict[str, float]:
    by_id = {p.episode_id: p for p in predictions}
    fire_tp = fire_fp = fire_fn = trigger_correct = 0
    indirect = indirect_fire = indirect_done = indirect_success = 0
    near_wrong = near_wrong_false = wrong_scope = wrong_scope_false = 0
    already = already_correct = conflict = conflict_correct = 0
    repaired = repaired_done = unrepaired = unrepaired_done = 0
    swap_total = swap_success = 0
    para_total = para_success = controlled_total = controlled_success = 0
    e2e = []
    family_scores: dict[str, list[float]] = defaultdict(list)
    total_utility = 0.0
    for ep in episodes:
        p = by_id[ep.episode_id]
        completed = action_completed(ep, p)
        success = episode_success(ep, p)
        e2e.append(float(success))
        family_scores[ep.family].append(float(success))
        total_utility += utility(ep, p)
        if p.predicted_state == ep.gold_state:
            trigger_correct += 1
        if ep.gold_state == "fire" and p.predicted_state == "fire":
            fire_tp += 1
        elif ep.gold_state != "fire" and p.predicted_state == "fire":
            fire_fp += 1
        elif ep.gold_state == "fire" and p.predicted_state != "fire":
            fire_fn += 1
        if ep.case_type == "indirect_trigger":
            indirect += 1
            indirect_fire += int(p.predicted_state == "fire")
            indirect_done += int(completed)
            indirect_success += int(success)
        if ep.case_type in {"near_miss", "wrong_scope"} and ep.gold_state == "suppress":
            near_wrong += 1
            near_wrong_false += int(p.predicted_state == "fire")
        if ep.case_type == "wrong_scope":
            wrong_scope += 1
            wrong_scope_false += int(p.predicted_state == "fire")
        if ep.gold_state == "already_satisfied":
            already += 1
            already_correct += int(p.predicted_state == "already_satisfied")
        if ep.gold_state == "conflict":
            conflict += 1
            conflict_correct += int(p.predicted_state == "conflict")
        if ep.gold_state == "fire" and p.predicted_state == "fire":
            if p.repaired:
                repaired += 1
                repaired_done += int(completed)
            else:
                unrepaired += 1
                unrepaired_done += int(completed)
        if ep.case_type == "contract_swap":
            swap_total += 1
            swap_success += int(success)
        if ep.set_type == "paraphrase":
            para_total += 1
            para_success += int(success)
        if ep.set_type == "controlled":
            controlled_total += 1
            controlled_success += int(success)
    precision = div(fire_tp, fire_tp + fire_fp)
    recall = div(fire_tp, fire_tp + fire_fn)
    family_means = [sum(vals) / len(vals) for vals in family_scores.values()]
    return {
        "trigger_accuracy": div(trigger_correct, len(episodes)),
        "fire_precision": precision,
        "fire_recall": recall,
        "fire_f1": div(2 * precision * recall, precision + recall),
        "indirect_trigger_recall": div(indirect_fire, indirect),
        "indirect_action_completion": div(indirect_done, indirect),
        "false_trigger_rate": div(near_wrong_false, near_wrong),
        "wrong_scope_false_trigger_rate": div(wrong_scope_false, wrong_scope),
        "already_satisfied_accuracy": div(already_correct, already),
        "conflict_accuracy": div(conflict_correct, conflict),
        "checker_repair_gain": div(repaired_done, repaired) - div(unrepaired_done, unrepaired),
        "contract_swap_sensitivity": div(swap_success, swap_total),
        "paraphrase_consistency": div(para_success, para_total) - div(controlled_success, controlled_total),
        "per_family_variance": pstdev(family_means) if len(family_means) > 1 else 0.0,
        "end_to_end_success": div(sum(e2e), len(e2e)),
        "end_to_end_success_indirect": div(indirect_success, indirect),
        "end_to_end_success_near_miss": _case_success(episodes, by_id, "near_miss"),
        "end_to_end_success_wrong_scope": _case_success(episodes, by_id, "wrong_scope"),
        "weighted_utility": div(total_utility, len(episodes)),
    }


def _case_success(episodes: list[Episode], by_id: dict[str, Prediction], case: str) -> float:
    vals = [episode_success(ep, by_id[ep.episode_id]) for ep in episodes if ep.case_type == case]
    return div(sum(vals), len(vals))


def group_predictions(predictions: list[Prediction]) -> dict[str, list[Prediction]]:
    out: dict[str, list[Prediction]] = defaultdict(list)
    for pred in predictions:
        out[pred.method].append(pred)
    return dict(out)


def binary_successes(episodes: list[Episode], predictions: list[Prediction]) -> list[int]:
    by_id = {p.episode_id: p for p in predictions}
    return [int(episode_success(ep, by_id[ep.episode_id])) for ep in episodes]


def format_summary(metrics: dict[str, dict[str, float]]) -> str:
    cols = ["method", "end_to_end_success", "end_to_end_success_indirect", "indirect_action_completion", "false_trigger_rate", "fire_precision", "contract_swap_sensitivity"]
    rows = [" | ".join(cols), " | ".join(["---"] * len(cols))]
    for method, vals in metrics.items():
        rows.append(" | ".join([method] + [f"{vals[c]:.3f}" for c in cols[1:]]))
    return "\n".join(rows)

