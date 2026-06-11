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


def activated(prediction: Prediction) -> bool:
    return prediction.predicted_state in {"fire", "conflict"} or prediction.action_completed


def correct_contract(episode: Episode, prediction: Prediction) -> bool:
    return prediction.predicted_contract_id in {episode.gold_contract_id, episode.target_contract_id}


def target_action_completed(episode: Episode, prediction: Prediction) -> bool:
    return action_completed(episode, prediction) and correct_contract(episode, prediction)


def predicted_contract_action_completed(prediction: Prediction) -> bool:
    return prediction.action_completed


def irrelevant_action_completed(episode: Episode, prediction: Prediction) -> bool:
    if not activated(prediction):
        return False
    if episode.gold_state == "suppress":
        return prediction.action_completed or prediction.predicted_contract_id != "none"
    return prediction.action_completed and not correct_contract(episode, prediction)


def conflict_safe_action(episode: Episode, prediction: Prediction) -> bool:
    if episode.gold_state != "conflict":
        return False
    return prediction.predicted_state in {"conflict", "fire"} and target_action_completed(episode, prediction)


def episode_success_strict(episode: Episode, prediction: Prediction) -> bool:
    if episode.gold_state == "fire":
        return prediction.predicted_state == "fire" and target_action_completed(episode, prediction)
    if episode.gold_state == "suppress":
        return not activated(prediction) and not irrelevant_action_completed(episode, prediction)
    if episode.gold_state == "already_satisfied":
        return prediction.predicted_state in {"already_satisfied", "suppress"} and not irrelevant_action_completed(episode, prediction)
    if episode.gold_state == "conflict":
        return prediction.predicted_state == "conflict" and conflict_safe_action(episode, prediction)
    return False


def episode_success_behavioral(episode: Episode, prediction: Prediction) -> bool:
    if episode.gold_state == "conflict":
        return conflict_safe_action(episode, prediction)
    return episode_success_strict(episode, prediction)


def episode_success(episode: Episode, prediction: Prediction) -> bool:
    return episode_success_strict(episode, prediction)


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
    near_wrong = near_wrong_false = 0
    near_miss = near_miss_false = wrong_scope = wrong_scope_false = 0
    suppress_total = suppress_false = suppress_no_swap = suppress_no_swap_false = 0
    contract_swap = contract_swap_false = contract_swap_wrong_action = 0
    already = already_correct = conflict = conflict_detected = conflict_safe = conflict_as_fire = conflict_as_suppress = 0
    target_action_total = target_action_done = predicted_action_total = predicted_action_done = irrelevant_action_total = irrelevant_action_done = 0
    repaired = repaired_done = unrepaired = unrepaired_done = 0
    swap_total = swap_success = 0
    para_total = para_success = controlled_total = controlled_success = 0
    naturalistic_total = naturalistic_success = scheduling_total = scheduling_success = 0
    e2e = []
    e2e_behavioral = []
    family_scores: dict[str, list[float]] = defaultdict(list)
    total_utility = 0.0
    for ep in episodes:
        p = by_id[ep.episode_id]
        completed = action_completed(ep, p)
        target_completed = target_action_completed(ep, p)
        success = episode_success_strict(ep, p)
        behavioral_success = episode_success_behavioral(ep, p)
        e2e.append(float(success))
        e2e_behavioral.append(float(behavioral_success))
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
            indirect_done += int(target_completed)
            indirect_success += int(success)
        if ep.case_type in {"near_miss", "wrong_scope"} and ep.gold_state == "suppress":
            near_wrong += 1
            near_wrong_false += int(activated(p))
        if ep.case_type == "near_miss" and ep.gold_state == "suppress":
            near_miss += 1
            near_miss_false += int(activated(p))
        if ep.case_type == "wrong_scope" and ep.gold_state == "suppress":
            wrong_scope += 1
            wrong_scope_false += int(activated(p))
        if ep.gold_state == "suppress":
            suppress_total += 1
            suppress_false += int(activated(p))
            if ep.case_type != "contract_swap":
                suppress_no_swap += 1
                suppress_no_swap_false += int(activated(p))
        if ep.case_type == "contract_swap" and ep.gold_state == "suppress":
            contract_swap += 1
            contract_swap_false += int(activated(p))
            contract_swap_wrong_action += int(activated(p) and p.predicted_contract_id != "none" and p.predicted_contract_id != ep.gold_contract_id and predicted_contract_action_completed(p))
        if ep.gold_state == "already_satisfied":
            already += 1
            already_correct += int(p.predicted_state == "already_satisfied")
        if ep.gold_state == "conflict":
            conflict += 1
            conflict_detected += int(p.predicted_state == "conflict")
            conflict_safe += int(conflict_safe_action(ep, p))
            conflict_as_fire += int(p.predicted_state == "fire")
            conflict_as_suppress += int(p.predicted_state == "suppress")
        if ep.gold_state in {"fire", "conflict"}:
            target_action_total += 1
            target_action_done += int(target_completed)
        if p.predicted_state in {"fire", "conflict"}:
            predicted_action_total += 1
            predicted_action_done += int(predicted_contract_action_completed(p))
        if ep.gold_state == "suppress" or not correct_contract(ep, p):
            irrelevant_action_total += 1
            irrelevant_action_done += int(irrelevant_action_completed(ep, p))
        if ep.gold_state == "fire" and p.predicted_state == "fire":
            if p.repaired:
                repaired += 1
                repaired_done += int(target_completed)
            else:
                unrepaired += 1
                unrepaired_done += int(target_completed)
        if ep.case_type == "contract_swap":
            swap_total += 1
            swap_success += int(success)
        if ep.set_type == "paraphrase":
            para_total += 1
            para_success += int(success)
        if ep.set_type == "controlled":
            controlled_total += 1
            controlled_success += int(success)
        if ep.set_type == "naturalistic":
            naturalistic_total += 1
            naturalistic_success += int(success)
        if ep.family == "scheduling":
            scheduling_total += 1
            scheduling_success += int(success)
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
        "near_miss_false_trigger_rate": div(near_miss_false, near_miss),
        "wrong_scope_false_trigger_rate": div(wrong_scope_false, wrong_scope),
        "wrong_contract_false_trigger_rate": div(contract_swap_false, contract_swap),
        "overall_false_trigger_rate": div(suppress_false, suppress_total),
        "false_trigger_rate_excluding_contract_swap": div(suppress_no_swap_false, suppress_no_swap),
        "false_trigger_rate_including_contract_swap": div(suppress_false, suppress_total),
        "already_satisfied_accuracy": div(already_correct, already),
        "conflict_accuracy": div(conflict_detected, conflict),
        "conflict_detection_accuracy": div(conflict_detected, conflict),
        "conflict_safe_action_accuracy": div(conflict_safe, conflict),
        "conflict_as_fire_rate": div(conflict_as_fire, conflict),
        "conflict_as_suppress_rate": div(conflict_as_suppress, conflict),
        "target_action_completion_rate": div(target_action_done, target_action_total),
        "predicted_contract_action_completion_rate": div(predicted_action_done, predicted_action_total),
        "irrelevant_action_completion_rate": div(irrelevant_action_done, irrelevant_action_total),
        "wrong_contract_action_completion_rate": div(contract_swap_wrong_action, contract_swap),
        "checker_repair_gain": div(repaired_done, repaired) - div(unrepaired_done, unrepaired),
        "contract_swap_sensitivity": div(swap_success, swap_total),
        "paraphrase_consistency": div(para_success, para_total) - div(controlled_success, controlled_total),
        "per_family_variance": pstdev(family_means) if len(family_means) > 1 else 0.0,
        "end_to_end_success": div(sum(e2e), len(e2e)),
        "end_to_end_success_strict": div(sum(e2e), len(e2e)),
        "end_to_end_success_behavioral": div(sum(e2e_behavioral), len(e2e_behavioral)),
        "end_to_end_success_indirect": div(indirect_success, indirect),
        "indirect_end_to_end_success_strict": div(indirect_success, indirect),
        "end_to_end_success_near_miss": _case_success(episodes, by_id, "near_miss"),
        "end_to_end_success_wrong_scope": _case_success(episodes, by_id, "wrong_scope"),
        "naturalistic_success": div(naturalistic_success, naturalistic_total),
        "scheduling_family_success": div(scheduling_success, scheduling_total),
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
    return [int(episode_success_strict(ep, by_id[ep.episode_id])) for ep in episodes]


def format_summary(metrics: dict[str, dict[str, float]]) -> str:
    cols = ["method", "end_to_end_success_strict", "end_to_end_success_behavioral", "indirect_action_completion", "false_trigger_rate_including_contract_swap", "wrong_contract_false_trigger_rate", "conflict_detection_accuracy"]
    rows = [" | ".join(cols), " | ".join(["---"] * len(cols))]
    for method, vals in metrics.items():
        rows.append(" | ".join([method] + [f"{vals[c]:.3f}" for c in cols[1:]]))
    return "\n".join(rows)
