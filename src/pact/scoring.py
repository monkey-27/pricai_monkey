"""Scoring utilities for the PACT pilot."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from pact.schema import Episode, Prediction


def _safe_div(num: float, denom: float) -> float:
    return 0.0 if denom == 0 else num / denom


def response_satisfies_episode(episode: Episode, prediction: Prediction) -> bool:
    if prediction.predicted_state not in {"fire", "conflict"}:
        return False
    response = prediction.response.lower()
    expected_ok = all(keyword.lower() in response for keyword in episode.expected_action_keywords)
    forbidden_ok = not any(keyword.lower() in response for keyword in episode.forbidden_action_keywords)
    return expected_ok and forbidden_ok


def score_method(episodes: list[Episode], predictions: list[Prediction]) -> dict[str, float]:
    by_id = {prediction.episode_id: prediction for prediction in predictions}
    if set(by_id) != {episode.episode_id for episode in episodes}:
        missing = sorted({episode.episode_id for episode in episodes} - set(by_id))
        extra = sorted(set(by_id) - {episode.episode_id for episode in episodes})
        raise ValueError(f"prediction ids mismatch missing={missing[:3]} extra={extra[:3]}")

    trigger_correct = 0
    fire_tp = fire_fp = fire_fn = 0
    indirect_fire = indirect_done = 0
    gold_fire = gold_fire_done = 0
    near_wrong_total = near_wrong_false = 0
    conflict_total = conflict_correct = 0
    already_total = already_correct = 0
    repaired_fire = repaired_fire_done = unrepaired_fire_done = 0

    for episode in episodes:
        prediction = by_id[episode.episode_id]
        gold = episode.gold_state
        predicted = prediction.predicted_state
        if predicted == gold:
            trigger_correct += 1
        if gold == "fire" and predicted == "fire":
            fire_tp += 1
        elif gold != "fire" and predicted == "fire":
            fire_fp += 1
        elif gold == "fire" and predicted != "fire":
            fire_fn += 1

        satisfied = response_satisfies_episode(episode, prediction)
        if gold == "fire":
            gold_fire += 1
            if satisfied:
                gold_fire_done += 1
        if gold == "fire" and episode.case_type == "indirect_trigger":
            indirect_fire += 1
            if satisfied:
                indirect_done += 1
        if episode.case_type in {"near_miss", "wrong_scope"} and gold == "suppress":
            near_wrong_total += 1
            if predicted == "fire":
                near_wrong_false += 1
        if gold == "conflict":
            conflict_total += 1
            conflict_correct += int(predicted == "conflict")
        if gold == "already_satisfied":
            already_total += 1
            already_correct += int(predicted == "already_satisfied")
        if gold == "fire" and predicted == "fire":
            if prediction.repaired:
                repaired_fire += 1
                repaired_fire_done += int(satisfied)
            else:
                unrepaired_fire_done += int(satisfied)

    precision = _safe_div(fire_tp, fire_tp + fire_fp)
    recall = _safe_div(fire_tp, fire_tp + fire_fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    repaired_rate = _safe_div(repaired_fire_done, repaired_fire)
    unrepaired_rate = _safe_div(unrepaired_fire_done, max(1, fire_tp - repaired_fire))
    return {
        "trigger_accuracy": _safe_div(trigger_correct, len(episodes)),
        "fire_precision": precision,
        "fire_recall": recall,
        "fire_f1": f1,
        "indirect_trigger_recall": _safe_div(
            sum(
                1
                for episode in episodes
                if episode.case_type == "indirect_trigger"
                and episode.gold_state == "fire"
                and by_id[episode.episode_id].predicted_state == "fire"
            ),
            sum(1 for episode in episodes if episode.case_type == "indirect_trigger" and episode.gold_state == "fire"),
        ),
        "false_trigger_rate_near_wrong": _safe_div(near_wrong_false, near_wrong_total),
        "action_completion_rate_gold_fire": _safe_div(gold_fire_done, gold_fire),
        "action_completion_rate_indirect_fire": _safe_div(indirect_done, indirect_fire),
        "conflict_accuracy": _safe_div(conflict_correct, conflict_total),
        "already_satisfied_accuracy": _safe_div(already_correct, already_total),
        "checker_repair_gain": repaired_rate - unrepaired_rate,
    }


def group_predictions(predictions: Iterable[Prediction]) -> dict[str, list[Prediction]]:
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.method].append(prediction)
    return dict(grouped)


def format_summary(metrics: dict[str, dict[str, float]]) -> str:
    columns = [
        "method",
        "trigger_accuracy",
        "fire_precision",
        "fire_recall",
        "indirect_trigger_recall",
        "false_trigger_rate_near_wrong",
        "action_completion_rate_indirect_fire",
    ]
    header = " | ".join(columns)
    sep = " | ".join(["---"] * len(columns))
    rows = [header, sep]
    for method, method_metrics in metrics.items():
        values = [method] + [f"{method_metrics[col]:.3f}" for col in columns[1:]]
        rows.append(" | ".join(values))
    return "\n".join(rows)

