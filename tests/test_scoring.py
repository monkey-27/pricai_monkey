from pact.dataset import build_episodes
from pact.schema import Prediction
from pact.scoring import action_completed, episode_success_behavioral, episode_success_strict, score_method


def pred(ep, state, response):
    return Prediction("m", ep.episode_id, ep.contract_id, state, 1.0, response, state in {"fire", "conflict"} and bool(response), False, "toy")


def test_trigger_and_completion_metrics():
    ep = next(e for e in build_episodes("pact_causal_520") if e.gold_state == "fire")
    response = " ".join(ep.expected_action_keywords)
    metrics = score_method([ep], [pred(ep, "fire", response)])
    assert metrics["fire_precision"] == 1.0
    assert metrics["end_to_end_success"] == 1.0
    assert action_completed(ep, pred(ep, "fire", response))


def test_false_trigger_denominator_and_special_states():
    near = next(e for e in build_episodes("pact_causal_520") if e.case_type == "near_miss")
    already = next(e for e in build_episodes("pact_causal_520") if e.gold_state == "already_satisfied")
    conflict = next(e for e in build_episodes("pact_causal_520") if e.gold_state == "conflict")
    metrics = score_method(
        [near, already, conflict],
        [pred(near, "suppress", ""), pred(already, "already_satisfied", ""), pred(conflict, "conflict", " ".join(conflict.expected_action_keywords))],
    )
    assert metrics["false_trigger_rate"] == 0.0
    assert metrics["already_satisfied_accuracy"] == 1.0
    assert metrics["conflict_accuracy"] == 1.0
    assert metrics["weighted_utility"] > 0


def test_false_trigger_split_denominators():
    eps = build_episodes("pact_causal_520")
    near = next(e for e in eps if e.case_type == "near_miss")
    wrong = next(e for e in eps if e.case_type == "wrong_scope")
    swap = next(e for e in eps if e.case_type == "contract_swap")
    preds = [
        pred(near, "fire", "wrong action"),
        pred(wrong, "suppress", ""),
        pred(swap, "fire", "wrong contract action"),
    ]
    metrics = score_method([near, wrong, swap], preds)
    assert metrics["near_miss_false_trigger_rate"] == 1.0
    assert metrics["wrong_scope_false_trigger_rate"] == 0.0
    assert metrics["wrong_contract_false_trigger_rate"] == 1.0
    assert metrics["false_trigger_rate_excluding_contract_swap"] != metrics["false_trigger_rate_including_contract_swap"]


def test_conflict_detection_vs_safe_action():
    conflict = next(e for e in build_episodes("pact_causal_520") if e.gold_state == "conflict")
    safe_response = " ".join(conflict.expected_action_keywords)
    fire_safe = pred(conflict, "fire", safe_response)
    detected = pred(conflict, "conflict", safe_response)
    suppress = pred(conflict, "suppress", "")
    fire_metrics = score_method([conflict], [fire_safe])
    detected_metrics = score_method([conflict], [detected])
    suppress_metrics = score_method([conflict], [suppress])
    assert fire_metrics["conflict_detection_accuracy"] == 0.0
    assert fire_metrics["conflict_safe_action_accuracy"] == 1.0
    assert detected_metrics["conflict_detection_accuracy"] == 1.0
    assert suppress_metrics["conflict_safe_action_accuracy"] == 0.0


def test_action_completion_split_and_strict_behavioral():
    fire = next(e for e in build_episodes("pact_causal_520") if e.gold_state == "fire")
    wrong_contract = Prediction("m", fire.episode_id, "wrong_contract", "fire", 1.0, " ".join(fire.expected_action_keywords), True, False, "")
    assert not episode_success_strict(fire, wrong_contract)
    assert score_method([fire], [wrong_contract])["target_action_completion_rate"] == 0.0
    assert score_method([fire], [wrong_contract])["irrelevant_action_completion_rate"] == 1.0
    suppress = next(e for e in build_episodes("pact_causal_520") if e.gold_state == "suppress")
    assert episode_success_strict(suppress, pred(suppress, "suppress", ""))
    assert not episode_success_strict(suppress, pred(suppress, "fire", "wrong action"))
    conflict = next(e for e in build_episodes("pact_causal_520") if e.gold_state == "conflict")
    fire_safe = pred(conflict, "fire", " ".join(conflict.expected_action_keywords))
    assert not episode_success_strict(conflict, fire_safe)
    assert episode_success_behavioral(conflict, fire_safe)
