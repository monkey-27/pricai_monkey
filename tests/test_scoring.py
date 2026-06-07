from pact.dataset import build_episodes
from pact.schema import Prediction
from pact.scoring import action_completed, score_method


def pred(ep, state, response):
    return Prediction("m", ep.episode_id, ep.contract_id, state, 1.0, response, True, False, "toy")


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

