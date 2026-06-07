from pact.scoring import score_method
from pact.schema import Episode, Prediction


def _episode(gold_state="fire", case_type="direct_trigger"):
    return Episode.from_dict(
        {
            "episode_id": "e1",
            "contract_id": "c1",
            "family": "f",
            "case_type": case_type,
            "history_summary": "history",
            "current_query": "query",
            "gold_state": gold_state,
            "expected_action_keywords": ["alpha", "beta"],
            "forbidden_action_keywords": ["bad"],
            "notes": "toy",
        }
    )


def _prediction(state="fire", response="alpha beta"):
    return Prediction(
        method="m",
        episode_id="e1",
        contract_id="c1",
        predicted_state=state,
        confidence=1.0,
        response=response,
        satisfied=True,
        repaired=False,
        rationale="toy",
    )


def test_scoring_toy_fire_completion():
    metrics = score_method([_episode()], [_prediction()])
    assert metrics["trigger_accuracy"] == 1.0
    assert metrics["fire_precision"] == 1.0
    assert metrics["action_completion_rate_gold_fire"] == 1.0


def test_scoring_zero_division_no_fire_predictions():
    metrics = score_method([_episode(gold_state="suppress", case_type="near_miss")], [_prediction(state="suppress", response="")])
    assert metrics["fire_precision"] == 0.0
    assert metrics["false_trigger_rate_near_wrong"] == 0.0

