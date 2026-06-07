from pathlib import Path

from pact.baselines import QueryOnlyClassifier
from pact.dataset import build_contracts, build_episodes


def test_prediction_modules_do_not_reference_forbidden_fields():
    root = Path(__file__).resolve().parents[1] / "src" / "pact"
    text = (root / "baselines.py").read_text() + (root / "pact.py").read_text()
    for forbidden in [
        "gold_state",
        "case_type",
        "expected_action_keywords",
        "forbidden_action_keywords",
        "completion_rubric",
        "contrast_role",
        "paraphrase_group_id",
    ]:
        assert forbidden not in text


def test_runtime_prediction_accepts_only_inference_episode():
    ep = build_episodes("pact_causal_520")[0]
    pred = QueryOnlyClassifier().predict(build_contracts(), ep.to_inference())
    assert pred.episode_id == ep.episode_id
    assert pred.predicted_state in {"fire", "suppress", "conflict", "already_satisfied"}

