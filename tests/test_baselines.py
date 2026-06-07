from pathlib import Path

from pact.baselines import KeywordTriggerBaseline, public_episode
from pact.dataset import build_contracts, build_episodes


def test_prediction_code_does_not_reference_scoring_only_fields():
    root = Path(__file__).resolve().parents[1] / "src" / "pact"
    text = (root / "baselines.py").read_text() + (root / "pact.py").read_text()
    for forbidden in (
        "gold_state",
        "case_type",
        "expected_action_keywords",
        "forbidden_action_keywords",
    ):
        assert forbidden not in text


def test_keyword_baseline_predicts_without_labels():
    contracts = build_contracts()
    episode = build_episodes()[0]
    prediction = KeywordTriggerBaseline().predict(contracts, public_episode(episode))
    assert prediction.episode_id == episode.episode_id
    assert prediction.predicted_state in {"fire", "suppress"}

