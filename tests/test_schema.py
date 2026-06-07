from pact.dataset import build_contracts, build_episodes
from pact.schema import InferenceEpisode


def test_valid_contracts_and_episodes():
    assert len(build_contracts()) == 12
    episode = build_episodes("pact_causal_520")[0]
    assert episode.episode_id
    assert episode.gold_state in {"fire", "suppress", "conflict", "already_satisfied"}


def test_inference_episode_excludes_scoring_fields():
    inf = build_episodes("pact_causal_520")[0].to_inference()
    assert isinstance(inf, InferenceEpisode)
    for forbidden in [
        "gold_state",
        "case_type",
        "expected_action_keywords",
        "forbidden_action_keywords",
        "completion_rubric",
        "notes",
        "contrast_role",
        "paraphrase_group_id",
        "set_type",
    ]:
        assert not hasattr(inf, forbidden)

