import pytest

from pact.schema import Episode, ProspectiveActionContract


def test_contract_validation_rejects_bad_priority():
    with pytest.raises(ValueError):
        ProspectiveActionContract.from_dict(
            {
                "contract_id": "x",
                "family": "f",
                "cue": "cue",
                "guard": "guard",
                "action": "action",
                "check": "check",
                "priority": "urgent",
                "status": "active",
            }
        )


def test_episode_validation_rejects_bad_case_type():
    with pytest.raises(ValueError):
        Episode.from_dict(
            {
                "episode_id": "e",
                "contract_id": "c",
                "family": "f",
                "case_type": "maybe",
                "history_summary": "history",
                "current_query": "query",
                "gold_state": "fire",
                "expected_action_keywords": [],
                "forbidden_action_keywords": [],
                "notes": "notes",
            }
        )

