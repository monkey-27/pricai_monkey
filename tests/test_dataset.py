from collections import Counter, defaultdict

from pact.dataset import build_contracts, build_episodes


def test_dataset_size_and_family_distribution():
    contracts = build_contracts()
    episodes = build_episodes()
    assert len(contracts) == 10
    assert len(episodes) == 100
    by_family = defaultdict(list)
    for episode in episodes:
        by_family[episode.family].append(episode)
    assert set(by_family) == {contract.family for contract in contracts}
    assert all(len(items) == 10 for items in by_family.values())


def test_each_family_has_required_case_type_distribution():
    by_family = defaultdict(list)
    for episode in build_episodes():
        by_family[episode.family].append(episode)
    for episodes in by_family.values():
        counts = Counter(episode.case_type for episode in episodes)
        assert counts["direct_trigger"] == 2
        assert counts["indirect_trigger"] == 3
        assert counts["near_miss"] == 3
        assert counts["wrong_scope"] == 1
        assert counts["conflict"] + counts["already_satisfied"] == 1

