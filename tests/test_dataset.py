from collections import Counter, defaultdict

from pact.dataset import build_episodes


def test_causal_520_counts():
    episodes = build_episodes("pact_causal_520")
    assert len(episodes) == 520
    assert Counter(ep.set_type for ep in episodes) == {"controlled": 360, "paraphrase": 120, "naturalistic": 40}
    assert len({ep.family for ep in episodes}) == 12


def test_controlled_family_distribution():
    controlled = [ep for ep in build_episodes("pact_causal_520") if ep.set_type == "controlled"]
    by_family = defaultdict(list)
    for ep in controlled:
        by_family[ep.family].append(ep)
    required = {
        "direct_trigger": 4,
        "indirect_trigger": 6,
        "near_miss": 6,
        "wrong_scope": 4,
        "already_satisfied": 3,
        "conflict": 3,
        "contract_swap": 4,
    }
    for eps in by_family.values():
        assert len(eps) == 30
        assert Counter(ep.case_type for ep in eps) == required


def test_paraphrase_groups_and_contrasts_and_split():
    episodes = build_episodes("pact_causal_520")
    groups = defaultdict(set)
    for ep in episodes:
        if ep.paraphrase_group_id != "none":
            groups[ep.paraphrase_group_id].add(ep.gold_state)
    assert groups
    assert all(len(labels) == 1 for labels in groups.values())
    roles = {ep.contrast_group_id.split("::")[1] for ep in episodes if "::" in ep.contrast_group_id}
    assert {
        "cue_present_vs_absent",
        "guard_satisfied_vs_guard_violated",
        "indirect_cue_vs_near_miss",
        "action_needed_vs_already_satisfied",
        "relevant_contract_vs_swapped_contract",
        "conflict_vs_no_conflict",
    } <= roles
    assert {ep.split for ep in episodes} == {"dev", "test"}

