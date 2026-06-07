from pact.stats import cluster_bootstrap_diff, holm, mcnemar, permutation_test
from pact.dataset import build_episodes
from pact.schema import Prediction


def _pred(ep, method, state):
    return Prediction(method, ep.episode_id, ep.contract_id, state, 1.0, " ".join(ep.expected_action_keywords), False, False, "")


def test_stats_edge_cases_and_determinism():
    eps = build_episodes("pact_causal_520")[:10]
    a = [_pred(ep, "a", ep.gold_state) for ep in eps]
    b = [_pred(ep, "b", "suppress") for ep in eps]
    ci = cluster_bootstrap_diff(eps, a, b, iters=20, seed=1)
    assert ci["ci_low"] <= ci["mean_diff"] <= ci["ci_high"]
    assert mcnemar([1, 1], [1, 1])["p"] == 1.0
    assert permutation_test([1, 0, 1], [0, 0, 1], iters=20, seed=2) == permutation_test([1, 0, 1], [0, 0, 1], iters=20, seed=2)
    adjusted = holm({"a": 0.01, "b": 0.04})
    assert adjusted["a"] <= adjusted["b"]

