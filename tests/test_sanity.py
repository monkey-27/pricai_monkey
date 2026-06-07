from pact.dataset import build_contracts, build_episodes
from pact.baselines import QueryOnlyClassifier


def test_query_only_cannot_use_contract_fields():
    ep = build_episodes("pact_causal_520")[0].to_inference()
    changed = type(ep)(ep.episode_id, ep.history_summary, ep.current_query, ["totally_different"])
    method = QueryOnlyClassifier()
    assert method.predict(build_contracts(), ep).predicted_state == method.predict(build_contracts(), changed).predicted_state

