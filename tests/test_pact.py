from pact.baselines import EpisodeInput
from pact.dataset import build_contracts
from pact.pact import PACTFull


def test_pact_checker_repair_increases_satisfaction():
    contracts = build_contracts()
    episode = EpisodeInput(
        episode_id="x",
        history_summary="No relevant prior steps are recorded.",
        current_query="Could Orbit-of-Thought work as a PRICAI submission if we frame it around agent planning?",
    )
    method = PACTFull()
    contract, score = method.retrieve(contracts, episode)
    pam = method.pam(contract, episode, score)
    initial = method.generate_response(contract, pam)
    assert not method.use_checker or not method.repair(contract, initial) == initial
    prediction = method.predict(contracts, episode)
    assert prediction.predicted_state == "fire"
    assert prediction.repaired
    assert prediction.satisfied

