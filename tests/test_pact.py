from pact.dataset import build_contracts, build_episodes
from pact.pact import PACTFull
from pact.scoring import action_completed, score_method


def test_checker_repair_increases_completion():
    contracts = build_contracts()
    ep = next(e for e in build_episodes("pact_causal_520") if e.case_type == "indirect_trigger")
    full = PACTFull().predict(contracts, ep.to_inference())
    no_checker = PACTFull(use_checker=False, name="PACT_no_checker").predict(contracts, ep.to_inference())
    assert full.repaired
    assert action_completed(ep, full)
    assert not action_completed(ep, no_checker)


def test_no_guard_and_shuffle_controls_on_toys():
    contracts = build_contracts()
    eps = [e for e in build_episodes("pact_causal_520") if e.case_type in {"wrong_scope", "indirect_trigger"}][:8]
    full = [PACTFull().predict(contracts, e.to_inference()) for e in eps]
    no_guard = [PACTFull(use_guard=False, name="PACT_no_guard").predict(contracts, e.to_inference()) for e in eps]
    shuffle = [PACTFull(shuffle=True, name="ContractShufflePACT").predict(contracts, e.to_inference()) for e in eps]
    assert score_method(eps, no_guard)["false_trigger_rate"] >= score_method(eps, full)["false_trigger_rate"]
    assert score_method(eps, shuffle)["end_to_end_success"] <= score_method(eps, full)["end_to_end_success"]

