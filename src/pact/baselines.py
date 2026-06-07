"""Deterministic baselines for PACT-Causal-520."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from pact.schema import InferenceEpisode, Prediction, ProspectiveActionContract

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = {"a", "an", "and", "are", "as", "for", "if", "in", "is", "it", "me", "my", "of", "or", "the", "this", "to", "when", "with"}
NEAR = ("what is", "what does", "explain", "define", "why do", "why use", "why avoid", "summarize", "how does", "give tips")
ALREADY = ("already", "previous turn", "was checked", "were checked", "has been reviewed", "is done")
CONFLICT = ("do not mention", "skip", "guarantee", "pretend", "without checking", "disable", "harmless", "same as", "prove it")
WRONG = (
    "fictional", "history essay", "poster", "road trip", "cron", "laptops", "phd", "poem", "railroads",
    "hello page", "pirate", "friend", "short story", "menu", "font", "docstring", "fantasy", "diary",
    "keyboards", "wi-fi", "phone cameras", "pride and prejudice", "liberal arts", "mba", "icon",
    "theme of a novel", "chart title", "variables", "health-check", "side effect", "jargon",
)


class Method(Protocol):
    name: str

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        ...


def tokenize(text: str) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(text.lower()) if tok not in STOP]


def contains(text: str, phrases: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in phrases)


def overlap(a: str, b: str) -> float:
    left, right = set(tokenize(a)), set(tokenize(b))
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def allowed_contracts(contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> list[ProspectiveActionContract]:
    by_id = {c.contract_id: c for c in contracts}
    return [by_id[cid] for cid in episode.available_contract_ids if cid in by_id] or contracts


def best_contract(contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> tuple[ProspectiveActionContract, float]:
    scored = [(c, max(overlap(c.raw_text, episode.text), overlap(c.cue + " " + c.guard, episode.text))) for c in allowed_contracts(contracts, episode)]
    scored.sort(key=lambda item: (item[1], item[0].contract_id), reverse=True)
    return scored[0]


def compiled_plan(contract: ProspectiveActionContract) -> str:
    return f"Plan: {contract.action} Check: {contract.check}"


def generic_response(contract: ProspectiveActionContract, complete: bool) -> str:
    if complete:
        return f"{compiled_plan(contract)} I will address the request after completing that prospective action."
    return "I found a relevant memory and will proceed carefully, but I will not spell out the required action."


def rough_complete(contract: ProspectiveActionContract, response: str) -> bool:
    tokens = set(tokenize(response))
    required = [tok for tok in tokenize(contract.action + " " + contract.check) if len(tok) > 5]
    return len(tokens & set(required)) >= min(3, len(set(required)))


def make_prediction(method: str, episode: InferenceEpisode, contract: ProspectiveActionContract | None,
                    state: str, confidence: float, response: str, repaired: bool, rationale: str) -> Prediction:
    return Prediction(
        method=method,
        episode_id=episode.episode_id,
        predicted_contract_id=contract.contract_id if contract and state != "suppress" else "none",
        predicted_state=state,
        confidence=confidence,
        response=response,
        action_completed=rough_complete(contract, response) if contract and state in {"fire", "conflict"} else False,
        repaired=repaired,
        rationale=rationale,
    )


class NoMemory:
    name = "NoMemory"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        return make_prediction(self.name, episode, None, "suppress", 0.9, "No stored prospective action is used.", False, "always suppress")


class KeywordTrigger:
    name = "KeywordTrigger"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        c, score = best_contract(contracts, episode)
        fires = score >= 0.22 and not contains(episode.text, NEAR + WRONG)
        return make_prediction(self.name, episode, c, "fire" if fires else "suppress", score, generic_response(c, fires), False, f"keyword score={score:.3f}")


class TfidfRawMemory:
    name = "TfidfRawMemory"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        pool = allowed_contracts(contracts, episode)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            docs = [c.raw_text for c in pool] + [episode.text]
            matrix = TfidfVectorizer(stop_words="english").fit_transform(docs)
            sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
            idx = int(sims.argmax())
            c, score = pool[idx], float(sims[idx])
        except Exception:
            c, score = best_contract(contracts, episode)
        fires = score >= 0.18
        return make_prediction(self.name, episode, c, "fire" if fires else "suppress", score, generic_response(c, fires), False, f"raw memory similarity={score:.3f}")


class FullHistory:
    name = "FullHistory"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        c, score = best_contract(contracts, episode)
        if contains(episode.text, ALREADY):
            state = "already_satisfied"
        elif contains(episode.text, CONFLICT):
            state = "conflict"
        elif contains(episode.text, NEAR + WRONG):
            state = "suppress"
        else:
            state = "fire" if score >= 0.24 else "suppress"
        return make_prediction(self.name, episode, c, state, min(0.95, score + 0.2), generic_response(c, state in {"fire", "conflict"}), False, "whole-history deterministic heuristic")


class RawMemorySelfCheck:
    name = "RawMemorySelfCheck"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        c, score = best_contract(contracts, episode)
        fires = score >= 0.20 and not contains(episode.text, NEAR + WRONG + ALREADY)
        response = generic_response(c, complete=False)
        if fires and not rough_complete(c, response):
            response = response + " " + c.action
        return make_prediction(self.name, episode, c, "fire" if fires else "suppress", score, response, fires, "raw memory plus self-check")


class QueryOnlyClassifier:
    name = "QueryOnlyClassifier"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        text = episode.text
        if contains(text, ALREADY):
            state = "already_satisfied"
        elif contains(text, CONFLICT):
            state = "conflict"
        elif contains(text, NEAR + WRONG):
            state = "suppress"
        else:
            state = "fire" if len(set(tokenize(text)) & {"review", "plan", "help", "can", "should", "analyze", "rewrite"}) >= 1 else "suppress"
        return make_prediction(self.name, episode, None, state, 0.55, "Query-only response without stored contract.", False, "no contract fields used")


class LabelPermutationSanity:
    name = "LabelPermutationSanity"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        states = ["suppress", "fire", "already_satisfied", "conflict"]
        state = states[sum(ord(ch) for ch in episode.episode_id) % len(states)]
        c = allowed_contracts(contracts, episode)[0]
        return make_prediction(self.name, episode, c, state, 0.25, "Deterministic permuted-label sanity response.", False, "permuted sanity")


class LLMStub:
    def __init__(self, name: str) -> None:
        self.name = name

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        c, _ = best_contract(contracts, episode)
        return make_prediction(self.name, episode, c, "suppress", 0.0, "LLM backend disabled; no external API call was made.", False, "stub")
