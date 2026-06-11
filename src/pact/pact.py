"""PACTFull, contract-aware baselines, and ablations."""

from __future__ import annotations

from dataclasses import dataclass

from pact.baselines import (
    ALREADY,
    CONFLICT,
    NEAR,
    WRONG,
    Method,
    allowed_contracts,
    best_contract,
    compiled_plan,
    contains,
    generic_response,
    make_prediction,
    overlap,
    rough_complete,
)
from pact.schema import InferenceEpisode, Prediction, ProspectiveActionContract

ACTION_SEEKING = (
    "could",
    "should",
    "can i",
    "can you",
    "help",
    "review",
    "plan",
    "analyze",
    "rewrite",
    "choose",
    "recommend",
    "assess",
    "interpret",
    "build",
    "schedule",
    "audit",
    "compare",
    "propose",
    "evaluate",
    "tell me",
    "make",
    "would",
    "i need",
    "shift",
    "which",
    "what should",
    "is this",
    "does this",
    "rank",
)
FAMILY_HINTS = {
    "research_ideation": {"paper", "pricai", "publishable", "submission", "benchmark", "agent", "novel", "method"},
    "food_safety": {"potluck", "restaurant", "bakery", "cake", "cookies", "granola", "lunch", "dessert", "takeout", "food"},
    "code_security": {"flask", "login", "secret", "secret_key", "cookie", "session", "auth", "decorator", "deploy"},
    "travel_planning": {"tokyo", "seoul", "taipei", "istanbul", "baku", "airport", "layover", "london", "paris", "vietnam", "cambodia", "trip", "route"},
    "medical_caution": {"headache", "rash", "dizzy", "medication", "pill", "throat", "fever", "chest", "antibiotic", "medicine"},
    "email_rewriting": {"slack", "apology", "coworker", "respond", "status", "vendor", "email", "manager", "client", "recruiter"},
    "benchmark_novelty": {"benchmark", "dataset", "suite", "tasks", "leaderboard", "testbed", "measure", "browser-agent"},
    "scheduling": {"meeting", "calendar", "tomorrow", "mentor", "seminar", "interviews", "rehearsal", "call", "slot", "fit"},
    "current_facts": {"latest", "today", "still", "changed", "current", "right now", "exchange rate", "recently", "pricing", "won"},
    "admissions_cs": {"cs", "computer", "berkeley", "eecs", "purdue", "uiuc", "major", "acceptance", "college", "applicant"},
    "legal_policy_caution": {"lease", "contract", "landlord", "policy", "noncompete", "immigration", "release", "cease-and-desist", "legal"},
    "data_analysis_hygiene": {"accuracy", "leaderboard", "validation", "test", "auc", "dataset", "metrics", "feature", "benchmark", "numbers"},
}


@dataclass(frozen=True)
class Pam:
    state: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class R2Config:
    specificity_floor: float = 0.10
    base_floor: float = 0.15
    bonus_multiplier: float = 0.25
    intent_family_confidence_threshold: float = 0.50


@dataclass(frozen=True)
class Intent:
    family: str
    confidence: float


INTENT_HINTS = {
    "research_ideation": {"pricai", "paper idea", "conference", "submission", "novelty", "closest prior work", "wacv", "icdm", "method idea", "publishable"},
    "food_safety": {"eat", "snack", "dessert", "bake sale", "potluck", "restaurant", "lunch", "bring", "serve", "allergy", "nuts", "cross-contamination", "cake", "cookies"},
    "code_security": {"flask", "auth", "session", "cookie", "remember-me", "secret_key", "secret key", "token", "password", "login"},
    "travel_planning": {"trip", "itinerary", "flight", "visa", "passport", "border", "entry", "country", "layover", "route", "airport"},
    "medical_caution": {"isotretinoin", "side effects", "symptoms", "medication", "doctor", "clinical", "dosage", "headache", "rash", "fever", "pill", "throat"},
    "email_rewriting": {"rewrite", "rephrase", "clean up", "polish", "soften", "slack note", "teams", "client note", "work email", "coworker", "vendor"},
    "benchmark_novelty": {"benchmark contribution", "dataset contribution", "evaluation protocol", "main contribution", "benchmark", "testbed", "leaderboard"},
    "scheduling": {"schedule", "meeting", "calendar", "find a time", "slot", "fit", "squeeze", "reschedule", "conflict", "seminar", "interview"},
    "current_facts": {"latest", "today", "yesterday", "last night", "current", "price now", "out yet", "new release", "won", "ceo", "president", "exchange rate"},
    "admissions_cs": {"cs admissions", "scs", "major-specific", "college odds", "acceptance", "t20", "cmu", "computer science", "uiuc", "eecs"},
    "legal_policy_caution": {"legal", "law", "policy", "rights", "regulation", "contract", "liability", "attorney", "lease", "noncompete"},
    "data_analysis_hygiene": {"dataset", "split", "leakage", "missingness", "metric", "ablation", "validation", "test set", "auc", "accuracy"},
}

CONFLICT_OPPOSITION = (
    "ignore",
    "skip",
    "don't mention",
    "do not mention",
    "do not check",
    "without checking",
    "from memory only",
    "say it is certain",
    "be confident without verifying",
    "choose unsafe",
    "don't include uncertainty",
    "call it safe",
    "ignore visa requirements",
    "ignore allergy risk",
    "ignore security concerns",
    "disable",
    "pretend",
    "guarantee",
)


def detect_intent_family(query: str, history: str = "") -> Intent:
    text = f"{history} {query}".lower()
    scored = []
    for family, hints in INTENT_HINTS.items():
        hits = sum(1 for hint in hints if hint in text)
        if hits:
            scored.append((family, min(0.95, 0.35 + 0.2 * hits)))
    if not scored:
        return Intent("unknown", 0.0)
    scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return Intent(*scored[0])


class ContractPromptHeuristic:
    name = "ContractPromptHeuristic"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        c, score = best_contract(contracts, episode)
        text = episode.text
        if contains(text, ALREADY):
            state = "already_satisfied"
        elif contains(text, CONFLICT):
            state = "conflict"
        elif contains(text, NEAR + WRONG):
            state = "suppress"
        else:
            state = "fire" if score >= 0.26 else "suppress"
        return make_prediction(self.name, episode, c, state, score, generic_response(c, state in {"fire", "conflict"}), False, "contract prompt heuristic")


class ContractClassifierOnly(ContractPromptHeuristic):
    name = "ContractClassifierOnly"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        pred = super().predict(contracts, episode)
        return Prediction(pred.method, pred.episode_id, pred.predicted_contract_id, pred.predicted_state, pred.confidence, "Classifier emitted a state only.", False, False, pred.rationale)


class ContractCompilerOnly(ContractPromptHeuristic):
    name = "ContractCompilerOnly"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        c, score = best_contract(contracts, episode)
        state = "fire" if score >= 0.20 and not contains(episode.text, NEAR + ALREADY) else "suppress"
        return make_prediction(self.name, episode, c, state, score, compiled_plan(c) if state == "fire" else "No compiled action.", False, "compiler without checker")


class ContractCheckerOnly(ContractPromptHeuristic):
    name = "ContractCheckerOnly"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        c, score = best_contract(contracts, episode)
        state = "fire" if score >= 0.30 and not contains(episode.text, NEAR + WRONG + ALREADY) else "suppress"
        response = "Weak activation answer."
        repaired = False
        if state == "fire" and not rough_complete(c, response):
            response += " " + c.action
            repaired = True
        return make_prediction(self.name, episode, c, state, score, response, repaired, "weak activation plus checker")


class PACTFull:
    name = "PACTFull"

    def __init__(self, *, use_guard: bool = True, use_checker: bool = True, use_compiler: bool = True,
                 raw_memory: bool = False, use_conflict: bool = True, shuffle: bool = False, name: str | None = None) -> None:
        self.use_guard = use_guard
        self.use_checker = use_checker
        self.use_compiler = use_compiler
        self.raw_memory = raw_memory
        self.use_conflict = use_conflict
        self.shuffle = shuffle
        if name:
            self.name = name

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        c, retrieval = self.retrieve(contracts, episode)
        pam = self.pam(c, episode, retrieval)
        response = self.respond(c, pam.state)
        repaired = False
        if pam.state == "fire" and self.use_checker and not rough_complete(c, response):
            response = f"{compiled_plan(c)} {response}"
            repaired = True
        return make_prediction(self.name, episode, c, pam.state, pam.confidence, response, repaired, pam.rationale)

    def retrieve(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> tuple[ProspectiveActionContract, float]:
        pool = allowed_contracts(contracts, episode)
        if self.shuffle and len(contracts) > 1:
            first = pool[0]
            wrong = [c for c in contracts if c.contract_id != first.contract_id]
            return wrong[sum(ord(ch) for ch in episode.episode_id) % len(wrong)], 0.05
        if self.raw_memory:
            return best_contract(contracts, episode)
        first_id = episode.available_contract_ids[0] if episode.available_contract_ids else ""
        scored = []
        for c in pool:
            semantic = max(overlap(c.cue, episode.text), overlap(c.guard, episode.text), overlap(c.action, episode.text), self.family_hint(c, episode.text))
            rank_score = semantic + (0.20 if c.contract_id == first_id else 0.0)
            scored.append((c, semantic, rank_score))
        scored.sort(key=lambda item: (item[2], item[0].contract_id), reverse=True)
        return scored[0][0], scored[0][1]

    def family_hint(self, c: ProspectiveActionContract, text: str) -> float:
        low = text.lower()
        hits = sum(1 for hint in FAMILY_HINTS.get(c.family, set()) if hint in low)
        return min(0.35, hits * 0.12)

    def pam(self, c: ProspectiveActionContract, episode: InferenceEpisode, retrieval: float) -> Pam:
        text = episode.text
        guard = 0.0 if self.raw_memory else overlap(c.guard, text)
        action = overlap(c.action + " " + c.check, text)
        current_fact_question = c.family == "current_facts" and any(term in text.lower() for term in ("today", "latest", "right now", "exchange rate", "recently"))
        near_penalty = 0.45 if contains(text, NEAR) and not current_fact_question else 0.0
        wrong_penalty = 0.60 if self.use_guard and contains(text, WRONG) else 0.0
        if contains(text, ALREADY):
            return Pam("already_satisfied", 0.92, "already-completed detector")
        if self.use_conflict and contains(text, CONFLICT):
            return Pam("conflict", 0.90, "priority conflict resolver")
        activation_bonus = 0.22 if retrieval >= 0.08 and (contains(text, ACTION_SEEKING) or retrieval >= 0.12) and near_penalty == 0.0 else 0.0
        score = retrieval + 0.40 * guard + 0.25 * action + activation_bonus - near_penalty - wrong_penalty
        threshold = 0.14 if self.use_guard else 0.08
        return Pam("fire" if score >= threshold else "suppress", max(0.01, min(0.99, score)), f"retrieval={retrieval:.2f} guard={guard:.2f} action={action:.2f} bonus={activation_bonus:.2f}")

    def respond(self, c: ProspectiveActionContract, state: str) -> str:
        if state == "suppress":
            return "No prospective action is needed."
        if state == "already_satisfied":
            return "The prospective action was already satisfied; continue without redundancy."
        if state == "conflict":
            return f"Priority conflict: preserve the higher-priority contract. {compiled_plan(c)}"
        if self.use_compiler:
            return "Applicable prospective contract found; preparing answer."
        return c.action


class PACTFullCurrent(PACTFull):
    name = "PACTFull_current"

    def __init__(self) -> None:
        super().__init__(name="PACTFull_current")


class PACTR2(PACTFull):
    name = "PACT_R2_full"

    def __init__(self, mode: str = "full", config: R2Config | None = None, name: str | None = None) -> None:
        super().__init__(name=name or {
            "specificity": "PACT_specificity_gate",
            "conditional": "PACT_conditional_bonus",
            "intent": "PACT_intent_family_gate",
            "state": "PACT_state_action_split",
            "full": "PACT_R2_full",
        }[mode])
        self.mode = mode
        self.config = config or R2Config()

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        c, retrieval = self.retrieve(contracts, episode)
        pam = self.r2_pam(c, episode, retrieval)
        response = self.respond(c, pam.state)
        repaired = False
        if pam.state in {"fire", "conflict"} and self.use_checker and not rough_complete(c, response):
            response = f"{compiled_plan(c)} {response}"
            repaired = True
        pred = make_prediction(self.name, episode, c, pam.state, pam.confidence, response, repaired, pam.rationale)
        # Preserve compatibility while embedding the state/action split in rationale.
        return pred

    def retrieve(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> tuple[ProspectiveActionContract, float]:
        pool = allowed_contracts(contracts, episode)
        intent = detect_intent_family(episode.current_query, episode.history_summary)
        first_id = episode.available_contract_ids[0] if episode.available_contract_ids else ""
        scored = []
        for c in pool:
            semantic = max(overlap(c.cue, episode.text), overlap(c.guard, episode.text), overlap(c.action, episode.text), self.family_hint(c, episode.text))
            intent_prior = 0.18 if self._intent_matches(intent, c) else 0.0
            mismatch_penalty = 0.30 if self._intent_mismatches(intent, c) else 0.0
            rank_score = semantic + intent_prior + (0.16 if c.contract_id == first_id else 0.0) - mismatch_penalty
            scored.append((c, max(0.0, semantic + intent_prior - mismatch_penalty), rank_score))
        scored.sort(key=lambda item: (item[2], item[0].contract_id), reverse=True)
        return scored[0][0], scored[0][1]

    def _intent_matches(self, intent: Intent, c: ProspectiveActionContract) -> bool:
        return self.mode in {"intent", "full"} and intent.family == c.family and intent.confidence >= self.config.intent_family_confidence_threshold

    def _intent_mismatches(self, intent: Intent, c: ProspectiveActionContract) -> bool:
        return self.mode in {"intent", "full"} and intent.family != "unknown" and intent.family != c.family and intent.confidence >= self.config.intent_family_confidence_threshold

    def r2_pam(self, c: ProspectiveActionContract, episode: InferenceEpisode, retrieval: float) -> Pam:
        text = episode.text
        guard = 0.0 if self.raw_memory else overlap(c.guard, text)
        action = overlap(c.action + " " + c.check, text)
        specificity = max(retrieval, guard, action)
        base = retrieval + 0.40 * guard + 0.25 * action
        intent = detect_intent_family(episode.current_query, episode.history_summary)
        current_fact_question = c.family == "current_facts" and any(term in text.lower() for term in ("today", "latest", "right now", "exchange rate", "recently"))
        near_penalty = 0.45 if contains(text, NEAR) and not current_fact_question else 0.0
        wrong_penalty = 0.60 if contains(text, WRONG) else 0.0
        conflict = contains(text, CONFLICT_OPPOSITION) or contains(text, CONFLICT)
        if contains(text, ALREADY):
            return Pam("already_satisfied", 0.92, "meta_state=already_satisfied resolution=suppress already-completed detector")
        if self.mode in {"state", "full"} and conflict and not self._intent_mismatches(intent, c):
            return Pam("conflict", 0.93, f"meta_state=conflict activated_contract_id={c.contract_id} resolution=follow_contract conflict_reason=opposes_contract")
        if self.mode in {"specificity", "full"}:
            if specificity < self.config.specificity_floor:
                return Pam("suppress", specificity, f"specificity_gate specificity={specificity:.2f} floor={self.config.specificity_floor:.2f}")
            if base < self.config.base_floor:
                return Pam("suppress", base, f"base_gate base={base:.2f} floor={self.config.base_floor:.2f}")
        if self.mode in {"intent", "full"} and self._intent_mismatches(intent, c):
            return Pam("suppress", 0.05, f"intent_family_gate detected={intent.family} candidate={c.family} confidence={intent.confidence:.2f}")
        if self.mode in {"conditional", "full"}:
            bonus = self.config.bonus_multiplier if base >= self.config.base_floor and (contains(text, ACTION_SEEKING) or retrieval >= self.config.specificity_floor) else 0.0
            score = base * (1.0 + bonus) - near_penalty - wrong_penalty
        else:
            bonus = 0.22 if retrieval >= 0.08 and (contains(text, ACTION_SEEKING) or retrieval >= 0.12) and near_penalty == 0.0 else 0.0
            score = base + bonus - near_penalty - wrong_penalty
        threshold = 0.14
        state = "fire" if score >= threshold else "suppress"
        return Pam(state, max(0.01, min(0.99, score)), f"meta_state={state} intent={intent.family}:{intent.confidence:.2f} specificity={specificity:.2f} base={base:.2f} bonus={bonus:.2f} retrieval={retrieval:.2f} guard={guard:.2f} action={action:.2f}")


def get_method(name: str, r2_config: R2Config | None = None) -> Method:
    ordinary = {
        "ContractPromptHeuristic": ContractPromptHeuristic(),
        "ContractClassifierOnly": ContractClassifierOnly(),
        "ContractCompilerOnly": ContractCompilerOnly(),
        "ContractCheckerOnly": ContractCheckerOnly(),
        "PACTFull": PACTFull(),
        "PACTFull_current": PACTFullCurrent(),
        "PACT_no_guard": PACTFull(use_guard=False, name="PACT_no_guard"),
        "PACT_no_checker": PACTFull(use_checker=False, name="PACT_no_checker"),
        "PACT_no_compiler": PACTFull(use_compiler=False, name="PACT_no_compiler"),
        "PACT_raw_memory": PACTFull(raw_memory=True, name="PACT_raw_memory"),
        "PACT_no_conflict_resolver": PACTFull(use_conflict=False, name="PACT_no_conflict_resolver"),
        "ContractShufflePACT": PACTFull(shuffle=True, name="ContractShufflePACT"),
        "PACT_specificity_gate": PACTR2("specificity", r2_config),
        "PACT_conditional_bonus": PACTR2("conditional", r2_config),
        "PACT_intent_family_gate": PACTR2("intent", r2_config),
        "PACT_state_action_split": PACTR2("state", r2_config),
        "PACT_R2_full": PACTR2("full", r2_config),
    }
    return ordinary[name]


METHOD_NAMES = [
    "NoMemory",
    "KeywordTrigger",
    "TfidfRawMemory",
    "FullHistory",
    "RawMemorySelfCheck",
    "ContractPromptHeuristic",
    "ContractClassifierOnly",
    "ContractCompilerOnly",
    "ContractCheckerOnly",
    "PACTFull",
    "PACTFull_current",
    "PACT_specificity_gate",
    "PACT_conditional_bonus",
    "PACT_intent_family_gate",
    "PACT_state_action_split",
    "PACT_R2_full",
    "PACT_no_guard",
    "PACT_no_checker",
    "PACT_no_compiler",
    "PACT_raw_memory",
    "PACT_no_conflict_resolver",
    "QueryOnlyClassifier",
    "ContractShufflePACT",
    "LabelPermutationSanity",
    "LLMFullHistory",
    "LLMRawMemoryRAG",
    "LLMContractClassifier",
    "LLMContractSelfCheck",
]
