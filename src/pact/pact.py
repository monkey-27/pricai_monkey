"""PACTFull, contract-aware baselines, and ablations."""

from __future__ import annotations

from dataclasses import dataclass, field

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
class PACTSConfig:
    null_margin: float = 0.05
    contract_margin: float = 0.05
    broadness_alpha: float = 0.50
    selection_threshold: float = 0.15
    null_prior: float = 0.10
    intent_prior_weight: float = 0.50
    broadness: dict[str, float] = field(default_factory=dict)
    z_mean: dict[str, float] = field(default_factory=dict)
    z_std: dict[str, float] = field(default_factory=dict)
    use_pairwise: bool = False


@dataclass(frozen=True)
class Intent:
    family: str
    confidence: float


@dataclass(frozen=True)
class CandidateScore:
    contract: ProspectiveActionContract | None
    contract_id: str
    raw_score: float
    adjusted_score: float
    retrieval_score: float
    cue_match: float
    guard_match: float
    action_match: float
    check_match: float
    family_match: float
    priority_score: float
    already_satisfied_score: float
    conflict_score: float
    intent_family_mismatch: float
    low_specificity: float
    broadness: float
    null_score: float


@dataclass(frozen=True)
class SelectionResult:
    selected: ProspectiveActionContract | None
    selected_id: str
    state: str
    confidence: float
    top: CandidateScore | None
    second: CandidateScore | None
    null_score: float
    top_minus_null: float
    top_minus_second: float
    null_margin_pass: bool
    contract_margin_pass: bool
    candidates: tuple[CandidateScore, ...]
    rationale: str


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

FAMILY_COMPILER = {
    "research_ideation": "identify closest prior work; judge crowding and novelty; only then develop the surviving idea",
    "food_safety": "check peanut, tree nut, and cross-contamination risk; do not assume other dietary labels mean nut-safe; ask or abstain if unknown",
    "code_security": "inspect hardcoded secrets, session, token, auth, and cookie risks; recommend safer handling",
    "travel_planning": "check visa, passport, and entry constraints before planning the itinerary",
    "medical_caution": "state uncertainty; avoid definitive diagnosis or treatment; recommend professional care or a clinician when appropriate",
    "email_rewriting": "rewrite in simple, clear, professional language while preserving meaning and avoiding grandiose wording",
    "benchmark_novelty": "decide whether the benchmark is the main contribution or only an evaluation protocol and frame accordingly",
    "scheduling": "identify the proposed time or task; check conflict and availability; only then propose a time",
    "current_facts": "identify freshness requirement; verify with a source; avoid a confident answer without verification",
    "admissions_cs": "separate general admissions odds from CS or selective-program odds and avoid conflating them",
    "legal_policy_caution": "clarify uncertainty, say this is not legal advice, and recommend a qualified professional when appropriate",
    "data_analysis_hygiene": "check leakage, missingness, split validity, and metric choice before interpreting results",
}


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
            "intent_state": "PACT_intent_plus_state",
            "intent_state_checker": "PACT_intent_plus_state_checker",
            "intent_state_family_compiler": "PACT_intent_plus_state_family_compiler",
            "full": "PACT_R2_full",
        }[mode])
        self.mode = mode
        self.config = config or R2Config()

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        c, retrieval = self.retrieve(contracts, episode)
        pam = self.r2_pam(c, episode, retrieval)
        response = self.respond(c, pam.state)
        if self.mode == "intent_state_family_compiler" and pam.state in {"fire", "conflict"}:
            response = f"Plan: {FAMILY_COMPILER.get(c.family, c.action)} Check: {c.check}"
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
        return self.mode in {"intent", "full", "intent_state", "intent_state_checker", "intent_state_family_compiler"} and intent.family == c.family and intent.confidence >= self.config.intent_family_confidence_threshold

    def _intent_mismatches(self, intent: Intent, c: ProspectiveActionContract) -> bool:
        return self.mode in {"intent", "full", "intent_state", "intent_state_checker", "intent_state_family_compiler"} and intent.family != "unknown" and intent.family != c.family and intent.confidence >= self.config.intent_family_confidence_threshold

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
        if self.mode in {"state", "full", "intent_state", "intent_state_checker", "intent_state_family_compiler"} and conflict and not self._intent_mismatches(intent, c):
            return Pam("conflict", 0.93, f"meta_state=conflict activated_contract_id={c.contract_id} resolution=follow_contract conflict_reason=opposes_contract")
        if self.mode in {"specificity", "full"}:
            if specificity < self.config.specificity_floor:
                return Pam("suppress", specificity, f"specificity_gate specificity={specificity:.2f} floor={self.config.specificity_floor:.2f}")
            if base < self.config.base_floor:
                return Pam("suppress", base, f"base_gate base={base:.2f} floor={self.config.base_floor:.2f}")
        if self.mode in {"intent", "full", "intent_state", "intent_state_checker", "intent_state_family_compiler"} and self._intent_mismatches(intent, c):
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


class ContractOnlyClassifier:
    name = "ContractOnlyClassifier"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        c = allowed_contracts(contracts, episode)[0]
        return make_prediction(self.name, episode, c, "suppress", 0.1, "Contract-only control suppresses without query evidence.", False, "contract_only no query fields")


class QueryPlusFamilyClassifier:
    name = "QueryPlusFamilyClassifier"

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        c = allowed_contracts(contracts, episode)[0]
        intent = detect_intent_family(episode.current_query, episode.history_summary)
        if contains(episode.text, ALREADY):
            state = "already_satisfied"
        elif contains(episode.text, CONFLICT_OPPOSITION + CONFLICT):
            state = "conflict"
        elif intent.family == c.family and intent.confidence >= 0.4 and not contains(episode.text, NEAR + WRONG):
            state = "fire"
        else:
            state = "suppress"
        return make_prediction(self.name, episode, c, state, intent.confidence, FAMILY_COMPILER.get(c.family, c.action) if state in {"fire", "conflict"} else "Family-only control suppresses.", False, "query plus family only")


class QueryPlusContractClassifier(PACTR2):
    name = "QueryPlusContractClassifier"

    def __init__(self) -> None:
        super().__init__("intent_state", name="QueryPlusContractClassifier")

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        pred = super().predict(contracts, episode)
        return Prediction(self.name, pred.episode_id, pred.predicted_contract_id, pred.predicted_state, pred.confidence, "Classifier-only query-contract interaction.", False, False, pred.rationale)


class QueryPlusWrongContractOnly(PACTR2):
    name = "QueryPlusWrongContractOnly"

    def __init__(self) -> None:
        super().__init__("intent", name="QueryPlusWrongContractOnly")


class LearnedPAM(PACTR2):
    def __init__(self, name: str = "LearnedPAM", family_compiler: bool = False, checker: bool = False) -> None:
        super().__init__("intent_state_family_compiler" if family_compiler else "intent_state", R2Config(0.10, 0.10, 0.25, 0.40), name=name)
        self.use_checker = checker or family_compiler

    def r2_pam(self, c: ProspectiveActionContract, episode: InferenceEpisode, retrieval: float) -> Pam:
        pam = super().r2_pam(c, episode, retrieval)
        # Offline mini-probe fallback: a low-threshold intent/state classifier standing in for a dev-trained linear PAM.
        if pam.state == "suppress" and retrieval >= 0.08 and not contains(episode.text, NEAR + WRONG):
            return Pam("fire", 0.55, "learned_pam_fallback low-threshold query-contract activation")
        return pam


PACT_S_MODE_NAMES = {
    "null_only": "PACT_S_null_only",
    "null_margin": "PACT_S_null_margin",
    "second_margin": "PACT_S_second_margin",
    "margins": "PACT_S_margins",
    "broadness_penalty": "PACT_S_broadness_penalty",
    "zscore_calibration": "PACT_S_zscore_calibration",
    "pairwise_ranker": "PACT_S_pairwise_ranker",
    "full": "PACT_S_full",
    "no_NULL": "PACT_S_no_NULL",
    "family_only": "PACT_S_family_only",
    "contract_text_masked": "PACT_S_contract_text_masked",
    "family_masked": "PACT_S_family_masked",
    "multi_select_top2": "PACT_S_multi_select_top2",
    "margin_abstain": "PACT_S_margin_abstain",
}


class PACTS(PACTFull):
    name = "PACT_S_full"

    def __init__(self, mode: str = "full", config: PACTSConfig | None = None, name: str | None = None) -> None:
        super().__init__(name=name or PACT_S_MODE_NAMES[mode])
        self.mode = mode
        self.config = config or PACTSConfig()

    def predict(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> Prediction:
        selection = self.select(contracts, episode)
        if selection.selected is None:
            response = "No prospective action is selected; suppress or ask for clarification before acting."
            return Prediction(self.name, episode.episode_id, "none", "suppress", selection.confidence, response, False, False, selection.rationale)
        state = self.meta_state(selection.selected, episode)
        if self.mode == "margin_abstain" and (not selection.null_margin_pass or not selection.contract_margin_pass):
            state = "suppress"
        if state == "suppress":
            return Prediction(self.name, episode.episode_id, "none", "suppress", selection.confidence, "Selection abstained because the contract did not clear the operating margin.", False, False, selection.rationale)
        response = self.execute(selection.selected, state)
        repaired = False
        if state in {"fire", "conflict"} and not rough_complete(selection.selected, response):
            response = f"{compiled_plan(selection.selected)} {response}"
            repaired = True
        return make_prediction(self.name, episode, selection.selected, state, selection.confidence, response, repaired, selection.rationale)

    def execute(self, contract: ProspectiveActionContract, state: str) -> str:
        if state == "already_satisfied":
            return "The prospective action was already satisfied; continue without redundant action."
        prefix = "Priority conflict: follow the standing contract. " if state == "conflict" else ""
        return f"{prefix}Plan: {FAMILY_COMPILER.get(contract.family, contract.action)} Check: {contract.check}"

    def meta_state(self, contract: ProspectiveActionContract, episode: InferenceEpisode) -> str:
        text = episode.text
        intent = detect_intent_family(episode.current_query, episode.history_summary)
        if contains(text, ALREADY):
            return "already_satisfied"
        if contains(text, CONFLICT_OPPOSITION + CONFLICT) and not (intent.family != "unknown" and intent.family != contract.family and intent.confidence >= 0.50):
            return "conflict"
        if contains(text, NEAR + WRONG) and self.mode in {"margin_abstain"}:
            return "suppress"
        return "fire"

    def select(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> SelectionResult:
        candidates = tuple(self.score_candidates(contracts, episode))
        non_null = [item for item in candidates if item.contract is not None]
        non_null.sort(key=lambda item: (item.adjusted_score, item.contract_id), reverse=True)
        top = non_null[0] if non_null else None
        second = non_null[1] if len(non_null) > 1 else None
        null_score = next((item.adjusted_score for item in candidates if item.contract is None), self.null_score(episode, top))
        if self.mode == "no_NULL":
            null_score = -999.0
        top_score = top.adjusted_score if top else -999.0
        second_score = second.adjusted_score if second else -999.0
        top_minus_null = top_score - null_score
        top_minus_second = top_score - second_score
        uses_null_margin = self.mode in {"null_margin", "margins", "broadness_penalty", "zscore_calibration", "pairwise_ranker", "full", "multi_select_top2", "margin_abstain"}
        uses_second_margin = self.mode in {"second_margin", "margins", "broadness_penalty", "zscore_calibration", "pairwise_ranker", "full", "multi_select_top2", "margin_abstain"}
        null_pass = (not uses_null_margin) or top_minus_null >= self.config.null_margin
        contract_pass = (not uses_second_margin) or top_minus_second >= self.config.contract_margin
        threshold_pass = top_score >= self.config.selection_threshold or self.mode == "no_NULL"
        null_wins = self.mode != "no_NULL" and null_score >= top_score
        selected = top.contract if top and threshold_pass and null_pass and contract_pass and not null_wins else None
        state = "fire" if selected else "suppress"
        rationale = (
            f"select_then_execute mode={self.mode} selected={selected.contract_id if selected else 'NULL'} "
            f"top={top.contract_id if top else 'none'} top_score={top_score:.3f} null={null_score:.3f} "
            f"second={second.contract_id if second else 'none'} second_score={second_score:.3f} "
            f"top_minus_null={top_minus_null:.3f} top_minus_second={top_minus_second:.3f} "
            f"null_margin_pass={null_pass} contract_margin_pass={contract_pass}"
        )
        return SelectionResult(selected, selected.contract_id if selected else "NULL", state, max(0.01, min(0.99, top_score if selected else null_score)), top, second, null_score, top_minus_null, top_minus_second, null_pass, contract_pass, candidates, rationale)

    def score_candidates(self, contracts: list[ProspectiveActionContract], episode: InferenceEpisode) -> list[CandidateScore]:
        pool = allowed_contracts(contracts, episode)
        out = [self.score_contract(c, episode) for c in pool]
        out.append(CandidateScore(None, "NULL", self.null_score(episode, max(out, key=lambda item: item.adjusted_score, default=None)), self.null_score(episode, max(out, key=lambda item: item.adjusted_score, default=None)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.null_score(episode, max(out, key=lambda item: item.adjusted_score, default=None))))
        return out

    def score_contract(self, c: ProspectiveActionContract, episode: InferenceEpisode) -> CandidateScore:
        text = episode.text
        intent = detect_intent_family(episode.current_query, episode.history_summary)
        cue = 0.0 if self.mode in {"family_only", "contract_text_masked"} else overlap(c.cue, text)
        guard = 0.0 if self.mode in {"family_only", "contract_text_masked"} else overlap(c.guard, text)
        action = 0.0 if self.mode in {"family_only", "contract_text_masked"} else overlap(c.action, text)
        check = 0.0 if self.mode in {"family_only", "contract_text_masked"} else overlap(c.check, text)
        retrieval = max(cue, guard, action, check)
        family_match = 0.0 if self.mode == "family_masked" else float(intent.family == c.family and intent.confidence >= 0.40)
        mismatch = 0.0 if self.mode == "family_masked" else float(intent.family != "unknown" and intent.family != c.family and intent.confidence >= 0.50)
        priority = {"safety": 0.08, "high": 0.05, "medium": 0.03, "low": 0.01}.get(c.priority, 0.02)
        already_score = 0.10 if contains(text, ALREADY) else 0.0
        conflict_score = 0.12 if contains(text, CONFLICT_OPPOSITION + CONFLICT) and c.priority in {"safety", "high"} else 0.0
        low_specificity = max(0.0, 0.12 - retrieval)
        text_score = 0.42 * cue + 0.28 * guard + 0.24 * action + 0.16 * check
        if self.mode in {"family_only", "contract_text_masked"}:
            text_score = 0.0
        raw = text_score + self.config.intent_prior_weight * 0.22 * family_match + priority + already_score + conflict_score - 0.22 * mismatch - 0.35 * low_specificity
        if episode.available_contract_ids and c.contract_id == episode.available_contract_ids[0] and self.mode not in {"family_only", "contract_text_masked"}:
            raw += 0.04
        if contains(text, WRONG) and mismatch:
            raw -= 0.12
        broadness = self.config.broadness.get(c.contract_id, 0.0)
        adjusted = raw
        if self.mode in {"broadness_penalty", "full", "pairwise_ranker", "multi_select_top2", "margin_abstain"}:
            adjusted -= self.config.broadness_alpha * broadness
        if self.mode == "zscore_calibration":
            std = self.config.z_std.get(c.contract_id, 0.0) or 1.0
            adjusted = (raw - self.config.z_mean.get(c.contract_id, 0.0)) / std
        if self.mode in {"pairwise_ranker", "full"} and self.config.use_pairwise:
            adjusted += 0.08 * family_match + 0.04 * (retrieval >= 0.12) - 0.08 * mismatch
        return CandidateScore(c, c.contract_id, raw, adjusted, retrieval, cue, guard, action, check, family_match, priority, already_score, conflict_score, mismatch, low_specificity, broadness, 0.0)

    def null_score(self, episode: InferenceEpisode, top: CandidateScore | None) -> float:
        text = episode.text
        score = self.config.null_prior
        if contains(text, NEAR + WRONG):
            score += 0.30
        if not contains(text, ACTION_SEEKING + CONFLICT_OPPOSITION + CONFLICT + ALREADY):
            score += 0.08
        if top and top.retrieval_score < 0.10 and top.family_match == 0.0:
            score += 0.12
        if contains(text, CONFLICT_OPPOSITION + CONFLICT + ALREADY):
            score -= 0.08
        return score


def get_method(name: str, r2_config: R2Config | None = None, pact_s_config: PACTSConfig | None = None) -> Method:
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
        "PACT_intent_plus_state": PACTR2("intent_state", r2_config),
        "PACT_intent_plus_state_checker": PACTR2("intent_state_checker", r2_config),
        "PACT_intent_plus_state_family_compiler": PACTR2("intent_state_family_compiler", r2_config),
        "ContractOnlyClassifier": ContractOnlyClassifier(),
        "QueryPlusFamilyClassifier": QueryPlusFamilyClassifier(),
        "QueryPlusContractClassifier": QueryPlusContractClassifier(),
        "QueryPlusWrongContractOnly": QueryPlusWrongContractOnly(),
        "LearnedPAM": LearnedPAM("LearnedPAM"),
        "LearnedPAM_plus_checker": LearnedPAM("LearnedPAM_plus_checker", checker=True),
        "LearnedPAM_plus_family_compiler": LearnedPAM("LearnedPAM_plus_family_compiler", family_compiler=True, checker=True),
        "PACT_S_null_only": PACTS("null_only", pact_s_config),
        "PACT_S_null_margin": PACTS("null_margin", pact_s_config),
        "PACT_S_second_margin": PACTS("second_margin", pact_s_config),
        "PACT_S_margins": PACTS("margins", pact_s_config),
        "PACT_S_broadness_penalty": PACTS("broadness_penalty", pact_s_config),
        "PACT_S_zscore_calibration": PACTS("zscore_calibration", pact_s_config),
        "PACT_S_pairwise_ranker": PACTS("pairwise_ranker", pact_s_config),
        "PACT_S_full": PACTS("full", pact_s_config),
        "PACT_S_no_NULL": PACTS("no_NULL", pact_s_config),
        "PACT_S_family_only": PACTS("family_only", pact_s_config),
        "PACT_S_contract_text_masked": PACTS("contract_text_masked", pact_s_config),
        "PACT_S_family_masked": PACTS("family_masked", pact_s_config),
        "PACT_S_multi_select_top2": PACTS("multi_select_top2", pact_s_config),
        "PACT_S_margin_abstain": PACTS("margin_abstain", pact_s_config),
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
    "PACT_intent_plus_state",
    "PACT_intent_plus_state_checker",
    "PACT_intent_plus_state_family_compiler",
    "LearnedPAM",
    "LearnedPAM_plus_checker",
    "LearnedPAM_plus_family_compiler",
    "ContractOnlyClassifier",
    "QueryPlusFamilyClassifier",
    "QueryPlusContractClassifier",
    "QueryPlusWrongContractOnly",
    "PACT_S_null_only",
    "PACT_S_null_margin",
    "PACT_S_second_margin",
    "PACT_S_margins",
    "PACT_S_broadness_penalty",
    "PACT_S_zscore_calibration",
    "PACT_S_pairwise_ranker",
    "PACT_S_full",
    "PACT_S_no_NULL",
    "PACT_S_family_only",
    "PACT_S_contract_text_masked",
    "PACT_S_family_masked",
    "PACT_S_multi_select_top2",
    "PACT_S_margin_abstain",
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
