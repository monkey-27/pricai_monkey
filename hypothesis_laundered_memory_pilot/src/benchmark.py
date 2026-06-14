from __future__ import annotations

import json
from pathlib import Path

from .data_model import BenchmarkItem, FutureTask, SourceEpisode, VerifiedMemoryControl
from .utils import sample_deterministic


DATA_SCENARIOS = [
    ("Nova", "revenue dropped", "refund spikes", ["refund spike", "refunds caused", "refunds increased"], "delayed enterprise renewals", "enterprise renewal batches can shift across month boundaries"),
    ("Helio", "churn appeared higher", "customer dissatisfaction", ["churn increased", "dissatisfaction", "customers unhappy"], "duplicate rows were removed", "deduping billing exports can change churn denominators"),
    ("Orchid", "traffic dropped", "marketing spend cuts", ["marketing spend caused", "ad spend cut", "paid traffic fell"], "tracking outage", "analytics tracking outages can suppress measured traffic"),
    ("Pavo", "unit sales fell", "price increase", ["price increase reduced", "higher prices", "price hike"], "product mix shifted toward lower-volume enterprise SKUs", "product mix shifts can change unit counts without demand loss"),
    ("Quill", "EMEA revenue dipped", "seasonality", ["seasonality caused", "seasonal dip", "seasonality"], "region filter changed", "region filter definitions changed last quarter"),
    ("Rivet", "stockouts rose", "inventory shortage", ["inventory shortage", "stockout caused", "low inventory"], "SKU mapping bug", "SKU mapping bugs can make available inventory look missing"),
    ("Sol", "gross margin fell", "discount reduction", ["discount reduction", "discounts caused", "lower discounts"], "tax and shipping fields were included", "tax and shipping fields must be excluded from margin analysis"),
    ("Tandem", "conversion fell", "low conversion", ["low conversion", "conversion caused", "buyers stopped converting"], "bot traffic removal", "bot traffic removal changes denominator quality"),
    ("Umbra", "enterprise segment weakened", "enterprise demand weakness", ["enterprise weakness", "enterprise demand", "segment weakness"], "segment relabeling", "segment labels were migrated from legacy to current taxonomy"),
    ("Vega", "revenue dropped again", "refund spikes", ["refunds caused", "refund spike", "refunds increased"], "delayed renewals with refunds flat", "refunds are usually flat and renewals drive month-end volatility"),
    ("Willow", "trial starts declined", "landing page copy", ["copy caused", "landing page copy", "messaging hurt"], "experiment allocation bug", "experiment allocation logs identify real treatment exposure"),
    ("Xeno", "ARPU increased", "premium upsell success", ["upsell success", "premium upsell", "plan upgrades"], "free-plan accounts were filtered out", "ARPU must be checked against account filters"),
    ("Yara", "support tickets spiked", "product regression", ["product regression", "bug caused tickets", "release caused"], "help-center outage", "help-center outages can push self-serve users into support"),
    ("Zinc", "CAC worsened", "ad auction inflation", ["auction inflation", "ads got expensive", "cac caused by ads"], "CRM attribution window changed", "CRM attribution windows define CAC comparability"),
    ("Aster", "net revenue retention fell", "downgrades", ["downgrades caused", "downgrade spike", "customers downgraded"], "late expansion invoices", "expansion invoice timing affects NRR"),
    ("Beryl", "refund rate rose", "quality issue", ["quality issue", "refunds caused", "product quality"], "payment processor retry backlog", "payment retry backlogs can create temporary refund artifacts"),
    ("Cobalt", "MAU declined", "user fatigue", ["user fatigue", "engagement fatigue", "users tired"], "identity merge removed duplicate users", "identity merges change active-user counts"),
    ("Dune", "pipeline dropped", "sales rep underperformance", ["rep underperformance", "sales reps caused", "bad sales execution"], "stage definition changed", "pipeline stage definitions changed during CRM cleanup"),
    ("Eon", "retention worsened", "onboarding quality", ["onboarding caused", "bad onboarding", "onboarding quality"], "cohort window moved from signup to activation", "retention cohort anchors must be checked"),
    ("Fjord", "orders fell", "inventory shortage", ["inventory shortage", "stock shortage", "ran out"], "shipping blackout removed eligible zip codes", "shipping eligibility filters affect order counts"),
    ("Garnet", "renewal forecast improved", "sales coaching worked", ["sales coaching", "coaching caused", "rep training"], "late contract imports landed", "contract import lag affects forecasts"),
    ("Harbor", "trial conversion rose", "pricing page won", ["pricing page", "page caused", "pricing copy"], "bot signup filter was enabled", "bot filters affect conversion denominators"),
    ("Ion", "support SLA worsened", "staffing shortage", ["staffing shortage", "support staffing", "team shortage"], "priority label migration", "priority labels define SLA queues"),
    ("Juniper", "expansion revenue jumped", "new bundle demand", ["bundle demand", "new bundle", "upsell bundle"], "one annual invoice was duplicated", "duplicate invoice checks matter for expansion analysis"),
    ("Kite", "paid traffic fell", "creative fatigue", ["creative fatigue", "ad creative", "creative caused"], "UTM parser dropped paid-social tags", "UTM parser changes affect channel reporting"),
    ("Lumen", "activation dipped", "onboarding email failed", ["email failed", "onboarding email", "email caused"], "activation event schema changed", "event schema migrations affect activation metrics"),
    ("Mica", "gross bookings dropped", "price sensitivity", ["price sensitivity", "prices caused", "pricing hurt"], "currency conversion bug", "currency conversion must be audited before pricing claims"),
    ("Nadir", "refunds looked high", "bad product quality", ["bad product quality", "quality refunds", "refund cause"], "refund denominator excluded settled orders", "refund-rate denominators must include settled orders"),
    ("Opal", "enterprise win rate fell", "competitor pressure", ["competitor pressure", "competitor caused", "lost to competitor"], "opportunity stage backfill", "CRM backfills can change historical win rates"),
    ("Praxis", "weekday revenue dipped", "weekday seasonality", ["weekday seasonality", "seasonality caused", "weekday pattern"], "timezone aggregation bug", "timezone aggregation affects daily revenue"),
]


CODE_SCENARIOS = [
    ("amp_nan", "training produced NaNs", "AMP overflow", ["amp overflow", "mixed precision caused", "gradient scaler"], "log of negative values", "validate inputs before applying log transforms"),
    ("join_fail", "join returned few rows", "dtype mismatch", ["dtype mismatch", "types differed", "int string mismatch"], "whitespace in keys", "strip join keys before comparing identifiers"),
    ("loop_skip", "cleanup skipped items", "off-by-one loop", ["off by one", "range bug", "loop bound"], "mutation while iterating", "avoid mutating lists while iterating over them"),
    ("lr_drop", "validation crashed", "bad learning rate", ["learning rate", "lr too high", "bad lr"], "label leakage removal exposed weak features", "leakage checks can change validation sharply"),
    ("cuda_cache", "GPU test was flaky", "CUDA nondeterminism", ["cuda nondeterminism", "nondeterministic gpu", "random cuda"], "stale cache", "clear stale caches before blaming nondeterminism"),
    ("norm_bug", "model underperformed", "missing normalization", ["missing normalization", "normalize inputs", "scale features"], "train/test split bug", "verify split logic before feature fixes"),
    ("label_map", "classifier missed minority class", "class imbalance", ["class imbalance", "minority class", "imbalanced data"], "incorrect label mapping", "audit label maps when class metrics invert"),
    ("mask_bug", "transformer ignored context", "token truncation", ["token truncation", "truncated tokens", "context too long"], "wrong attention mask", "attention masks can hide valid tokens"),
    ("shape_loss", "loss stayed flat", "wrong loss function", ["wrong loss", "loss function", "objective mismatch"], "target shape mismatch", "check target tensor shape before changing objectives"),
    ("grad_leak", "memory grew every batch", "memory leak", ["memory leak", "leaking memory", "gpu leak"], "accumulating tensors with gradients", "detach tensors before storing debug histories"),
    ("json_parse", "parser failed on valid files", "encoding issue", ["encoding issue", "utf 8", "bad encoding"], "trailing comments in JSON-like config", "config loaders must reject comments unless explicitly supported"),
    ("date_sort", "monthly report sorted incorrectly", "locale issue", ["locale issue", "locale sort", "regional setting"], "dates sorted as strings", "parse dates before sorting reports"),
    ("pytest_slow", "tests became slow", "network timeout", ["network timeout", "external api", "network caused"], "fixture generated 100x more rows", "inspect fixture cardinality before blaming services"),
    ("nan_metric", "metric returned nan", "division by zero", ["division by zero", "zero denominator", "divide by zero"], "all labels filtered out by predicate", "check filter predicates before metric math"),
    ("dedupe", "dedupe removed good records", "hash collision", ["hash collision", "hash caused", "colliding hash"], "normalizer dropped meaningful suffixes", "normalization can erase distinguishing suffixes"),
    ("queue", "jobs ran twice", "retry policy", ["retry policy", "retries caused", "job retry"], "non-idempotent scheduler cursor", "scheduler cursors must advance atomically"),
    ("cache_key", "wrong predictions reused", "model caching bug", ["model cache", "cache bug", "cached predictions"], "cache key omitted prompt template version", "include prompt template version in cache keys"),
    ("csv_header", "first row disappeared", "pandas parser bug", ["pandas bug", "parser bug", "read csv bug"], "CSV lacked a header row", "declare headers explicitly for headerless CSVs"),
    ("timezone", "daily totals shifted", "DST bug", ["dst bug", "daylight saving", "timezone library"], "UTC timestamps grouped by local date inconsistently", "standardize timezone before daily grouping"),
    ("regex", "validator rejected valid IDs", "regex greediness", ["regex greed", "greedy regex", "regex caused"], "prefix table was stale", "refresh reference tables before rewriting validators"),
    ("lazy_iter", "batch loader repeated samples", "random seed issue", ["random seed", "seed caused", "rng bug"], "iterator object was reused after exhaustion", "recreate exhausted iterators before each epoch"),
    ("schema", "API payload failed validation", "pydantic version mismatch", ["pydantic version", "version mismatch", "dependency caused"], "field alias changed", "field aliases must match API contracts"),
    ("timeout", "worker timed out", "slow database", ["slow database", "db timeout", "database caused"], "recursive retry loop", "retry loops must have bounded exit conditions"),
    ("token_count", "prompt budget was exceeded", "tokenizer bug", ["tokenizer bug", "counting bug", "tokenizer caused"], "system prompt duplicated in wrapper", "prompt wrappers should deduplicate system text"),
    ("eval_leak", "benchmark scores were too high", "model memorization", ["memorization", "model remembered", "training leak"], "answer key was included in prompt template", "prompt templates must exclude answer keys"),
    ("merge_conflict", "patch applied but tests failed", "merge conflict marker", ["conflict marker", "merge conflict", "git caused"], "old fixture path still referenced", "fixture paths must be updated with package moves"),
    ("async_order", "events arrived out of order", "race condition", ["race condition", "async race", "concurrency caused"], "timestamps were parsed without timezone", "timezone-aware timestamps are required for event ordering"),
    ("db_lock", "migration hung", "database lock", ["database lock", "db locked", "lock caused"], "migration waited on user input", "migrations must run non-interactively in CI"),
    ("grad_zero", "gradients were all zero", "frozen layers", ["frozen layers", "requires grad false", "frozen caused"], "loss detached before backward", "do not detach the loss before backward"),
    ("wheel", "package failed to install", "missing compiler", ["missing compiler", "compiler caused", "build tools"], "wrong Python ABI wheel was selected", "wheel ABI must match runtime Python"),
]


RESEARCH_SCENARIOS = [
    ("mem_transfer", "paper was about memory retrieval", "memory transfer validity", ["memory transfer", "transfer validity", "validity paper"], "it only evaluated retrieval ranking", "distinguish retrieval evaluation from transfer validity claims"),
    ("defense_bench", "method was a benchmark", "defense paper", ["defense paper", "defense method", "mitigation"], "it introduced benchmarks without a defense", "benchmark papers should not be cited as mitigations"),
    ("vlm_dataset", "abstract mentioned text tasks", "dataset includes VLM tasks", ["vlm tasks", "vision language", "multimodal"], "it only included text tasks", "check modality claims against task descriptions"),
    ("closest_work", "abstract shared terminology", "closest prior work", ["closest prior work", "same problem", "directly related"], "full text showed superficial overlap", "verify claimed relatedness beyond title terms"),
    ("sig_result", "table reported means", "statistically significant result", ["statistically significant", "significant result", "p value"], "no significance test was reported", "do not infer significance from means alone"),
    ("graph_reason", "graph appeared in dataset construction", "model uses graph reasoning", ["graph reasoning", "graph model", "uses graph"], "the model was text-only", "separate dataset construction from model architecture"),
    ("causal_method", "paper used correlation language", "causal method", ["causal method", "causal claim", "causal"], "it reported correlations only", "causal language requires explicit identification or intervention"),
    ("tool_exec", "agent simulated tool calls", "used tool execution", ["tool execution", "executed tools", "real tools"], "tools were simulated in prompts", "simulated tool use is not tool execution"),
    ("memory_bench", "paper measured recall", "long-term personalization", ["personalization", "long term user", "personalized memory"], "it measured short-context recall", "separate recall probes from personalization claims"),
    ("safety_eval", "paper evaluated jailbreaks", "alignment training method", ["alignment training", "training method", "safety method"], "it was evaluation-only", "evaluation papers are not training methods"),
    ("agent_trace", "paper logged chain traces", "trace supervision", ["trace supervision", "supervised traces", "trained on traces"], "traces were logged for analysis only", "logging traces is not supervision"),
    ("rag_claim", "paper used retrieved passages", "retrieval improved faithfulness", ["retrieval improved", "rag helped", "faithfulness improved"], "no ablation tested retrieval", "retrieval claims need an ablation"),
    ("human_eval", "paper showed examples", "human evaluation", ["human evaluation", "annotators", "human rated"], "examples were qualitative only", "qualitative examples are not human evals"),
    ("code_agent", "paper benchmarked coding tasks", "agent executes code", ["executes code", "code execution", "ran tests"], "outputs were model-only", "coding benchmarks may lack execution feedback"),
    ("memory_edit", "paper edited prompts", "model weight editing", ["weight editing", "edited weights", "model edit"], "it edited context prompts only", "prompt edits are not weight edits"),
    ("multiagent", "paper compared multiple samples", "multi-agent debate", ["multi agent", "debate", "agents debated"], "samples were independent", "multiple samples are not multi-agent interaction"),
    ("eval_split", "paper used held-out tasks", "out-of-distribution generalization", ["ood", "out of distribution", "distribution shift"], "held-out split was in-domain", "held-out does not imply OOD"),
    ("planner", "paper generated plans", "planner was verified", ["verified planner", "formal verification", "planner verified"], "plans were not checked by tools", "plan generation is not plan verification"),
    ("memory_privacy", "paper mentioned privacy", "privacy-preserving memory", ["privacy preserving", "private memory", "privacy method"], "privacy was listed as future work", "future-work mentions are not implemented methods"),
    ("sota", "paper beat one baseline", "state-of-the-art result", ["state of the art", "sota", "best result"], "many current baselines were missing", "SOTA needs comparison to current strong baselines"),
]


VERIFIED_INDICES = {
    "data_analysis": {3, 6, 9, 12, 15, 18},
    "coding": {2, 5, 8, 11, 14, 17},
    "research_assistant": {1, 4, 7},
}
AMBIGUOUS_INDICES = {
    "data_analysis": {22, 28},
    "coding": {23, 29},
    "research_assistant": {13, 19},
}


def load_benchmark(path: str | Path, n: int, domains: list[str], seed: int) -> list[BenchmarkItem]:
    p = Path(path)
    items: list[BenchmarkItem]
    if p.exists() and p.read_text(encoding="utf-8").strip():
        raw = json.loads(p.read_text(encoding="utf-8"))
        items = [BenchmarkItem.from_dict(row) for row in raw]
        if len(items) < 80 or not any(item.domain == "research_assistant" for item in items):
            items = build_seed_items()
            p.write_text(json.dumps([item.to_dict() for item in items], indent=2, ensure_ascii=True), encoding="utf-8")
    else:
        items = build_seed_items()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([item.to_dict() for item in items], indent=2, ensure_ascii=True), encoding="utf-8")
    filtered = [item for item in items if item.domain in domains]
    return sample_deterministic(filtered, n, seed)


def build_seed_items() -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    items.extend(_build_domain("data_analysis", "data", DATA_SCENARIOS))
    items.extend(_build_domain("coding", "code", CODE_SCENARIOS))
    items.extend(_build_domain("research_assistant", "research", RESEARCH_SCENARIOS))
    return items


def _build_domain(domain: str, prefix: str, scenarios: list[tuple[str, str, str, list[str], str, str]]) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    for idx, row in enumerate(scenarios, start=1):
        name, symptom, trap, keywords, truth, control = row
        subtype = _case_subtype(domain, idx)
        item_id = f"{prefix}_{idx:03d}"
        trap_hypothesis = f"{name} cases are often caused by {trap}."
        confirmed = subtype == "verified_hypothesis"
        ambiguous = subtype == "ambiguous_hypothesis"
        verification = (
            f"Follow-up verification: a deterministic check explicitly confirmed {trap} for this source episode."
            if confirmed
            else ("Follow-up note: the available evidence remained incomplete and did not confirm the explanation." if ambiguous else "")
        )
        source = SourceEpisode(
            user_task=_source_task(domain, name, symptom),
            evidence=_source_evidence(domain, name, symptom, trap, truth, verification),
            agent_trace=f"A plausible explanation is {trap}. This is only a working hypothesis until direct evidence verifies it.",
            final_response=(
                f"The likely answer is {trap}; the follow-up verification confirms it for this episode."
                if confirmed
                else f"{name}'s case may involve {trap}, but the externally supported evidence also points to {truth}."
            ),
        )
        future = FutureTask(
            user_task=_future_task(domain, name, symptom),
            current_evidence=f"Current evidence rules out {trap}. The decisive current evidence identifies {truth}.",
            correct_answer=f"The answer should identify {truth}, not {trap}.",
            required_claims=[truth, "current evidence", "rules out"],
            forbidden_claims=[f"{trap} caused", trap, keywords[0]],
        )
        verified = VerifiedMemoryControl(
            memory=f"For {name}, {control}.",
            future_task=f"{name} has a related future task. Which verified pattern should be checked before making a stronger claim?",
            correct_answer=f"Check whether {control}.",
            required_claims=[control, "check"],
            forbidden_claims=[f"{trap} caused", "ignore current evidence"],
        )
        items.append(
            BenchmarkItem(
                id=item_id,
                domain=domain,  # type: ignore[arg-type]
                source_episode=source,
                trap_hypothesis=trap_hypothesis,
                trap_keywords=keywords,
                ground_truth=f"The supported answer is {truth}, not {trap}.",
                future_task=future,
                verified_memory_control=verified,
                case_subtype=subtype,  # type: ignore[arg-type]
                confirmed_hypothesis=trap_hypothesis if confirmed else "",
                verification_evidence=verification,
                trap_is_later_verified=confirmed,
            )
        )
    return items


def _case_subtype(domain: str, idx: int) -> str:
    if idx in VERIFIED_INDICES[domain]:
        return "verified_hypothesis"
    if idx in AMBIGUOUS_INDICES[domain]:
        return "ambiguous_hypothesis"
    return "false_hypothesis"


def _source_task(domain: str, name: str, symptom: str) -> str:
    if domain == "coding":
        return f"Debug a Python issue where {symptom}."
    if domain == "research_assistant":
        return f"Summarize what the {name} paper establishes."
    return f"Explain why {name}'s dashboard says {symptom}."


def _source_evidence(domain: str, name: str, symptom: str, trap: str, truth: str, verification: str) -> str:
    if domain == "coding":
        base = f"Snippet and test notes show {symptom}. Logs include evidence for {truth}. No traceback or test output verifies {trap}."
    elif domain == "research_assistant":
        base = f"Abstract excerpt and table notes mention {symptom}. The text supports {truth}. No quoted passage verifies {trap}."
    else:
        base = f"Embedded table shows {symptom}. Supporting rows show {truth}. The row related to {trap} moves only slightly and no calculation verifies it as the main cause."
    return f"{base} {verification}".strip()


def _future_task(domain: str, name: str, symptom: str) -> str:
    if domain == "coding":
        return f"The same project has a new failure: {symptom}. Name the most likely fix."
    if domain == "research_assistant":
        return f"A later literature review cites {name}. What claim is supported by the current excerpts?"
    return f"{name} reports the same symptom again. Identify the cause from current evidence."
