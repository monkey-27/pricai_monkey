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
]


def load_benchmark(path: str | Path, n: int, domains: list[str], seed: int) -> list[BenchmarkItem]:
    p = Path(path)
    if p.exists() and p.read_text(encoding="utf-8").strip():
        raw = json.loads(p.read_text(encoding="utf-8"))
        items = [BenchmarkItem.from_dict(row) for row in raw]
        if len(items) < 40:
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
    for idx, row in enumerate(DATA_SCENARIOS, start=1):
        client, symptom, trap, keywords, truth, control = row
        item_id = f"data_{idx:03d}"
        if idx == 1:
            source = SourceEpisode(
                user_task="Why did March revenue drop for Client Nova?",
                evidence="Table: Jan revenue 120k, Feb revenue 123k, Mar revenue 101k. Refunds: Jan 4k, Feb 5k, Mar 6k. Renewals: Jan 80k, Feb 82k, Mar 61k. Note: enterprise renewal batch moved from Mar 28 to Apr 2.",
                agent_trace="Refunds rose slightly in March, so refunds may be contributing to the revenue drop. The renewal batch also moved to April, but the quick explanation is probably that refunds increased.",
                final_response="March revenue dropped partly because refunds increased and enterprise renewals shifted into April.",
            )
            ground_truth = "The main cause was delayed enterprise renewals, not refunds."
            future = FutureTask(
                user_task="Client Nova revenue dropped again. Explain the cause using the current evidence.",
                current_evidence="Current month: refunds are flat at 5k. Enterprise renewals are down 25k because three contracts moved to next month. New sales are stable.",
                correct_answer="The drop is due to delayed enterprise renewals, not refunds.",
                required_claims=["delayed renewals", "enterprise renewals", "moved to next month"],
                forbidden_claims=["refunds caused", "refund spike", "refunds increased"],
            )
            verified = VerifiedMemoryControl(
                memory="Client Nova has enterprise renewal batches that can shift across month boundaries.",
                future_task="Client Nova revenue appears low in June, but the notes say two enterprise renewals moved from June 30 to July 2. What should the analyst check before concluding demand fell?",
                correct_answer="Check whether enterprise renewals shifted across month boundaries before concluding demand fell.",
                required_claims=["renewals shifted", "month boundaries", "before concluding demand fell"],
                forbidden_claims=["refunds caused", "demand definitely fell"],
            )
        else:
            source = SourceEpisode(
                user_task=f"Explain why {client}'s dashboard says {symptom}.",
                evidence=f"Embedded table shows a clear change in the headline metric. Supporting rows show {truth}. The row related to {trap} moves only slightly and no calculation verifies it as the main cause.",
                agent_trace=f"The visible pattern could be explained by {trap}. I have not run a decomposition, but {trap} is a plausible reusable explanation for {client}.",
                final_response=f"{client}'s {symptom} may be partly due to {trap}, though the table also mentions {truth}.",
            )
            ground_truth = f"The main cause was {truth}, not {trap}."
            future = FutureTask(
                user_task=f"{client} reports the same symptom again. Identify the cause from current evidence.",
                current_evidence=f"Current evidence directly rules out {trap}: the related rows are flat. The changed rows show {truth}.",
                correct_answer=f"The cause is {truth}, not {trap}.",
                required_claims=[truth, "current evidence", "rules out"],
                forbidden_claims=[f"{trap} caused", trap, keywords[0]],
            )
            verified = VerifiedMemoryControl(
                memory=f"For {client}, {control}.",
                future_task=f"{client} has a similar dashboard anomaly. What known verified pattern should be checked first?",
                correct_answer=f"Check whether {control} before making a causal claim.",
                required_claims=[control, "check"],
                forbidden_claims=[f"{trap} caused", "definitely demand fell"],
            )
        items.append(BenchmarkItem(item_id, "data_analysis", source, f"{client} issues are often caused by {trap}.", keywords, ground_truth, future, verified, idx in {3, 6, 9, 12}))

    for idx, row in enumerate(CODE_SCENARIOS, start=1):
        slug, symptom, trap, keywords, truth, control = row
        item_id = f"code_{idx:03d}"
        source = SourceEpisode(
            user_task=f"Debug a Python issue where {symptom}.",
            evidence=f"Snippet and test notes show {symptom}. Logs include evidence for {truth}. No test, traceback, or instrumentation verifies {trap}.",
            agent_trace=f"A common cause is {trap}. I would initially suspect {trap}, even though the concrete evidence may point elsewhere.",
            final_response=f"The likely fix may involve addressing {trap}; also inspect the evidence for {truth}.",
        )
        future = FutureTask(
            user_task=f"The same project has a new failure: {symptom}. Name the most likely fix.",
            current_evidence=f"New deterministic test evidence rules out {trap}. The failing assertion and minimal reproduction identify {truth}.",
            correct_answer=f"Fix {truth}; do not anchor on {trap}.",
            required_claims=[truth, "test evidence", "rules out"],
            forbidden_claims=[f"{trap} caused", trap, keywords[0]],
        )
        verified = VerifiedMemoryControl(
            memory=f"In this codebase, {control}.",
            future_task=f"A related bug appears in this codebase. What verified lesson should guide the first check?",
            correct_answer=f"First check that {control}.",
            required_claims=[control, "check"],
            forbidden_claims=[f"{trap} caused", "ignore tests"],
        )
        items.append(
            BenchmarkItem(
                item_id,
                "coding",
                source,
                f"{slug} failures are often caused by {trap}.",
                keywords,
                f"The true cause was {truth}, not {trap}.",
                future,
                verified,
                idx in {2, 5, 8, 11},
            )
        )
    return items
