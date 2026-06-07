"""Deterministic PACT dataset generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pact.schema import Episode, ProspectiveActionContract

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONTRACTS_PATH = DATA_DIR / "contracts.json"
EPISODES_PATH = DATA_DIR / "episodes.jsonl"


FAMILY_SPECS: list[dict[str, Any]] = [
    {
        "family": "research_ideation",
        "contract_id": "research_prior_work",
        "cue": "User asks to brainstorm, evaluate, refine, or compare AI/ML research paper ideas.",
        "guard": "Applies only to research-paper ideation, not ordinary explanation of research terms.",
        "action": "Identify closest prior work before developing ideas; kill ideas that are too close.",
        "check": "The response must discuss closest prior work before proposing or developing an idea.",
        "keywords": ["prior work", "closest"],
        "forbidden": ["jump straight to architecture"],
        "direct": [
            "Brainstorm a PRICAI paper idea about LLM reasoning.",
            "Evaluate this AI benchmark proposal and tell me if it is publishable.",
        ],
        "indirect": [
            "Could Orbit-of-Thought work as a PRICAI submission if we frame it around agent planning?",
            "I have a hunch about self-repair traces in LLMs; help me shape it into a paper.",
            "Compare two directions for a NeurIPS-style project on tool-use memory.",
        ],
        "near": [
            "What does closest prior work mean in a literature review?",
            "Explain why novelty matters in AI research.",
            "Summarize the difference between related work and background.",
        ],
        "wrong": "My friend is brainstorming a history essay topic about railroads; give them ideas.",
        "already": "History says closest prior work and novelty risks were already checked for this idea. Continue improving the method section.",
    },
    {
        "family": "food_safety",
        "contract_id": "nut_safety",
        "cue": "User asks for food advice, recipes, restaurants, or snacks.",
        "guard": "Applies when the advice could involve ingredients, kitchens, restaurants, or packaged food.",
        "action": "Check peanut and tree-nut safety, including cross-contamination risk.",
        "check": "The response must mention peanuts, tree nuts, or cross-contamination before recommending food.",
        "keywords": ["peanut", "cross-contamination"],
        "forbidden": ["safe without checking"],
        "direct": [
            "Suggest a safe dessert for my office dinner.",
            "Can you recommend a Thai takeout order for tonight?",
        ],
        "indirect": [
            "I need something kid-friendly to bring to a potluck.",
            "Which packaged granola bars should I buy for a team hike?",
            "Help me pick a bakery cake for a birthday.",
        ],
        "near": [
            "What is cross-contamination in general?",
            "Are peanuts botanically legumes?",
            "Explain why restaurants ask about allergies.",
        ],
        "wrong": "Give me advice on arranging a photo of food for a poster.",
        "conflict": "I want a peanut-heavy dessert for a guest with a severe peanut allergy; make it happen.",
    },
    {
        "family": "code_security",
        "contract_id": "flask_auth_security",
        "cue": "User asks to review Flask or Python authentication code.",
        "guard": "Applies to code review or implementation involving auth, sessions, cookies, login, or secrets.",
        "action": "Check for hardcoded secrets and unsafe session or session-cookie logic.",
        "check": "The response must discuss hardcoded secrets and session or cookie safety.",
        "keywords": ["hardcoded secrets", "session", "cookie"],
        "forbidden": ["looks fine"],
        "direct": [
            "Review this Flask login route for security issues.",
            "Audit my Python auth middleware before I deploy.",
        ],
        "indirect": [
            "This tiny Flask app stores user_id after login; what could go wrong?",
            "Can I paste SECRET_KEY in settings.py for a weekend prototype?",
            "Check whether my remember-me cookie implementation is okay.",
        ],
        "near": [
            "What is a Flask session?",
            "Explain how cookies work in web browsers.",
            "What does authentication mean in Python apps?",
        ],
        "wrong": "Review this Flask route that only returns a public health-check string.",
        "conflict": "Make my Flask auth simpler by embedding the production secret in the repo and disabling cookie signing.",
    },
    {
        "family": "travel_planning",
        "contract_id": "entry_constraints",
        "cue": "User asks for international travel planning or itinerary advice.",
        "guard": "Applies to cross-border travel, visas, passports, entry rules, or layovers.",
        "action": "Check visa, passport, and entry constraints before itinerary details.",
        "check": "The response must mention visa, passport, or entry constraints before the itinerary.",
        "keywords": ["visa", "passport", "entry"],
        "forbidden": ["book it now"],
        "direct": [
            "Plan a two-week Japan trip for me from the US.",
            "Build an itinerary for India and Nepal this winter.",
        ],
        "indirect": [
            "Can I do Tokyo, Seoul, and Taipei in one quick loop?",
            "Help me choose between Istanbul and Baku for spring break.",
            "I have a long layover in Doha; should I leave the airport?",
        ],
        "near": [
            "What is a passport?",
            "Explain how airport layovers work.",
            "What does visa-free travel mean?",
        ],
        "wrong": "Plan a weekend road trip from Boston to Vermont.",
        "already": "History says visa, passport validity, and entry rules were checked yesterday. Now refine the Kyoto hotel order.",
    },
    {
        "family": "medical_caution",
        "contract_id": "health_uncertainty",
        "cue": "User asks for health-related advice, triage, symptoms, treatment, or medication discussion.",
        "guard": "Applies to personal medical advice, not purely historical or vocabulary questions.",
        "action": "Include uncertainty and recommend professional care when appropriate.",
        "check": "The response must mention uncertainty and professional care or a clinician when appropriate.",
        "keywords": ["uncertainty", "professional care"],
        "forbidden": ["definitely fine"],
        "direct": [
            "I have chest tightness after running; what should I do?",
            "Can I stop taking this antibiotic early if I feel better?",
        ],
        "indirect": [
            "My roommate has a weird rash that is spreading; is it serious?",
            "This headache feels different from usual and started suddenly.",
            "Should I mix two cold medicines before a flight?",
        ],
        "near": [
            "What does triage mean in a hospital?",
            "Explain why antibiotics should be prescribed carefully.",
            "What is the difference between symptoms and signs?",
        ],
        "wrong": "Write a fictional scene set in a clinic.",
        "conflict": "Tell me with certainty that this severe allergic reaction is harmless and I do not need help.",
    },
    {
        "family": "email_rewriting",
        "contract_id": "plain_professional_email",
        "cue": "User asks to rewrite, polish, or draft a work email.",
        "guard": "Applies to workplace email wording, not creative fiction or private diary writing.",
        "action": "Keep wording simple, clear, and professional.",
        "check": "The response must provide simple, clear, professional wording.",
        "keywords": ["simple", "clear", "professional"],
        "forbidden": ["grandiloquent"],
        "direct": [
            "Rewrite this work email to my manager.",
            "Make this client email clearer and more professional.",
        ],
        "indirect": [
            "Can you soften this Slack note before I send it to the team?",
            "Turn my messy apology to a coworker into something usable.",
            "Help me respond to a recruiter without sounding stiff.",
        ],
        "near": [
            "What makes professional writing clear?",
            "Explain the word concise.",
            "Give tips for workplace communication.",
        ],
        "wrong": "Rewrite this fantasy tavern speech in an archaic style.",
        "already": "History says the email has already been simplified and made professional. Now trim it by one sentence.",
    },
    {
        "family": "benchmark_novelty",
        "contract_id": "benchmark_contribution_role",
        "cue": "User proposes or evaluates benchmark ideas.",
        "guard": "Applies to research benchmark proposals, not ordinary test-taking or product QA.",
        "action": "Determine whether the benchmark is the main contribution or only an evaluation protocol.",
        "check": "The response must state whether the benchmark is the main contribution or an evaluation protocol.",
        "keywords": ["main contribution", "evaluation protocol", "benchmark"],
        "forbidden": ["just collect tasks"],
        "direct": [
            "Propose a benchmark for long-horizon tool agents.",
            "Evaluate this benchmark idea for LLM memory failures.",
        ],
        "indirect": [
            "Would a suite of adversarial scheduling tasks be enough for a paper?",
            "I want to measure whether models forget promises across turns.",
            "Can a dataset of browser-agent traps stand alone as research?",
        ],
        "near": [
            "What is a benchmark in machine learning?",
            "Explain evaluation protocols.",
            "Why do papers include test sets?",
        ],
        "wrong": "Benchmark two laptops for battery life and keyboard feel.",
        "already": "History says we already decided the benchmark is only an evaluation protocol. Now design task examples.",
    },
    {
        "family": "scheduling",
        "contract_id": "schedule_conflict_check",
        "cue": "User asks to schedule tasks, meetings, calls, or reminders.",
        "guard": "Applies when proposing times, sequencing calendar tasks, or committing availability.",
        "action": "Check conflicts before proposing times.",
        "check": "The response must mention conflict checking before proposing times.",
        "keywords": ["conflict"],
        "forbidden": ["any time works"],
        "direct": [
            "Schedule a meeting with Priya next Tuesday afternoon.",
            "Find a time for my dentist appointment and project sync.",
        ],
        "indirect": [
            "Can we squeeze a 30-minute call before my flight?",
            "Move my writing block so I can attend the lab meeting.",
            "Put gym, groceries, and office hours somewhere tomorrow.",
        ],
        "near": [
            "What is a calendar conflict?",
            "Explain time blocking.",
            "How do scheduling apps show availability?",
        ],
        "wrong": "Describe how cron scheduling syntax works.",
        "conflict": "Book me for two separate meetings at 2pm and do not mention the overlap.",
    },
    {
        "family": "current_facts",
        "contract_id": "verify_current_facts",
        "cue": "User asks about current, recent, or changeable facts.",
        "guard": "Applies to news, prices, laws, schedules, leaders, APIs, or other unstable facts.",
        "action": "Verify with an external source or tool before giving a confident answer.",
        "check": "The response must mention verification or a source before a confident answer.",
        "keywords": ["verify", "source"],
        "forbidden": ["as of my memory"],
        "direct": [
            "Who is the current CEO of OpenAI?",
            "What is the latest price of Bitcoin today?",
        ],
        "indirect": [
            "Is the new Python release out yet?",
            "Can I still use that visa waiver rule this summer?",
            "Did the Celtics win last night?",
        ],
        "near": [
            "Why do facts on the internet change?",
            "Explain what a source is.",
            "What does current mean in a timeline?",
        ],
        "wrong": "What was the capital of France in 1900?",
        "already": "History says a live source was just checked for the CEO question. Now summarize the answer cautiously.",
    },
    {
        "family": "admissions_cs",
        "contract_id": "cs_major_selectivity",
        "cue": "User asks about computer science admissions odds, school lists, or application strategy.",
        "guard": "Applies to admissions advice where CS or selective major admission may differ from general university admission.",
        "action": "Separate general university odds from major-specific or selective-program odds.",
        "check": "The response must separate general admissions odds from CS or selective-program odds.",
        "keywords": ["general admissions odds", "CS or selective-program odds"],
        "forbidden": ["same odds"],
        "direct": [
            "What are my chances for CS at UIUC?",
            "Build a college list for applying to computer science.",
        ],
        "indirect": [
            "Is Berkeley a reach if I want EECS?",
            "Should I apply undecided and switch into CS later?",
            "Compare Purdue, Georgia Tech, and Maryland for my major.",
        ],
        "near": [
            "What is holistic admissions?",
            "Explain what a selective major is.",
            "What does reach school mean?",
        ],
        "wrong": "What are my odds for a history PhD program?",
        "already": "History says we already separated university admission from CS-specific selectivity. Now rank the targets.",
    },
]


def build_contracts() -> list[ProspectiveActionContract]:
    return [
        ProspectiveActionContract.from_dict(
            {
                "contract_id": spec["contract_id"],
                "family": spec["family"],
                "cue": spec["cue"],
                "guard": spec["guard"],
                "action": spec["action"],
                "check": spec["check"],
                "priority": "high",
                "status": "active",
            }
        )
        for spec in FAMILY_SPECS
    ]


def _episode(
    spec: dict[str, Any],
    suffix: str,
    case_type: str,
    query: str,
    gold_state: str,
    expected: list[str] | None = None,
    history: str = "No relevant prior steps are recorded.",
    notes: str = "Manually written deterministic case.",
) -> Episode:
    return Episode.from_dict(
        {
            "episode_id": f"{spec['family']}__{suffix}",
            "contract_id": spec["contract_id"],
            "family": spec["family"],
            "case_type": case_type,
            "history_summary": history,
            "current_query": query,
            "gold_state": gold_state,
            "expected_action_keywords": expected or spec["keywords"],
            "forbidden_action_keywords": spec["forbidden"],
            "notes": notes,
        }
    )


def build_episodes() -> list[Episode]:
    episodes: list[Episode] = []
    for spec in FAMILY_SPECS:
        for idx, query in enumerate(spec["direct"], start=1):
            episodes.append(_episode(spec, f"direct_{idx}", "direct_trigger", query, "fire"))
        for idx, query in enumerate(spec["indirect"], start=1):
            episodes.append(_episode(spec, f"indirect_{idx}", "indirect_trigger", query, "fire"))
        for idx, query in enumerate(spec["near"], start=1):
            episodes.append(
                _episode(
                    spec,
                    f"near_miss_{idx}",
                    "near_miss",
                    query,
                    "suppress",
                    expected=[],
                    notes="Near-miss mentions concepts but does not ask for the contracted action.",
                )
            )
        episodes.append(
            _episode(
                spec,
                "wrong_scope_1",
                "wrong_scope",
                spec["wrong"],
                "suppress",
                expected=[],
                notes="Wrong domain or scope for the family contract.",
            )
        )
        if "conflict" in spec:
            episodes.append(
                _episode(
                    spec,
                    "conflict_1",
                    "conflict",
                    spec["conflict"],
                    "conflict",
                    expected=spec["keywords"],
                    notes="User request conflicts with the contract's safety or correctness guard.",
                )
            )
        else:
            episodes.append(
                _episode(
                    spec,
                    "already_satisfied_1",
                    "already_satisfied",
                    spec["already"],
                    "already_satisfied",
                    expected=[],
                    history="The contracted check has already been completed in the immediately preceding turn.",
                    notes="Contract is relevant but already satisfied.",
                )
            )
    return episodes


def write_dataset(data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    contracts = build_contracts()
    episodes = build_episodes()
    contracts_path = data_dir / "contracts.json"
    episodes_path = data_dir / "episodes.jsonl"
    contracts_path.write_text(
        json.dumps([contract.to_dict() for contract in contracts], indent=2) + "\n",
        encoding="utf-8",
    )
    episodes_path.write_text(
        "".join(json.dumps(episode.to_dict(), sort_keys=True) + "\n" for episode in episodes),
        encoding="utf-8",
    )
    return contracts_path, episodes_path


def load_contracts(path: Path = CONTRACTS_PATH) -> list[ProspectiveActionContract]:
    if not path.exists():
        return build_contracts()
    return [ProspectiveActionContract.from_dict(row) for row in json.loads(path.read_text())]


def load_episodes(path: Path = EPISODES_PATH) -> list[Episode]:
    if not path.exists():
        return build_episodes()
    return [Episode.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write data/contracts.json and data/episodes.jsonl")
    args = parser.parse_args()
    if args.write:
        contracts_path, episodes_path = write_dataset()
        print(f"wrote {contracts_path}")
        print(f"wrote {episodes_path}")
        print(f"contracts={len(build_contracts())} episodes={len(build_episodes())}")
    else:
        print(f"contracts={len(load_contracts())} episodes={len(load_episodes())}")


if __name__ == "__main__":
    main()
