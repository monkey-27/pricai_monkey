"""Command-line evaluation runner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pact.audits import run_all_audits
from pact.baselines import (
    ContractPromptHeuristicBaseline,
    KeywordTriggerBaseline,
    NoMemoryBaseline,
    TfidfMemoryBaseline,
    public_episode,
)
from pact.dataset import ROOT, load_contracts, load_episodes, write_dataset
from pact.pact import METHOD_NAMES, get_pact_method
from pact.schema import Prediction
from pact.scoring import format_summary, group_predictions, score_method

OUTPUT_DIR = ROOT / "outputs"


def build_methods(methods: str):
    if methods == "all":
        names = METHOD_NAMES
    else:
        names = [name.strip() for name in methods.split(",") if name.strip()]
    built = []
    for name in names:
        if name == "NoMemoryBaseline":
            built.append(NoMemoryBaseline())
        elif name == "KeywordTriggerBaseline":
            built.append(KeywordTriggerBaseline())
        elif name == "TfidfMemoryBaseline":
            built.append(TfidfMemoryBaseline())
        elif name == "ContractPromptHeuristicBaseline":
            built.append(ContractPromptHeuristicBaseline())
        else:
            built.append(get_pact_method(name))
    return built


def write_predictions(path: Path, predictions: list[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [prediction.to_dict() for prediction in predictions]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(methods: str = "all", audit: bool = False) -> dict[str, dict[str, float]]:
    write_dataset()
    contracts = load_contracts()
    episodes = load_episodes()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_predictions: list[Prediction] = []
    for method in build_methods(methods):
        for episode in episodes:
            all_predictions.append(method.predict(contracts, public_episode(episode)))
    grouped = group_predictions(all_predictions)
    metrics = {
        method_name: score_method(episodes, predictions)
        for method_name, predictions in grouped.items()
    }
    write_predictions(OUTPUT_DIR / "predictions.csv", all_predictions)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    summary = format_summary(metrics)
    (OUTPUT_DIR / "audit_report.md").write_text(
        "# PACT Pilot Summary\n\n" + summary + "\n\nHeadline: action completion under indirect triggers at low false-trigger rate.\n",
        encoding="utf-8",
    )
    if audit:
        run_all_audits(contracts, episodes, metrics, OUTPUT_DIR)
    print(summary)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="all", help="all or comma-separated method names")
    parser.add_argument("--audit", action="store_true", help="run named audit phases")
    args = parser.parse_args()
    run(methods=args.methods, audit=args.audit)


if __name__ == "__main__":
    main()

