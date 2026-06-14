from __future__ import annotations

import argparse

from src.experiment import ALL_DOMAINS, ALL_METHODS, run_experiment
from src.report import write_summary


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hypothesis-laundered memory pilot.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--domains", default="coding,data_analysis")
    parser.add_argument("--methods", default="no_memory,naive,reflection,evidence_labeled")
    parser.add_argument("--out", default="outputs/run_001")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=800)
    args = parser.parse_args()

    domains = parse_csv(args.domains)
    methods = parse_csv(args.methods)
    unknown_domains = sorted(set(domains) - set(ALL_DOMAINS))
    unknown_methods = sorted(set(methods) - set(ALL_METHODS))
    if unknown_domains:
        raise SystemExit(f"Unknown domains: {unknown_domains}. Valid domains: {ALL_DOMAINS}")
    if unknown_methods:
        raise SystemExit(f"Unknown methods: {unknown_methods}. Valid methods: {ALL_METHODS}")
    scores, items = run_experiment(
        model=args.model,
        n=args.n,
        domains=domains,
        methods=methods,
        out=args.out,
        mock=args.mock,
        seed=args.seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    write_summary(args.out, scores, items, methods)
    print(f"Wrote pilot outputs to {args.out}")


if __name__ == "__main__":
    main()

