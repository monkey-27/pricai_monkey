from __future__ import annotations

import argparse

from src.experiment import ALL_DOMAINS, ALL_METHODS, DEFAULT_METHODS, run_experiment
from src.model_config import load_model_config, select_models
from src.report import write_summary


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hypothesis-laundered memory pilot.")
    parser.add_argument("--backend", default="openai_compatible", choices=["openai_compatible", "transformers"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--models-config", default="configs/open_models.yaml")
    parser.add_argument("--model-tier", default="recommended_main", choices=["smoke", "recommended_small", "recommended_main", "optional_stronger", "all"])
    parser.add_argument("--allow-download", default="false")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--benchmark", default="data/benchmark_seed.json")
    parser.add_argument("--domains", default="coding,data_analysis,research_assistant")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--out", default="outputs/run_001")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--max-tokens", type=int, default=None, help="Backward-compatible alias for --max-new-tokens.")
    parser.add_argument("--audit-sample-size", type=int, default=20)
    parser.add_argument("--judge-backend", default="none", choices=["none", "same"])
    parser.add_argument("--judge-model", default="same")
    args = parser.parse_args()

    max_new_tokens = args.max_tokens if args.max_tokens is not None else args.max_new_tokens
    allow_download = parse_bool(args.allow_download)
    hf_model = args.hf_model
    if args.backend == "transformers" and hf_model is None:
        config = load_model_config(args.models_config)
        selected = select_models(config, args.model_tier)
        if not selected:
            raise SystemExit(f"No models found for tier {args.model_tier} in {args.models_config}")
        hf_model = selected[0]
        if args.model == "gpt-4.1-mini":
            args.model = hf_model
    domains = parse_csv(args.domains)
    methods = parse_csv(args.methods)
    unknown_domains = sorted(set(domains) - set(ALL_DOMAINS))
    unknown_methods = sorted(set(methods) - set(ALL_METHODS))
    if unknown_domains:
        raise SystemExit(f"Unknown domains: {unknown_domains}. Valid domains: {ALL_DOMAINS}")
    if unknown_methods:
        raise SystemExit(f"Unknown methods: {unknown_methods}. Valid methods: {ALL_METHODS}")
    scores, items, metadata = run_experiment(
        model=args.model,
        n=args.n,
        domains=domains,
        methods=methods,
        out=args.out,
        mock=args.mock,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=max_new_tokens,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        hf_model=hf_model,
        device=args.device,
        dtype=args.dtype,
        audit_sample_size=args.audit_sample_size,
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        allow_download=allow_download,
        benchmark=args.benchmark,
    )
    write_summary(args.out, scores, items, methods, metadata)
    print(f"Wrote pilot outputs to {args.out}")


if __name__ == "__main__":
    main()
