"""Manual audit export entrypoint."""

from __future__ import annotations

from pact.run_eval import run


def main() -> None:
    run(dataset="pact_causal_520", methods="all", split="test", audit=False, bootstrap_iters=100)


if __name__ == "__main__":
    main()

