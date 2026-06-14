from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def summarize(audit: str | Path, out: str | Path) -> None:
    rows = list(csv.DictReader(Path(audit).open("r", encoding="utf-8")))
    completed = [r for r in rows if r.get("human_label", "").strip()]
    total = len(completed)
    agree = sum(1 for r in completed if r.get("human_label") == r.get("auto_label"))
    auto_counts = Counter(r.get("auto_label", "") for r in completed)
    human_counts = Counter(r.get("human_label", "") for r in completed)
    lines = [
        "# Manual Audit Summary",
        "",
        f"- completed rows: `{total}`",
        f"- exact agreement: `{(agree / total):.3f}`" if total else "- exact agreement: `NA`",
        f"- auto labels: `{dict(auto_counts)}`",
        f"- human labels: `{dict(human_counts)}`",
        "",
        "## Disagreements",
        "",
    ]
    disagreements = [r for r in completed if r.get("human_label") != r.get("auto_label")]
    for row in disagreements[:20]:
        lines.append(
            f"- `{row.get('item_id')}` `{row.get('method')}` auto=`{row.get('auto_label')}` human=`{row.get('human_label')}` notes={row.get('human_notes', '')}"
        )
    if not disagreements:
        lines.append("- No disagreements in completed rows.")
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize manual audit labels.")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summarize(args.audit, args.out)


if __name__ == "__main__":
    main()
