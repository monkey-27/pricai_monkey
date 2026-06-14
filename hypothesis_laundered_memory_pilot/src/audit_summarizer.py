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
    contamination_agree = sum(1 for r in completed if _is_contaminated(r.get("human_label", "")) == _is_contaminated(r.get("auto_label", "")))
    false_promotion_rows = [r for r in completed if r.get("human_false_promotion", "").strip()]
    false_promotion_agree = sum(
        1
        for r in false_promotion_rows
        if _as_bool(r.get("human_false_promotion", "")) == _as_bool(r.get("auto_false_promotion", ""))
    )
    auto_counts = Counter(r.get("auto_label", "") for r in completed)
    human_counts = Counter(r.get("human_label", "") for r in completed)
    lines = [
        "# Manual Audit Summary",
        "",
        f"- completed rows: `{total}`",
        f"- exact agreement: `{(agree / total):.3f}`" if total else "- exact agreement: `NA`",
        f"- contamination agreement: `{(contamination_agree / total):.3f}`" if total else "- contamination agreement: `NA`",
        (
            f"- false promotion agreement: `{(false_promotion_agree / len(false_promotion_rows)):.3f}`"
            if false_promotion_rows
            else "- false promotion agreement: `NA`"
        ),
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


def _is_contaminated(label: str) -> bool:
    return label in {"contaminated", "mixed_endorsed_trap"}


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize manual audit labels.")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summarize(args.audit, args.out)


if __name__ == "__main__":
    main()
