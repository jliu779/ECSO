from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_summary_acc(path: Path) -> float | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("id") == "__summary__" and "acc=" in (row.get("pred") or ""):
                try:
                    return float(row["pred"].split("acc=")[1])
                except (IndexError, ValueError):
                    return None
    return None


def read_judge_rate(path: Path, field: str) -> float | None:
    if not path.is_file():
        return None
    import json

    total = 0
    positive = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if field not in row:
                continue
            total += 1
            val = str(row[field]).lower()
            if val in {"yes", "no"}:
                positive += int(val == "yes")
            elif field == "over_refusal":
                positive += int(val == "yes")
            elif field in {"actionable", "context_unsafe"}:
                positive += int(val == "yes")
    return positive / total if total else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--method", default="ecso")
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    lines = [f"# {args.method} summary", ""]

    mcq_benches = {
        "scienceqa_imgval_full": "sciqa_full_score.csv",
        "mmstar": "mmstar_score.csv",
        "mme_realworld": "mme_realworld_score.csv",
    }
    lines.append("## Utility (accuracy)")
    for stem, csv_name in mcq_benches.items():
        acc = read_summary_acc(out_dir / csv_name)
        lines.append(f"- **{stem}**: {acc:.4f}" if acc is not None else f"- **{stem}**: n/a")

    for stem in ("mathvista", "colorbench"):
        acc = read_summary_acc(out_dir / f"{stem}_score.csv")
        lines.append(f"- **{stem}**: {acc:.4f}" if acc is not None else f"- **{stem}**: n/a")

    lines.append("")
    lines.append("## Safety")
    safety_specs = [
        ("vlsafe_examine_eval", "actionable", "actionable unsafe rate"),
        ("spa_vl_test_530", "actionable", "actionable unsafe rate"),
        ("mmsb_vision_risk_sdtypo", "actionable", "actionable unsafe rate"),
        ("mm_safetybench_300", "actionable", "actionable unsafe rate"),
        ("siuo_167", "context_unsafe", "context unsafe rate"),
        ("mssbench_unsafe_full", "context_unsafe", "context unsafe rate"),
        ("mossbench", "over_refusal", "over-refusal rate"),
        ("xstest_safe", "over_refusal", "over-refusal rate"),
    ]
    for stem, field, label in safety_specs:
        rate = read_judge_rate(out_dir / f"{stem}.judged.jsonl", field)
        lines.append(f"- **{stem}** ({label}): {rate:.4f}" if rate is not None else f"- **{stem}**: n/a")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
