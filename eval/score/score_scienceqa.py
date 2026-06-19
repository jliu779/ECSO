from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from eval.io import read_jsonl
from eval.manifest import read_manifest

OPTIONS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def extract_choice(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if text[0] in OPTIONS and (len(text) == 1 or text[1:3] in {". ", ")", ": "}):
        return text[0]
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])\b",
        r"\(([A-Z])\)",
        r"\b([A-Z])\.\s",
        r"at the end\.?\s*([A-Z])\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    for ch in OPTIONS:
        if re.search(rf"\b{ch}\b", text):
            return ch
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = {r.id: r for r in read_manifest(args.manifest)}
    gens = read_jsonl(args.generations)

    correct = 0
    total = 0
    rows_out = []
    for row in gens:
        rec = manifest.get(row["id"])
        if rec is None or not rec.answer_letter:
            continue
        pred = extract_choice(row.get("response", ""))
        gold = rec.answer_letter.upper()
        ok = pred == gold
        total += 1
        correct += int(ok)
        rows_out.append(
            {
                "id": row["id"],
                "gold": gold,
                "pred": pred or "",
                "correct": int(ok),
            }
        )

    acc = correct / total if total else 0.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "gold", "pred", "correct"])
        writer.writeheader()
        writer.writerows(rows_out)
        writer.writerow({"id": "__summary__", "gold": "", "pred": f"acc={acc:.4f}", "correct": correct})

    print(f"accuracy={acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
