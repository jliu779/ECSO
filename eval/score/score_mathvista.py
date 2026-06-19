from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from eval.io import read_jsonl
from eval.manifest import read_manifest

from eval.score.score_scienceqa import extract_choice


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"[.,;:!?]+$", "", text)
    return text


def score_row(row: dict, rec) -> tuple[bool, str, str]:
    response = row.get("response", "")
    meta = rec.metadata or {}
    qtype = meta.get("question_type", "multiple_choice")

    if qtype == "multiple_choice" or rec.answer_letter:
        pred = extract_choice(response) or ""
        gold = (rec.answer_letter or meta.get("answer") or "").strip().upper()
        if len(gold) == 1:
            return pred == gold, pred, gold
        return normalize_answer(pred) == normalize_answer(gold), pred, gold

    gold = str(meta.get("answer", "")).strip()
    pred = response.strip()
    if meta.get("answer_type") == "integer":
        nums = re.findall(r"-?\d+", pred)
        pred_norm = nums[-1] if nums else pred
        gold_norm = re.findall(r"-?\d+", gold)
        gold_val = gold_norm[0] if gold_norm else gold
        return pred_norm == gold_val, pred_norm, gold_val

    if meta.get("answer_type") == "float" and meta.get("precision") is not None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", pred)
        pred_norm = nums[-1] if nums else pred
        try:
            return abs(float(pred_norm) - float(gold)) < 1e-6, pred_norm, gold
        except ValueError:
            return False, pred_norm, gold

    return normalize_answer(pred) == normalize_answer(gold), pred, gold


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
        if rec is None:
            continue
        ok, pred, gold = score_row(row, rec)
        total += 1
        correct += int(ok)
        rows_out.append({"id": row["id"], "gold": gold, "pred": pred, "correct": int(ok)})

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
