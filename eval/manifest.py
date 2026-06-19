from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.io import read_jsonl


@dataclass
class ManifestRecord:
    id: str
    query: str
    image_path: str
    answer_letter: str | None = None
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> ManifestRecord:
        known = {"id", "query", "image_path", "answer_letter", "caption", "metadata"}
        extra = {k: v for k, v in row.items() if k not in known}
        return cls(
            id=str(row["id"]),
            query=str(row["query"]),
            image_path=str(row["image_path"]),
            answer_letter=row.get("answer_letter"),
            caption=row.get("caption"),
            metadata=dict(row.get("metadata") or {}),
            extra=extra,
        )


def read_manifest(path: str | Path, *, limit: int | None = None) -> list[ManifestRecord]:
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return [ManifestRecord.from_dict(row) for row in rows]


def resolve_image_path(image_path: str) -> Path:
    return Path(image_path).expanduser()
