from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from eval.ecso.backends.base import ECSOBackend
from eval.ecso.prompts import CAP_PROMPT, LLM_GEN_PROMPT, TELL_PROMPT, is_unsafe
from eval.manifest import resolve_image_path


def run_ecso(
    backend: ECSOBackend,
    records,
    *,
    max_new_tokens_direct: int,
    max_new_tokens_ecso: int,
) -> list[dict]:
    outputs: list[dict] = []
    for record in tqdm(records, desc="ecso"):
        image_path = str(resolve_image_path(record.image_path))
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"image not found for {record.id}: {image_path}")

        direct_answer = backend.generate(
            record.query,
            image_path=image_path,
            max_new_tokens=max_new_tokens_direct,
        )

        tell_qs = TELL_PROMPT.format(query=record.query, response=direct_answer[:1927])
        tell_out = backend.generate(tell_qs, image_path=image_path, max_new_tokens=32)

        row = {
            "id": record.id,
            "query": record.query,
            "image_path": record.image_path,
            "direct_answer": direct_answer,
            "response": direct_answer,
            "method": "ecso",
            "unsafe": "No",
        }
        if record.answer_letter is not None:
            row["answer_letter"] = record.answer_letter
        if record.metadata:
            row["metadata"] = record.metadata
        row.update(record.extra)

        if is_unsafe(tell_out):
            caption = backend.generate(
                CAP_PROMPT.format(query=record.query),
                image_path=image_path,
                max_new_tokens=max_new_tokens_ecso,
            )
            safe_answer = backend.generate(
                LLM_GEN_PROMPT.format(caption=caption, query=record.query),
                image_path=None,
                max_new_tokens=max_new_tokens_ecso,
            )
            row.update(
                {
                    "unsafe": "Yes",
                    "caption": caption,
                    "unsafe_answer": direct_answer,
                    "response": safe_answer,
                }
            )

        outputs.append(row)
    return outputs
