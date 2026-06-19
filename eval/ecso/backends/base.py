from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ECSOBackend(Protocol):
    """Unified interface for ECSO four-step generation on a subject VLM."""

    def generate(
        self,
        query: str,
        *,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> str:
        """Generate a response. Pass ``image_path`` for vision steps; ``None`` for text-only (step 4)."""
        ...
