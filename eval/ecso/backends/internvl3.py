from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import GenerationConfig

from procrustes.internvl3_utils import build_internvl3_question, load_internvl3
from procrustes.internvl_utils import load_image_pixel_values


@dataclass
class InternVL3Backend:
    """ECSO backend for OpenGVLab/InternVL3-8B (custom ``model.chat`` API)."""

    model_path: str
    temperature: float = 0.0
    use_flash_attn: bool = False
    max_num_tiles: int = 12
    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        path = str(Path(self.model_path).expanduser())
        self.model, self.tokenizer = load_internvl3(
            path,
            dtype=self.torch_dtype,
            use_flash_attn=self.use_flash_attn,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _generation_config(self, max_new_tokens: int) -> GenerationConfig:
        if self.temperature > 0:
            return GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
            )
        return GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)

    def generate(
        self,
        query: str,
        *,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> str:
        has_image = image_path is not None
        question = build_internvl3_question(query, has_image=has_image)

        pixel_values = None
        if has_image:
            pixel_values = load_image_pixel_values(
                image_path,
                max_num=self.max_num_tiles,
                dtype=self.torch_dtype,
            ).to(self.device)

        if not hasattr(self.model, "chat"):
            raise AttributeError(
                "Loaded InternVL checkpoint has no .chat(); use the custom-format "
                "OpenGVLab/InternVL3-8B weights (not the -hf processor-only build)."
            )

        gen_config = self._generation_config(max_new_tokens)
        with torch.inference_mode():
            result = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                gen_config,
            )

        if isinstance(result, tuple):
            result = result[0]
        return str(result).strip()
