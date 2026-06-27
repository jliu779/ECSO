from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from procrustes.glm41v_utils import build_glm41v_messages, load_glm41v

# Cap at ~1M pixels to prevent vision encoder OOM on high-res images.
# Glm4vProcessor silently ignores max_pixels kwarg, so we resize with PIL.
_MAX_PIXELS = 1280 * 28 * 28


@dataclass
class Glm41vBackend:
    """ECSO backend for zai-org/GLM-4.1V-9B-Thinking."""

    model_path: str
    temperature: float = 0.0
    attn_implementation: str = "eager"
    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        path = str(Path(self.model_path).expanduser())
        self.model, self.processor = load_glm41v(
            path,
            dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _resize_image_if_needed(self, image_path: str) -> tuple[str, bool]:
        """Return (path_to_use, created_tmp).

        GLM-4V smart_resize requires both dimensions >= 28 (patch factor) and
        rejects images exceeding _MAX_PIXELS. Handle both bounds here.
        """
        from PIL import Image as PilImage
        _PATCH = 28
        img = PilImage.open(image_path).convert("RGB")
        w, h = img.size

        # Scale down if too large
        if w * h > _MAX_PIXELS:
            scale = (_MAX_PIXELS / (w * h)) ** 0.5
            w, h = max(int(w * scale), _PATCH), max(int(h * scale), _PATCH)

        # Scale up if either dimension is below the minimum patch size
        if w < _PATCH:
            h = max(int(h * _PATCH / w), _PATCH)
            w = _PATCH
        if h < _PATCH:
            w = max(int(w * _PATCH / h), _PATCH)
            h = _PATCH

        if (w, h) == img.size:
            return image_path, False

        img = img.resize((w, h), PilImage.LANCZOS)
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        img.save(tmp, "JPEG")
        return tmp, True

    def _prepare_inputs(self, query: str, image_path: str | None) -> dict:
        resolved = str(Path(image_path).expanduser().resolve()) if image_path else None
        tmp_created = False
        if resolved:
            resolved, tmp_created = self._resize_image_if_needed(resolved)
        try:
            messages = build_glm41v_messages(query, resolved)
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            if tmp_created and resolved and os.path.exists(resolved):
                os.unlink(resolved)
        if isinstance(inputs, dict):
            inputs.pop("token_type_ids", None)
            return {
                key: val.to(self.device) if hasattr(val, "to") else val
                for key, val in inputs.items()
            }
        inputs.pop("token_type_ids", None)
        return inputs.to(self.device)

    def generate(
        self,
        query: str,
        *,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> str:
        inputs = self._prepare_inputs(query, image_path)

        gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_ids = inputs["input_ids"]
        trimmed = [out[len(inp) :] for inp, out in zip(input_ids, output_ids)]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()
