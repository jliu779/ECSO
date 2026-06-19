from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from procrustes.phi4_utils import build_phi4_prompt, load_phi4


@dataclass
class Phi4Backend:
    """ECSO backend for microsoft/Phi-4-multimodal-instruct."""

    model_path: str
    temperature: float = 0.0
    attn_implementation: str = "eager"
    disable_cache: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        path = str(Path(self.model_path).expanduser())
        self.model, self.processor = load_phi4(
            path,
            dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
            use_cache=not self.disable_cache,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _processor_inputs(self, query: str, image_path: str | None) -> dict:
        has_image = image_path is not None
        prompt = build_phi4_prompt(self.processor, query, has_image=has_image)
        if has_image:
            image = Image.open(image_path).convert("RGB")
            try:
                return self.processor(text=prompt, images=image, return_tensors="pt")
            except TypeError:
                return self.processor(prompt, images=image, return_tensors="pt")
        try:
            return self.processor(text=prompt, return_tensors="pt")
        except TypeError:
            return self.processor(prompt, return_tensors="pt")

    def generate(
        self,
        query: str,
        *,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> str:
        inputs = self._processor_inputs(query, image_path)
        inputs = {
            key: val.to(self.device) if hasattr(val, "to") else val
            for key, val in inputs.items()
        }

        gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        decoded = tokenizer.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()
