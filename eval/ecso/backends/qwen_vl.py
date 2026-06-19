from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

QWEN_VLMS = frozenset({"qwen25vl", "qwen3vl"})


def _resolve_model_cls(vlm: str):
    if vlm == "qwen25vl":
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration, "qwen2"
    if vlm == "qwen3vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "qwen3vl requires transformers with Qwen3VLForConditionalGeneration "
                "(install a recent transformers release)"
            ) from exc
        return Qwen3VLForConditionalGeneration, "qwen3"
    raise ValueError(f"not a Qwen-VL vlm tag: {vlm}")


@dataclass
class QwenVLBackend:
    """ECSO backend for Qwen2.5-VL (qwen25vl) and Qwen3-VL (qwen3vl)."""

    vlm: str
    model_path: str
    temperature: float = 0.0
    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.vlm not in QWEN_VLMS:
            raise ValueError(f"vlm must be one of {sorted(QWEN_VLMS)}, got {self.vlm!r}")

        from transformers import AutoProcessor

        model_cls, self._variant = _resolve_model_cls(self.vlm)
        path = str(Path(self.model_path).expanduser())

        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = model_cls.from_pretrained(
            path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def build_messages(query: str, image_path: str | None) -> list[dict]:
        content: list[dict] = []
        if image_path is not None:
            resolved = Path(image_path).expanduser().resolve()
            content.append({"type": "image", "image": f"file://{resolved}"})
        content.append({"type": "text", "text": query})
        return [{"role": "user", "content": content}]

    def _prepare_inputs(self, messages: list[dict]) -> dict:
        if self._variant == "qwen3":
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if isinstance(inputs, dict):
                inputs.pop("token_type_ids", None)
                return {
                    key: val.to(self.device) if hasattr(val, "to") else val
                    for key, val in inputs.items()
                }
            inputs.pop("token_type_ids", None)
            return inputs.to(self.device)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            has_image = any(
                part.get("type") == "image"
                for msg in messages
                for part in (msg.get("content") or [])
                if isinstance(part, dict)
            )
            if has_image:
                from PIL import Image

                image_path = None
                for msg in messages:
                    for part in msg.get("content") or []:
                        if isinstance(part, dict) and part.get("type") == "image":
                            raw = part["image"]
                            image_path = raw[7:] if str(raw).startswith("file://") else raw
                            break
                if image_path is None:
                    raise RuntimeError("image message present but path not found")
                image = Image.open(image_path).convert("RGB")
                inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            else:
                inputs = self.processor(text=[text], padding=True, return_tensors="pt")

        return {
            key: val.to(self.device) if hasattr(val, "to") else val
            for key, val in inputs.items()
        }

    def generate(
        self,
        query: str,
        *,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> str:
        messages = self.build_messages(query, image_path)
        inputs = self._prepare_inputs(messages)

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
