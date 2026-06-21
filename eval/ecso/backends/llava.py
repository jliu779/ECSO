from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image


def _is_hf_format(model_path: str) -> bool:
    """Return True if the checkpoint uses the HF LlavaForConditionalGeneration format."""
    import json
    cfg = Path(model_path) / "config.json"
    if cfg.is_file():
        with open(cfg) as f:
            return json.load(f).get("model_type") == "llava"
    return False


# ---------------------------------------------------------------------------
# HF-format backend (llava-hf/llava-1.5-7b-hf)
# ---------------------------------------------------------------------------

@dataclass
class _HFLlavaBackend:
    model_path: str
    temperature: float = 0.0

    def __post_init__(self) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        path = str(Path(self.model_path).expanduser())
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(
        self,
        query: str,
        *,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> str:
        content: list[dict] = []
        if image_path is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": query})

        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
        )

        image = Image.open(image_path).convert("RGB") if image_path else None
        if image is not None:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs: dict = {"max_new_tokens": max_new_tokens, "repetition_penalty": 1.1}
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        return self.processor.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()


# ---------------------------------------------------------------------------
# Original-format backend (liuhaotian/llava-v1.5-*)
# ---------------------------------------------------------------------------

@dataclass
class _OriginalLlavaBackend:
    model_path: str
    model_base: str | None = None
    conv_mode: str = "vicuna_v1"
    temperature: float = 0.0
    top_p: float | None = None
    num_beams: int = 1

    def __post_init__(self) -> None:
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        # keep references for generate()
        from llava import constants, conversation as conv_mod, mm_utils
        self._constants = constants
        self._conv_mod = conv_mod
        self._mm_utils = mm_utils

        disable_torch_init()
        model_path = str(Path(self.model_path).expanduser())
        model_name = get_model_name_from_path(model_path)
        if (
            "plain" in model_name
            and "finetune" not in model_name.lower()
            and "mmtag" not in self.conv_mode
        ):
            self.conv_mode = self.conv_mode + "_mmtag"
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, self.model_base, model_name
        )
        self.model.eval()

    def _prepare_image_tensor(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self._mm_utils.process_images([image], self.image_processor, self.model.config)[0]

    def generate(
        self,
        query: str,
        *,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> str:
        c = self._constants
        conv_templates = self._conv_mod.conv_templates
        SeparatorStyle = self._conv_mod.SeparatorStyle
        tokenizer_image_token = self._mm_utils.tokenizer_image_token
        IMAGE_TOKEN_INDEX = self._mm_utils.IMAGE_TOKEN_INDEX

        prompt = query
        image_tensor = None
        if image_path is not None:
            image_tensor = self._prepare_image_tensor(image_path)
            if self.model.config.mm_use_im_start_end:
                prompt = (
                    c.DEFAULT_IM_START_TOKEN + c.DEFAULT_IMAGE_TOKEN
                    + c.DEFAULT_IM_END_TOKEN + "\n" + prompt
                )
            else:
                prompt = c.DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda() if image_tensor is not None else None,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()
        return outputs


# ---------------------------------------------------------------------------
# Public facade — auto-detects checkpoint format
# ---------------------------------------------------------------------------

@dataclass
class LlavaBackend:
    model_path: str
    model_base: str | None = None
    conv_mode: str = "vicuna_v1"
    temperature: float = 0.0
    top_p: float | None = None
    num_beams: int = 1
    _backend: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        path = str(Path(self.model_path).expanduser())
        if _is_hf_format(path):
            self._backend = _HFLlavaBackend(
                model_path=path,
                temperature=self.temperature,
            )
        else:
            self._backend = _OriginalLlavaBackend(
                model_path=path,
                model_base=self.model_base,
                conv_mode=self.conv_mode,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
            )

    def generate(
        self,
        query: str,
        *,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> str:
        return self._backend.generate(query, image_path=image_path, max_new_tokens=max_new_tokens)
