from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


@dataclass
class LlavaBackend:
    model_path: str
    model_base: str | None = None
    conv_mode: str = "vicuna_v1"
    temperature: float = 0.0
    top_p: float | None = None
    num_beams: int = 1

    def __post_init__(self) -> None:
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
        return process_images([image], self.image_processor, self.model.config)[0]

    def generate(
        self,
        query: str,
        *,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> str:
        prompt = query
        image_tensor = None
        if image_path is not None:
            image_tensor = self._prepare_image_tensor(image_path)
            if self.model.config.mm_use_im_start_end:
                prompt = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
                )
            else:
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

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
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()
        return outputs
