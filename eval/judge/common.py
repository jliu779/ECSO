from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import json


def load_judge_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except ImportError:
            pass
    return json.loads(text)


def extract_yes_no(text: str) -> str | None:
    lowered = text.strip().lower()
    if re.match(r"^yes\b", lowered):
        return "yes"
    if re.match(r"^no\b", lowered):
        return "no"
    return None


class JudgeModel:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model_path = cfg.get("judge_model_path") or ""
        self._model = None
        self._tokenizer = None

    @property
    def enabled(self) -> bool:
        return bool(self.model_path)

    def _ensure_loaded(self) -> None:
        if self._model is not None or not self.enabled:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def generate(self, prompt: str) -> str:
        if not self.enabled:
            return ""
        self._ensure_loaded()
        import torch

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.get("max_new_tokens", 64)),
                do_sample=False,
            )
        gen = out[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(gen, skip_special_tokens=True).strip()
