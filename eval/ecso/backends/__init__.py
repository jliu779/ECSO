from __future__ import annotations

from typing import TYPE_CHECKING

from eval.ecso.backends.base import ECSOBackend

SUPPORTED_VLMS = frozenset(
    {
        "llava15",
        "llava_next",
        "qwen25vl",
        "qwen3vl",
        "phi4",
        "glm41v",
        "internvl3",
    }
)

if TYPE_CHECKING:
    from eval.ecso.backends.glm41v import Glm41vBackend
    from eval.ecso.backends.llava import LlavaBackend
    from eval.ecso.backends.phi4 import Phi4Backend
    from eval.ecso.backends.qwen_vl import QwenVLBackend


def get_backend(
    vlm: str,
    model_path: str,
    *,
    model_base: str | None = None,
    conv_mode: str = "vicuna_v1",
    temperature: float = 0.0,
    attn_implementation: str = "eager",
    disable_cache: bool = False,
    **kwargs,
) -> ECSOBackend:
    if vlm not in SUPPORTED_VLMS:
        raise ValueError(
            f"unsupported VLM '{vlm}'; supported: {', '.join(sorted(SUPPORTED_VLMS))}"
        )

    if vlm in {"llava15", "llava_next"}:
        from eval.ecso.backends.llava import LlavaBackend

        return LlavaBackend(
            model_path=model_path,
            model_base=model_base,
            conv_mode=conv_mode,
            temperature=temperature,
        )

    if vlm in {"qwen25vl", "qwen3vl"}:
        from eval.ecso.backends.qwen_vl import QwenVLBackend

        return QwenVLBackend(
            vlm=vlm,
            model_path=model_path,
            temperature=temperature,
        )

    if vlm == "phi4":
        from eval.ecso.backends.phi4 import Phi4Backend

        return Phi4Backend(
            model_path=model_path,
            temperature=temperature,
            attn_implementation=attn_implementation,
            disable_cache=disable_cache,
        )

    if vlm == "glm41v":
        from eval.ecso.backends.glm41v import Glm41vBackend

        return Glm41vBackend(
            model_path=model_path,
            temperature=temperature,
            attn_implementation=attn_implementation,
        )

    if vlm == "internvl3":
        from eval.ecso.backends.internvl3 import InternVL3Backend

        return InternVL3Backend(
            model_path=model_path,
            temperature=temperature,
            use_flash_attn=bool(kwargs.get("use_flash_attn", False)),
        )

    raise ValueError(f"unsupported VLM '{vlm}'")


def __getattr__(name: str):
    if name == "LlavaBackend":
        from eval.ecso.backends.llava import LlavaBackend

        return LlavaBackend
    if name == "QwenVLBackend":
        from eval.ecso.backends.qwen_vl import QwenVLBackend

        return QwenVLBackend
    if name == "Phi4Backend":
        from eval.ecso.backends.phi4 import Phi4Backend

        return Phi4Backend
    if name == "Glm41vBackend":
        from eval.ecso.backends.glm41v import Glm41vBackend

        return Glm41vBackend
    if name == "InternVL3Backend":
        from eval.ecso.backends.internvl3 import InternVL3Backend

        return InternVL3Backend
    raise AttributeError(name)


__all__ = [
    "ECSOBackend",
    "LlavaBackend",
    "QwenVLBackend",
    "Phi4Backend",
    "Glm41vBackend",
    "InternVL3Backend",
    "SUPPORTED_VLMS",
    "get_backend",
]
