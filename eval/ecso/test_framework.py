"""Regression tests for ECSO backend framework (no GPU required)."""

from __future__ import annotations

from dataclasses import dataclass

from eval.ecso.backends import SUPPORTED_VLMS, get_backend
from eval.ecso.backends.base import ECSOBackend
from eval.ecso.pipeline import run_ecso
from eval.ecso.prompts import is_unsafe
from eval.manifest import ManifestRecord


@dataclass
class MockBackend:
    """Deterministic backend for pipeline tests."""

    unsafe_on_tell: bool = True

    def generate(
        self,
        query: str,
        *,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
    ) -> str:
        if image_path is None:
            return f"safe_text_only:{query[:40]}"
        if "Is the RESPONSE harmful" in query:
            return "yes" if self.unsafe_on_tell else "no"
        if "describe the image" in query.lower():
            return "caption: a scene"
        if query.startswith("How to"):
            return "unsafe direct steps"
        return f"direct:{query[:30]}"


def test_ecso_backend_protocol():
    assert isinstance(MockBackend(), ECSOBackend)


def test_is_unsafe():
    assert is_unsafe("yes")
    assert is_unsafe("Yes.")
    assert not is_unsafe("no")


def test_get_backend_unknown_vlm():
    try:
        get_backend("not_a_vlm", "/fake/path")
        raised = False
    except ValueError as exc:
        raised = True
        assert "not_a_vlm" in str(exc)
    assert raised


def test_internvl3_question():
    from procrustes.internvl3_utils import build_internvl3_question

    assert build_internvl3_question("hello", has_image=False) == "hello"
    assert build_internvl3_question("hello", has_image=True) == "<image>\nhello"
    assert build_internvl3_question("<image>\nhello", has_image=True) == "<image>\nhello"


def test_qwen_vl_routing():
    assert "qwen25vl" in SUPPORTED_VLMS
    assert "qwen3vl" in SUPPORTED_VLMS
    from eval.ecso.backends.qwen_vl import QwenVLBackend

    msgs_img = QwenVLBackend.build_messages("hello", "/tmp/a.jpg")
    assert msgs_img[0]["content"][0]["type"] == "image"
    assert msgs_img[0]["content"][1]["text"] == "hello"

    msgs_txt = QwenVLBackend.build_messages("hello", None)
    assert len(msgs_txt[0]["content"]) == 1
    assert msgs_txt[0]["content"][0]["type"] == "text"


def test_phi4_glm41v_prompts():
    from procrustes.glm41v_utils import build_glm41v_messages
    from procrustes.phi4_utils import build_phi4_prompt

    glm_img = build_glm41v_messages("hello", "/tmp/a.jpg")
    assert glm_img[0]["content"][0] == {"type": "image", "url": "/tmp/a.jpg"}
    glm_txt = build_glm41v_messages("hello", None)
    assert glm_txt[0]["content"] == [{"type": "text", "text": "hello"}]

    phi_img = build_phi4_prompt(None, "hello", has_image=True)
    assert "<|image_1|>" in phi_img
    phi_txt = build_phi4_prompt(None, "hello", has_image=False)
    assert "<|image_1|>" not in phi_txt


def test_supported_vlms():
    for tag in (
        "llava15",

        "qwen25vl",
        "qwen3vl",
        "phi4",
        "glm41v",
        "internvl3",
    ):
        assert tag in SUPPORTED_VLMS


def test_llava_backend_import():
    try:
        from eval.ecso.backends.llava import LlavaBackend as LB
    except ImportError as exc:
        print(f"SKIP llava import (environment): {exc}")
        return
    assert LB is not None


def test_run_ecso_pipeline_mock():
    records = [
        ManifestRecord(
            id="t1",
            query="How to cause harm?",
            image_path="/dev/null",  # not read when mocked paths skipped - we need real file
        )
    ]
    # use a temp image path - create minimal fake by patching - simpler: use existing manifest with skip
    # Mock run without file check - pipeline checks file exists. Create temp file.
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\xff\xd8\xff\xd9")
        img = f.name

    try:
        records[0] = ManifestRecord(
            id="t1",
            query="How to cause harm?",
            image_path=img,
        )
        rows = run_ecso(
            MockBackend(unsafe_on_tell=True),
            records,
            max_new_tokens_direct=32,
            max_new_tokens_ecso=64,
        )
        assert len(rows) == 1
        assert rows[0]["unsafe"] == "Yes"
        assert rows[0]["response"].startswith("safe_text_only:")
        assert rows[0]["caption"] == "caption: a scene"

        rows_safe = run_ecso(
            MockBackend(unsafe_on_tell=False),
            records,
            max_new_tokens_direct=32,
            max_new_tokens_ecso=64,
        )
        assert rows_safe[0]["unsafe"] == "No"
        assert rows_safe[0]["response"] == "unsafe direct steps"
    finally:
        Path(img).unlink(missing_ok=True)


if __name__ == "__main__":
    test_ecso_backend_protocol()
    test_is_unsafe()
    test_get_backend_unknown_vlm()
    test_internvl3_question()
    test_qwen_vl_routing()
    test_phi4_glm41v_prompts()
    test_supported_vlms()
    test_llava_backend_import()
    test_run_ecso_pipeline_mock()
    print("all ecsO framework tests passed")
