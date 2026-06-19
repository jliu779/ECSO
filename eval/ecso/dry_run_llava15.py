#!/usr/bin/env python3
"""Dry-run llava15 ECSO without loading weights (checks wiring + mock pipeline)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run llava15 ECSO wiring")
    parser.add_argument(
        "--manifest",
        default="manifests/dryrun_llava15.jsonl",
        help="Manifest jsonl (default: manifests/dryrun_llava15.jsonl)",
    )
    parser.add_argument(
        "--model_path",
        default="models/llava-1.5-7b",
        help="Subject model path (validated only, not loaded)",
    )
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))

    from eval.ecso.backends import SUPPORTED_VLMS, get_backend
    from eval.ecso.pipeline import run_ecso
    from eval.ecso.test_framework import MockBackend
    from eval.manifest import read_manifest

    print("=== llava15 ECSO dry run ===")
    print(f"ROOT={root}")
    print(f"SUPPORTED_VLMS={sorted(SUPPORTED_VLMS)}")

    manifest = root / args.manifest
    if not manifest.is_file():
        print(f"FAIL: manifest missing: {manifest}")
        return 1
    records = read_manifest(manifest, limit=args.limit)
    print(f"manifest: {manifest} ({len(records)} record(s))")
    for rec in records:
        img = Path(rec.image_path)
        ok = img.is_file()
        print(f"  id={rec.id} image={'OK' if ok else 'MISSING'} {rec.image_path}")
        if not ok:
            return 1

    if "llava15" not in SUPPORTED_VLMS:
        print("FAIL: llava15 not in SUPPORTED_VLMS")
        return 1
    print("get_backend routing: llava15 registered")

    try:
        get_backend("llava15", args.model_path)
        print("get_backend(llava15): weights loaded (full run ready)")
    except ImportError as exc:
        print(f"get_backend(llava15): skip weight load — llava import failed ({exc})")
    except Exception as exc:
        print(f"get_backend(llava15): skip weight load — {type(exc).__name__}: {exc}")

    rows = run_ecso(
        MockBackend(unsafe_on_tell=False),
        records,
        max_new_tokens_direct=32,
        max_new_tokens_ecso=64,
    )
    out = root / "outputs/dryrun_llava15/mock_ecso.jsonl"
    from eval.io import write_jsonl

    write_jsonl(out, rows)
    print(f"mock ECSO output: {out}")
    print(f"  response preview: {rows[0]['response'][:80]!r}")

    runner = root / "runners/run_ecso_full.sh"
    r = subprocess.run(["bash", "-n", str(runner)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL bash -n {runner}: {r.stderr}")
        return 1
    print(f"runner syntax: OK ({runner})")

    gen_help = subprocess.run(
        [sys.executable, str(root / "eval/ecso/generate.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(root),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(root)},
    )
    if gen_help.returncode != 0:
        print(f"FAIL generate.py --help: {gen_help.stderr}")
        return 1
    print("generate.py --help: OK")

    print("To run real ECSO (requires GPU + MODEL_PATH):")
    for vlm in ("llava15", "qwen25vl", "qwen3vl"):
        print(
            f"  VLM={vlm} MODEL_PATH=/path/to/model LIMIT=1 SKIP_JUDGE=1 \\\n"
            "    bash runners/run_ecso_full.sh"
        )
    print("=== dry run passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
