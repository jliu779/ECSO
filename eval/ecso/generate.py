from __future__ import annotations

import argparse

from eval.ecso.backends import SUPPORTED_VLMS, get_backend
from eval.ecso.pipeline import run_ecso
from eval.io import write_jsonl
from eval.manifest import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="ECSO generation from manifest jsonl")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vlm", default="llava15", choices=sorted(SUPPORTED_VLMS))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_base", default=None)
    parser.add_argument("--conv_mode", default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--disable_cache", action="store_true")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_new_tokens_ecso", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = read_manifest(args.manifest, limit=args.limit)
    backend = get_backend(
        args.vlm,
        args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        attn_implementation=args.attn_implementation,
        disable_cache=args.disable_cache,
        use_flash_attn=args.use_flash_attn,
    )
    rows = run_ecso(
        backend,
        records,
        max_new_tokens_direct=args.max_new_tokens,
        max_new_tokens_ecso=args.max_new_tokens_ecso,
    )
    write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()
