#!/usr/bin/env python3
"""Run cache-only Qwen3-VL inference on one explicitly selected NAS image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vlm_foundation.config import LabPaths  # noqa: E402
from vlm_foundation.grounding import grounding_prompt  # noqa: E402
from vlm_foundation.qwen_adapter import generate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Directory under /nas/datahub/min")
    parser.add_argument("--image", required=True, help="Image path relative to data root")
    parser.add_argument("--prompt")
    parser.add_argument("--ground", nargs="+", metavar="LABEL")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly allow from_pretrained to download missing model files",
    )
    args = parser.parse_args()
    if bool(args.prompt) == bool(args.ground):
        parser.error("provide exactly one of --prompt or --ground")

    paths = LabPaths.from_env(data_root=args.data_root)
    image_path = paths.read_path(args.image)
    prompt = args.prompt or grounding_prompt(args.ground)
    result = generate(
        image_path,
        prompt,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        allow_download=args.allow_download,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

