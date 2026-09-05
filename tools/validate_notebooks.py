#!/usr/bin/env python3
"""Compile and execute notebook code cells without model loading or training."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    os.environ["MPLBACKEND"] = "Agg"
    original = Path.cwd()
    os.chdir(ROOT)
    try:
        notebooks = sorted((ROOT / "notebooks").glob("**/*.ipynb"))
        for path in notebooks:
            payload = json.loads(path.read_text(encoding="utf-8"))
            namespace: dict[str, object] = {"__name__": "__notebook__"}
            for index, cell in enumerate(payload["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                exec(compile(source, f"{path}:cell-{index}", "exec"), namespace)
            print(f"PASS {path.relative_to(ROOT)}")
    finally:
        os.chdir(original)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
