#!/usr/bin/env python3
"""Read-only Qwen3-VL environment and GPU audit."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

PACKAGES = {
    "numpy": "numpy",
    "pytest": "pytest",
    "ruff": "ruff",
    "Pillow": "PIL",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "qwen-vl-utils": "qwen_vl_utils",
    "peft": "peft",
    "jupyterlab": "jupyterlab",
    "ipykernel": "ipykernel",
}


def status(distribution: str, module: str) -> dict[str, object]:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"importable": importlib.util.find_spec(module) is not None, "version": version}


def main() -> int:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "source_exists": Path("/nas/home/mhlee/vlm-foundation-7days").exists(),
        "data_boundary_exists": Path("/nas/datahub/min").exists(),
        "packages": {name: status(name, module) for name, module in PACKAGES.items()},
    }
    if importlib.util.find_spec("torch"):
        import torch

        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "torch_cuda": torch.version.cuda,
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

