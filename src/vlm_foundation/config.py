"""Strict user-owned source and NAS data boundaries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path("/nas/home/mhlee/vlm-foundation-7days")
NAS_DATA_BOUNDARY = Path("/nas/datahub/min")
DEFAULT_DATA_ROOT = NAS_DATA_BOUNDARY / "vlm-foundation-7days"


def _within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return PROJECT_ROOT


@dataclass(frozen=True)
class LabPaths:
    project_root: Path
    data_root: Path
    artifact_root: Path

    @classmethod
    def from_env(
        cls,
        data_root: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> LabPaths:
        project = Path(project_root or find_project_root()).expanduser().resolve()
        data = Path(
            data_root or os.environ.get("VLM_DATA_ROOT") or DEFAULT_DATA_ROOT
        ).expanduser().resolve()
        boundary = NAS_DATA_BOUNDARY.resolve()
        if not _within(data, boundary):
            raise ValueError(f"data_root must stay under {boundary}, got {data}")
        return cls(project, data, data / "artifacts")

    def read_path(self, relative: str | Path, *, must_exist: bool = True) -> Path:
        candidate = (self.data_root / relative).resolve()
        if not _within(candidate, self.data_root):
            raise ValueError(f"path escapes data_root: {relative}")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(
                f"Data not found: {candidate}. Set VLM_DATA_ROOT or pass --data-root."
            )
        return candidate

    def artifact_path(self, relative: str | Path) -> Path:
        candidate = (self.artifact_root / relative).resolve()
        if not _within(candidate, self.artifact_root):
            raise ValueError(f"path escapes artifact_root: {relative}")
        return candidate
