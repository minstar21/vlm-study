from pathlib import Path

import pytest

from vlm_foundation.config import LabPaths


def test_data_boundary() -> None:
    with pytest.raises(ValueError):
        LabPaths.from_env(data_root="/nas/home/another-user", project_root=Path.cwd())


def test_read_path_rejects_escape() -> None:
    paths = LabPaths.from_env(data_root="/nas/datahub/min/mhlee-course", project_root=Path.cwd())
    with pytest.raises(ValueError):
        paths.read_path("../other-user/file.jpg", must_exist=False)
