from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture(scope="module")
def test_path(project_root: Path):
    return project_root / "test"
