from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def examples_path(project_root: Path):
    return project_root / "docs" / "source" / "examples"


@pytest.fixture(scope="module")
def test_docs_examples_path(test_path: Path):
    return test_path / "docs_examples"
