import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def notebooks_path(project_root):
    notebooks_dir = project_root / "docs" / "source" / "notebooks"
    with patch.dict(os.environ, {"PYXSURF_NOTEBOOKS_PATH": str(notebooks_dir)}):
        yield notebooks_dir


@pytest.fixture(scope="module")
def test_docs_notebooks_path(test_path):
    return test_path / "docs_notebooks"
