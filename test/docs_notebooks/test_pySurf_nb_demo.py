from pathlib import Path

import pytest
from testbook import testbook
from testbook.client import TestbookNotebookClient


@pytest.fixture(scope="module")
def notebook(notebooks_path: Path):
    nb_dir = notebooks_path / "nb_demo"
    with testbook(
        nb_dir / "pySurf_nb_demo.ipynb",
        execute=True,
        # Set the current working directory to the notebook directory
        # so that it can find the test data with in that directory.
        resources={"metadata": {"path": nb_dir}},
    ) as nb:
        yield nb


def test_function(notebook: TestbookNotebookClient):
    assert "pySurf function" == notebook.ref("function")()


def test_data_function(notebook: TestbookNotebookClient):
    assert "pySurf functionality data\n" == notebook.ref("data_function")()


def test_common_data_function(notebook: TestbookNotebookClient):
    assert "common functionality data\n" == notebook.ref("common_data_function")()


def test_nb_test_data(notebook: TestbookNotebookClient):
    assert "nb test data\n" == notebook.ref("nb_test_data")()
