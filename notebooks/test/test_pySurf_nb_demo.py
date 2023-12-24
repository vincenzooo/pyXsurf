import pytest
from testbook import testbook


@pytest.fixture(scope="module")
def notebook(notebooks_path):
    with testbook(notebooks_path / "pySurf_nb_demo.ipynb", execute=True) as nb:
        yield nb


def test_function(notebook):
    assert "pySurf function" == notebook.ref("function")()


def test_data_function(notebook):
    assert "pySurf functionality data\n" == notebook.ref("data_function")()


def test_common_data_function(notebook):
    assert "common functionality data\n" == notebook.ref("common_data_function")()


def test_nb_test_data(notebook):
    assert "nb test data\n" == notebook.ref("nb_test_data")()
