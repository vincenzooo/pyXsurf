import pytest
from testbook import testbook


@pytest.fixture(scope="module")
def notebook(notebooks_path, monkeymodule):
    nb_dir = notebooks_path / "nb_demo"
    # Change current working directory to the notebooks directory
    # so that the notebook can find the data files
    monkeymodule.chdir(nb_dir)
    with testbook(nb_dir / "pySurf_nb_demo.ipynb", execute=True) as nb:
        yield nb


def test_function(notebook):
    assert "pySurf function" == notebook.ref("function")()


def test_data_function(notebook):
    assert "pySurf functionality data\n" == notebook.ref("data_function")()


def test_common_data_function(notebook):
    assert "common functionality data\n" == notebook.ref("common_data_function")()


def test_nb_test_data(notebook):
    assert "nb test data\n" == notebook.ref("nb_test_data")()
