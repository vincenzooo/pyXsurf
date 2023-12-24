import pyXsurf.pySurf.demo_module as psdm


def test_function():
    assert "pySurf function" == psdm.function()


def test_data_function():
    assert "pySurf functionality data\n" == psdm.data_function()


def test_common_data_function():
    assert "common functionality data\n" == psdm.common_data_function()
