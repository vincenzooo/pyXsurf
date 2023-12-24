import pyXsurf.pyProfile.demo_module as ppdm


def test_function():
    assert "pyProfile function" == ppdm.function()


def test_data_function():
    assert "pyProfile functionality data\n" == ppdm.data_function()


def test_common_data_function():
    assert "common functionality data\n" == ppdm.common_data_function()
