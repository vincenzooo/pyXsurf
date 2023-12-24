from importlib_resources import files


def function():
    return "pyProfile function"


def data_function():
    return files("pyXsurf.pyProfile.data").joinpath("pyprofile_data.txt").read_text()


def common_data_function():
    return files("pyXsurf.data").joinpath("common_data.txt").read_text()
