from importlib_resources import as_file, files


def function():
    return "pySurf function"


def data_function():
    return files("pyXsurf.pySurf.data").joinpath("pysurf_data.txt").read_text()


def common_data_function():
    data_file = files("pyXsurf.data").joinpath("common_data.txt")
    with as_file(data_file) as data_file:
        return data_file.read_text()
