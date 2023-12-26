from importlib_resources import as_file, files

"""See analogous functions in pyProfile."""


def function():
    return "pySurf function"


def data_function():
    return files("pySurf.data").joinpath("pysurf_data.txt").read_text()


def common_data_function():
    data_file = files("data").joinpath("common_data.txt")
    with as_file(data_file) as data_file:
        return data_file.read_text()


def image_generate_file(destination_file):
    # This is a dummy function that resembes writing an image file to disk.
    with open(destination_file, "wb") as fh:
        fh.write(b"Hello World!")
