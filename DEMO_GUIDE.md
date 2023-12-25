1. Clone the repository

```
git clone --branch new-pyproject-additions-plus git@github.com:robeyes/pyXsurf.git
cd pyXsurf
```

2. Install the package for development

[comment]: <> (TODO: what is the function of `.[dev]`, rather than `.`?)
```
pip install -e .[dev]
```

3. I have placed multiple demo files to get a sense of the usage in different contexts.
I think it should cover all the usecases that we have discussed.

## `test/units/pySurf/test_demo_module.py`

This is a simple demo of the usage of the `pySurf` module.
The file itself a is pytest test module, which can be tested with:
```
pytest test/units/pySurf/
```

- `test_test_data`: is an example to show how to read some sample data for testing.

## `pyXsurf/pySurf/demo_module.py`

This is a demo implementation of couple of demo functions.

- `data_function`: example of how to read data from file in the `pyXsurf/pySurf/data`. Note that these will functionality data files, and not the data files for testing. ie. the implementation wouldn't work without this data.

- `common_data_function`: example of how to read a data file in `pyXsurf/data`. These data files will be used by multiple modules like `pySurf` and `pyProfile` etc. So, they are located in a the upper directory.

## `test/units/pyProfile/test_demo_module.py` and `pyXsurf/pyProfile/demo_module.py`

These are the same as the `pySurf` demo files, but for the `pyProfile` module.

## `notebooks/pySurf_nb_demo.ipynb`

This is a jupyter notebook demo of the `pySurf` module. It is a good place to start to get a sense of the usage.

- `nb_test_data`: is an example to show how to read some sample data for notebooks.
- `data_function` and `common_data_function` - shows how to use the functions from the `pySurf` module, that will read the data files with in the package.

## `notebooks/test/test_pySurf_nb_demo.py`

This is a pytest test module for the `pySurf_nb_demo.ipynb` notebook. It is an example of how to test a notebook.

Can be tested with:
```
pytest notebooks/test/
```

## `test/scripts/pyGeo3D/test_standalone_demo.py`

This is normal script / standalone script to run a standalone module in the `pyGeo3D` module: `pyXsurf/pyGeo3D/standalone_demo.py`
Right now the way to run these scripts is run the script directly from the command line.

```
python test/scripts/pyGeo3D/test_standalone_demo.py
```

4. Guide to updating this pull request:

```
git clone --branch new-pyproject-additions-plus git@github.com:robeyes/pyXsurf.git
cd pyXsurf
```

- Make changes and add files as needed.
- Commit the changes to the local repository
- Push the changes to the remote repository as usual with

```
git push --set-upstream origin new-pyproject-additions-plus
```
