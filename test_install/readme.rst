.. _installation_tests:

-----------------
Installation tests

In this folder are general python test to verify correct installation.
Each test is run with `python test_script.py` and can call specific tests in subfolders.

Call these scripts after installation e.g. from (potentially OS dependant) shell script.
An example can be (not that this works on any system, provided all tools are installed)::

python test_install\test_dep.py
python test_install\test_import.py
py.test --nbval profile_demo_rise.ipynb


