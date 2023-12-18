pyXsurf
-------

Python library for X-Ray Optics, Metrology Data Analysis and Telescopes Design.

This library starts from code collected during many years of work on surface metrology and X-ray optics and
wants to offer to the community an extensible tool to perfom common operations on surface and profile data,
together with a set of tools to perform typical operations on data, in particular related to X-ray optics.

Installation
============

.. code:: bash

    pip install pyXsurf

Uninstalling
============

Project can be uninstalled running ``pip uninstall pyXsurf`` or equivalent.

How to use
==========

All modules are packaged into `pyXsurf` package, you can import them with something like:

.. code:: python

    from pyXsurf.pySurf.data2D import data2D


Additional resources for usage guidance
=======================================

- **docs** `official documentation <https://pyxsurf.readthedocs.io>`
- **tests** contains a collection of tests and examples.
- **notebooks** contains a collection of Jupyter notebooks with examples and tutorials.

Modules
=======

A basic description of the different modules is:

- **dataIO** Generic routines for accessing and manipulating data and files.
- **plotting** Plotting functions for pySurf data.
- **pyGeo3D** Functions for geometry in space (lines and planes).
- **pyProfile** Equivalent of pySurf acting on Profiles.
- **pySurf** Functions and classes acting on 3D points or surfaces.
- **thermal** Functions for modelling of thermal forming of glass.


Contributing
============

Please report bugs or feature requests, missing documentation,
or open a issue on github https://github.com/vincenzooo/pyXsurf/issues.

Acknowledgements
================

The code in this library is the result of many years of work.
Many colleagues from my current or former Institutions contributed
directly and indirectly with exchange of code, ideas, data and good time.
The ongoing improvements to this project are funded by INAF “Bando per innovazione tecnologica”,
which the author also thanks for the supportive and stimulating working environment.

.. Data used for development and in examples are courtesy of ..

Citation
========

.. image:: https://zenodo.org/badge/165474659.svg
   :target: https://zenodo.org/badge/latestdoi/165474659

License
=======

This project is Copyright (c) Vincenzo Cotroneo and licensed under
the terms of the BSD 3-Clause license. See the licenses folder for
more information.


Author
======

Vincenzo Cotroneo vincenzo.cotroneo@inaf.it
