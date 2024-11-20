pyXsurf
-------------------------

Python library for X-Ray Optics, Metrology Data Analysis and Telescopes
Design. 

This library starts from code collected during many years of work on surface metrology and X-ray optics and wants to offer to the community an extensible tool to perfom common operations on surface and profile data, together with a set of tools to perform typical operations on data, in particular related to X-ray optics.

News
=======
2023/07/15 Starting implementing major changes towards common standards and best-practices. 


Installation Process
=====================

It is the generic procedure, see links for the details of different environments. It is described how to prepare the system (update packages and optionally create a dedicated environment), and installing and testing the library.

Optional Steps 
^^^^^^^^^^^^^^^^

It is recommended to update ``pip``, to the most recent version. You can run ``python -m pip install --upgrade pip`` to upgrade pip, or install it as described in https://pip.pypa.io/en/stable/installation.

If you want to try the package without modifying your current environment, it can be useful to work in a separate space, creating a new environment (e.g. this is done in ``conda`` with  ``conda env create <envname>`` and ``conda activate <envname>``, you can check https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/ for the equivalent `venv` commands). In any case, make sure you are working in the right environment.

You should now be ready to install ``pyXsurf`` and all dependencies.

Installation
^^^^^^^^^^^^^

You can get the most updated version by cloning the source code from github and installing from there. 

To do this:

1. use the ``Code`` button at top of page, or clone from command-lineç
``git clone https://github.com/vincenzooo/pyXSurf.git``. 

2. move to the folder with the code and call from command line ``pip install .`` (equivalent to ``python setup.py install``). 
This will perform a "regular" installation  (meaning that the code is copied to ``site-packages`` folders, which holds all installed Python libraries, and made accessible to your system).

If instead you plan to modify pyXsurf code, or want a non-permanente installation, you can install in developer mode ``pip install -e .`` (also, read the additional information on the :ref:``README_developers`` page).


Uninstalling
------------

Code can be uninstalled calling ``pip uninstall pyXsurf`` from a command prompt. N.B.: previous version can also be uninstalled from source code folder calling directly the setup file ``python setup.py develop  -u``, this is no more advised (if you installed as developer, deinstallation is expected to work only with most recent pip versions, at least >=19.1.1, otherwise it should be enough to delete the folder).

How to use
==========

At this point you can open Python and try ``import pySurf`` (`pySurf` is the Python module responsible of surface data). If this
works without errors, you can now import and use the different modules (see description below) which are part of the library, e.g., with:

.. code:: python

    from pySurf.data2D_class import Data2D

or try the library in a Jupyter notebook, you can start from the example `pySurf_demo.ipynb`, in the install folder.

In addition to the `official documentation <https://pyxsurf.readthedocs.io>`_ , you can find examples and data in a ``test`` subfolder of each
module and at the end of module files. Functions are usually documented with docstrings (but these might be in the wrong place, please be patient and look well in the code). 
There is a lot more, but this is a big work-in-progress, and they are not easily accessible yet, please read below for details, or write an email or an issue.

Modules
========

A basic description of the different modules is (N.B.: some of these functions have been temporarily moved out of the codebase to simplify the reimplementation, in case these are needed, please send an email):

-  **dataIO** Generic routines for accessing and manipulating data and files.

-  **notebooks** Jupyter notebooks, not necessarily related to the libraries, include test and experiments on python.

-  **plotting** Plotting functions for pySurf data.

-  **pyGeo3D** Functions for geometry in space (lines and planes).

-  **pyProfile** Equivalent of pySurf acting on Profiles.

-  **pySurf** Functions and classes acting on 3D points or surfaces.

-  **thermal** Functions for modelling of thermal forming of glass.



Status of the library and additional resources
===============================================

The main part of the library is well defined and it works well. I am
constantly adding functions when I find they are needed during my daily work. 

I have tried the installation according to the above instructions on a few computers and it worked smoothly. You are very welcome to help signaling any problem or missing information, please see :ref:`Contributing` below.

There are many examples scattered around in different folders ``Demo``, ``Tutorial``, ``Test``, etc. , some of which are not precisely documented, or even finished. I am still trying to give an organic structure to documentation (it might takes time, as I am completely inesperienced). Until then, the best thing is to dig for notebooks .ipynb or .py files in these folders, or look in the single .py files from code library, which often contain small test functions or examples.

There is not a real API documentation, as I am still trying to understand the standards, but there are a good amount of docstrings (even if they might be in the wrong place or format, sorry for that). Please use Python introspection (``?``, ``??``, autocompletion, etc.), and some patience, to find more. If you are a documentation expert I could really use some advice on that.

There are developer branches on github, which have extra functionalities, at the moment under development (they are quite messy, and some of the features might have been forgotten, so feel free to ask if you are looking for something in particular). These can be installed from code in the same way as the main branch and should work equally (just have more unfinished stuff). For git beginners, the only thing you need to do differently, is to switch branch, e.g. ``git checkout documentation`` (where ``documentation`` is the name of the branch) after cloning the repository and before running the setup. Please check developers notes for a list of active branches and their features and for more details. Also, if you plan to make changes to the code and want to keep the changes automatically in synch, remember to install the code as "developer" (as explained above). Otherwise you can still make changes to the installed code (in site-packages), but you will need to reimport after every change.

See developer notes :doc:`README_developers.rst` for a detailed status of developement, how to access more recent features and last status of documentation (on developer brach), especially if you think you can help.
Expecially installation and release mechanism, are in phase of improvement, as well as documentation.

Contributing
============

Please report bugs or feature requests, missing documentation, or open a
issue on github https://github.com/vincenzooo/pyXsurf/issues.

Expecially appreciated is if you can provide templates, examples or
hints on how to handle, documentation (Sphinx), packaging, continuous
integration (Github).

Please check :ref:`developersnotes` for the status of the
development, or if are willing to help in any way.

Aknowledgements
============

The code in this library is the result of many years of work. Many colleagues from my current or former Institutions contributed directly and indirectly with exchange of code, ideas, data and good time.
The ongoing improvements to this project are funded by INAF “Bando per innovazione tecnologica”, which the author also thanks for the supportive and stimulating working environment.

.. Data used for development and in examples are courtesy of .. 

Citation
========

.. image:: https://zenodo.org/badge/165474659.svg
   :target: https://zenodo.org/badge/latestdoi/165474659

License
=========

This project is Copyright (c) Vincenzo Cotroneo and licensed under
the terms of the BSD 3-Clause license. See the licenses folder for
more information.


Author
=======

Vincenzo Cotroneo vincenzo.cotroneo@inaf.it
