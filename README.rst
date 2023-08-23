pyXsurf (formerly pyXTel)
=========================

Python library for X-Ray Optics, Metrology Data Analysis and Telescopes
Design. 

This library starts from code collected during many years of work on surface metrology and X-ray optics and wants to offer to the community an extensible tool to perfom common operations on surface and profile data.


Changes
-------
2023/07/15 Starting implementing major changes towards common standards and best-practices. 
2022/09/06 Created new branch ``pyXsurf_nbdev`` for experimenting with
2021/07/21 Upgraded installation mechanism.
2020/11/09 Changed repository name from ``pyXTel`` to
``pyXSurf``.

Installation
------------

Last stable release can be installed from command line with:

``pip install pyXsurf==1.6.1`` 
   
However this is not advised, as the most updated versions can contain bug fixes and updated functionalities.
To install these you should install from code, which can be done quite easily. This can also be useful if you plan to make changes to the code.

In this case, first you need to download or clone the source code from github. You can use the ``Code`` button at top of page, or git clone from command-line
``git clone https://github.com/vincenzooo/pyXSurf.git``. 

At this point you have two options (before proceding, it can be advisable to update your environment, and especially ``pip``, to the most recent version):

1. Python installer (suggested for basic users): move to the folder with the code and call ``python setup.py install``. This will perform a "regular" installation  (meaning that the code is copied to ``site-packages`` folders, which holds all installed Python libraries, and made accessible to your system).

2. developer: as in option 1, just use ``pip install -e .``or ``pip install -e '.[dev]'``. Doing this, the library will be installed from current position (no local copy), any change to the code in this folder will be immediately available in the imported library (the second command, with the ``[dev]``\ option, will install also additional files with tests and tools).

Uninstalling
------------

Code can be uninstalled callig ``pip uninstall pyXsurf``, or from source code folder calling directly the setup file ``python setup.py develop  -u`` (if you installed as developer, deinstallation is expected to work only with most recent pip versions, at least >=19.1.1, otherwise it should be enough to delete the folder).

How to use
----------

At this point you can open Python and try ``import pySurf``, if this
works without errors, you can now import and use the different modules (see description below) which are part of the library, e.g., with:

.. code:: python

   from pySurf.data2D import data2D
   
Some examples and data can be found in a ``test`` subfolder of each
module and at the end of module files. Functions are usually documented with docstrings (but these might be in the wrong place, please be patient and look well in the code). 
There is a lot more, but this is a big work-in-progress, and they are not easily accessible yet, please read below for details, or write an email or an issue.

Modules
-------

.. image:: /resources/wip.png
   :class: wip-icon
   
A basic description of the different modules is (N.B.: some of these functions have been temporarily moved out of the codebase to simplify the reimplementation, in case these are needed, please send an email):

-  **dataIO** Generic routines for accessing and manipulating data and
   files.

-  **notebooks** Jupyter notebooks, not necessarily related to the
   libraries, include test and experiments on python.

-  **plotting** Plotting functions for pySurf data.

-  **pyGeo3D** Functions for geometry in space (lines and planes).

-  **pyProfile** Equivalent of pySurf acting on Profiles.

-  **pySurf** Functions and classes acting on 3D points or surfaces.

-  **thermal** Functions for modelling of thermal forming of glass.



Status of the library and additional resources
-----------------------------------------------

The main part of the library is well defined and it works well. I am
constantly adding functions when I find they are needed during my daily work. 

I have tried the installation according to the above instructions on a few computers and it worked smoothly. You are very welcome to help signaling any problem or missing information, please see :ref:`Contributing` below.

There are many examples scattered around in different folders ``Demo``, ``Tutorial``, ``Test``, etc. , some of which are not precisely documented, or even finished. I am still trying to give an organic structure to documentation (it might takes time, as I am completely inesperienced). Until then, the best thing is to dig for notebooks .ipynb or .py files in these folders, or look in the single .py files from code library, which often contain small test functions or examples.

There is not a real API documentation, as I am still trying to understand the standards, but there are a good amount of docstrings (even if they might be in the wrong place or format, sorry for that). Please use Python introspection (``?``, ``??``, autocompletion, etc.), and some patience, to find more. If you are a documentation expert I could really use some advice on that.

There are developer branches on github, which have extra functionalities, at the moment under development (they are quite messy, and some of the features might have been forgotten, so feel free to ask if you are looking for something in particular). These can be installed from code in the same way as the main branch and should work equally (just have more unfinished stuff). For git beginners, the only thing you need to do differently, is to switch branch, e.g. ``git checkout documentation`` (where ``documentation`` is the name of the branch) after cloning the repository and before running the setup. Please check developers notes for a list of active branches and their features and for more details. Also, if you plan to make changes to the code and want to keep the changes automatically in synch, remember to install the code as "developer" (as explained above). Otherwise you can still make changes to the installed code (in site-packages), but you will need to reimport after every change.

See developer notes :ref:`developersnotes` for a detailed status of developement, how to access more recent features and last status of documentation (on developer brach), especially if you think you can help.
Expecially installation and release mechanism, are in phase of improvement, as well as documentation.

.. _contributing

Contributing
------------

Please report bugs or feature requests, missing documentation, or open a
issue on github https://github.com/vincenzooo/pyXsurf/issues.

Expecially appreciated is if you can provide templates, examples or
hints on how to handle, documentation (Sphinx), packaging, continuous
integration (Github).

Please check :ref:``README_developers`` for the status of the
development, or if are willing to help in any way.

Citation
--------

.. image:: https://zenodo.org/badge/165474659.svg
   :target: https://zenodo.org/badge/latestdoi/165474659

License
-------

This project is Copyright (c) Vincenzo Cotroneo and licensed under
the terms of the BSD 3-Clause license. Parts of this package are based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.


Author
------

Vincenzo Cotroneo vincenzo.cotroneo@inaf.it
