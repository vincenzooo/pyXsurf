pyXsurf (formerly pyXTel)
=========================

Python library for X-Ray Optics, Metrology Data Analysis and Telescopes
Design. 

2021/07/21 Upgraded installation mechanism.
2020/11/09 Changed repository name from ``pyXTel`` to
``pyXSurf``.

Installation
------------

Download or fork the project. You can use git clone command by:
``git clone https://github.com/vincenzooo/pyXSurf.git`` or the button
``download`` at top of page.
At this point you have two options:

1) Python installer (suggested): move to the folder with the code and call
``python setup.py install``
2) Manual (developer): put the folder with code in a path accessible to 
python on your system (generally this means it must be in the system path) 
and start using it. In this case you must install
all dependencies by yourself.

For users new to Python, the library was developed on anaconda
distribution. Any other distribution should be equivalent, however I
never had any problem with Anaconda and find it very easy to work with,
so I can only recommend it. Same way, I used for developement
notepad++/IPython/jupyter on Windows; Spyder; VScode and feel to
recommend any of these. Of course any other environment or editor works
same way.

Status of the library
--------------------------------

The main part of the library is well defined and it works well. I am
constantly adding functions when I find they are needed during my daily
work. 

I have tried the installation on a few computers and it worked smoothly by `setup.py`. You are very welcome to help signaling any problem or missing information, please see Contributing below.

If everything worked well with the installation, there
are a decent number of tutorial and examples, but they are quite scattered around in
folders ``Demo``, ``Tutorial``, ``Test``, etc. 

You should be able to access information by usual python introspection (``?``, ``??``, autocompletion, etc.).

See developer notes for a detailed status of developement, how to access more recent features and last status of documentation (on developer brach), especially if you think you can help.
Expecially installation and release mechanism, are in phase of improvement, as well as documentation.

Contributing
--------------------------------

Please report bugs or feature requests, missing documentation, or open a issue on github.

Expecially appreciated is if you can provide templates, examples or hints on how to handle, documentation (Sphinx), packaging, continuous integration (Github).

Please check developers notes for the status of the development, or if you think you can help in any way. 


Modules
-------

A basic description of the different modules is: 

* **dataIO** Generic routines for accessing and manipulating data and files. 

* **notebooks**  Jupyter notebooks, not necessarily related to the libraries, include test and experiments on python. 

* **plotting** Plotting functions for pySurf data. 

* **pyGeo3D** Functions for geometry in space (lines and planes). 

* **pyProfile** Equivalent of pySurf acting on Profiles. 

* **pySurf** Functions and classes acting on 3D points or surfaces. 

* **thermal** Functions for modelling of thermal forming of glass.

Some examples and data can be found in a ``test`` subfolder of each
module.

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
