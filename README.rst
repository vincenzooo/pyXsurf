pyXSurf (formerly pyXTel)
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

Status of code and documentation
--------------------------------

The main part of the library is well defined and it works well. I am
constantly adding functions when I find they are needed during my daily
work. The selection of modules for public release is in phase of improvement,
there are still a few modules that are in early stage or not of general interest, 
but are included for backwards compatibility, future needs or just to stay on the safe side.

Code is usually commented, and sometimes even in the right place for self-documentation to work, 
but this is quite non-uniform,
as it mixes different conventions I tried over the time, and it will remain like this
until I find one tool that I can use to maintain the documentation.  
In the meanwhile, you should be able to access
it by usual python introspection (``?``, ``??``, autocompletion, etc.). There
are some tutorial and examples, but they are quite scattered around in
folders ``Demo``, ``Tutorial``, ``Test``, etc.

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

Author
------

Vincenzo Cotroneo vincenzo.cotroneo@inaf.it
