# pyXSurf (formerly pyXTel)
Python library for X-Ray Optics, Metrology Data Analysis and Telescopes Design.
2020/11/09 Changed repository name from `pyXTel` to `pyXSurf`.

## Installation
Download or fork the project. You can use git clone command by: `git clone https://github.com/vincenzooo/pyXSurf.git` or the button `download` at top of page.
The folder containing code must be made accessible to python on your system (generally this means it must be in the system path).

For users new to Python, the library was developed on anaconda distribution. Any other distribution should be equivalent, however I never had any problem with Anaconda and find it very easy to work with, so I can only recommend it.
Same way, I used for developement notepad++/IPython/jupyter on Windows; Spyder; VScode and feel to recommend any of these. Of course any other environment or editor works same way. 

## Status of code and documentation
The main part of the library is well defined and it works well. I am constantly adding functions when I find they are needed during my daily work.
Not all modules and functions in the repository are meant to be included in the library, with a few of them in early stage development or abandoned, however I don't know how to remove them from the public distribution, while keeping them for backwards compatibility or future needs, so they are included. 

For the there is a good amount of comments in the code, and sometimes even in the right place for self-documentation to work. Unfortunately I am not proficient enough to automatically generate docs from code, or even follow a consistent standard (which one?), howevere there is a good amount of comments and docstrings in code, you should be able to access it by usual python introspection (?, ??, autocompletion, etc.).
There are some tutorial and examples, but they are quite scattered around in folders `Demo`, `Tutorial`, `Test`, etc.

**TODO:** remove unessential files. Learn how to generate documentation from other functions docstring.

## Modules

A basic description of the different modules is:
* **dataIO**
Generic routines for accessing and manipulating data and files.
* **notebooks**
Jupyter notebooks, not necessarily related to the libraries, include test and experiments on python.
* **plotting**
Plotting functions for pySurf data.
* **pyGeo3D**
Functions for geometry in space (lines and planes).
* **pyProfile**
Equivalent of pySurf acting on Profiles.
* **pySurf**
Functions and classes acting on 3D points or surfaces.
* **thermal**
Functions for modelling of thermal forming of glass.

Some examples and data can be found in a `test` subfolder of each module.

## Author
Vincenzo Cotroneo
vincenzo.cotroneo@inaf.it
