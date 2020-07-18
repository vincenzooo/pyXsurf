# pyXTel
Python library for X-Ray Optics, Metrology Data Analysis and Telescopes Design.

## Installation
Download or fork the project in a folder accessible on your system.

## Status
This is a pre-release version, I tried to exclude modules with mature enough development and comments,
but some still in development might be included. Some of these are not essential or well commented.

For the most important functions and modules, there is a good amount of documentation in form of docstrings, 
code comments, tests and jupyter notebooks.
I tried to give them some structure, but I am not expert enough to automatically generate documentation (any help or directions in that sense is welcome),
and this makes it very hard to maintain and keep updated, so I prefer for now to just keep comments updated and consistent,
rather than giving them a correct structure.
Hopefully one day it will get more complete and sorted, and I will be able to give it a good final structure.
Until this happens, you can just dig through folders and code and look for files, folders and functions with 
representative names (`test`, `tutorials` etc.).

**TODO:** remove unessential files. Learn how to generate documentation from other functions docstring.

## Modules

A basic description of the different modules is:
* **dataIO**
Generic routines for accessing and manipulating data.
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
