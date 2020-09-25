# pyXTel
Python library for X-Ray Optics, Metrology Data Analysis and Telescopes Design.

## Installation
Download or fork the project in a folder accessible on your system.

## Status
This is a pre-release version, modules under preliminary development are excluded. 

First version of repository includes several scripts that are not essential or well commented, however there is a good amount of comments in the code, and sometimes even in the right place for self-documentation to work.
There are some tutorial and examples, but they are quite scattered around in folders `Demo`, `Tutorial`, `Test`, etc.

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
