pyXsurf (formerly pyXTel)
=========================

Python library for X-Ray Optics, Metrology Data Analysis and Telescopes
Design. 

2022/03/22 Started restructuring repository. Renamed "source" folder to source (from pyXsurf).Add installation instructions for developers to this file and readme.txt.
2021/07/21 Upgraded installation mechanism.
2020/11/09 Changed repository name from ``pyXTel`` to
``pyXSurf``.

Installation for developers
-----------
I write here some notes on how to install for developers according to what I understood so far with my setup, please let me know if anything is incorrect or doesn't apply to different conditions.


Developers Notes
------------

As a developer, you probably want to:
1) have all dependencies installed
2) refer to the most updated version of the code, making it sure that you are not referring to an installed version.

Python installation done with setup.py will install the library copying to a specific location (site-packages) and installing all dependencies specified in the file.
If something is changed, the package must be reinstalled.

So, you can satisfy point 1 by installing manually 



https://stackoverflow.com/questions/37006114/anaconda-permanently-include-external-packages-like-in-pythonpath

I found two answers to my question in the Anaconda forum:

1.) Put the modules into into site-packages, i.e. the directory $HOME/path/to/anaconda/lib/pythonX.X/site-packages which is always on sys.path. This should also work by creating a symbolic link.

2.) Add a .pth file to the directory $HOME/path/to/anaconda/lib/pythonX.X/site-packages. This can be named anything (it just must end with .pth). A .pth file is just a newline-separated listing of the full path-names of directories that will be added to your path on Python startup.

Alternatively, if you only want to link to a particular conda environment then add the .pth file to ~/anaconda3/envs/{NAME_OF_ENVIRONMENT}/lib/pythonX.X/site-packages/

Both work straightforward and I went for the second option as it is more flexible.

*** UPDATE:

3.) Use conda develop i. e. conda-develop /path/to/module/ to add the module which creates a .pth file as described under option 2.).

4.) Create a setup.py in the folder of your package and install it using pip install -e /path/to/package which is the cleanest option from my point of view because you can also see all installations using pip list. Note that the option -e allows to edit the package code. See here for more information.


2021/07/21
Really trying to improve installation and release.
Mostly following https://medium.com/free-code-camp/from-a-python-project-to-an-open-source-package-an-a-to-z-guide-c34cb7139a22

and based on astropy template.

This is a good page explaining the basic usage of ``git`` 
https://kbroman.org/github_tutorial/pages/routine.html

if you are not sure of what you are doing (as I am most of the time), feel free to keep it locally and use only ``git`` or to use any expertise you have (and I don't necessarily do have).
