.. _developersnotes:

Developers Notes
=================

Here is summarised the status of development for code, documentation, packaging, testing.

There are many things which don't evolve so fast because I am still learning or deciding how to handle. Help about these topics are highly welcome.
Also very welcome are signaling any problem or missing information, especially if anything is not work as documented or if you can point to templates or examples.

To start contributing, or just to understand more in depth the status of develpment, you can just checkout the developer branches, and give a look around: the only developer branch at the moment is ``documentation ``, you can check it out with ``git checkout documentation``.

At this point you can compile the documentation by putting yourself in the ``docs`` directory and calling:

* ``make html`` will normally compile the "official" documentation for the pyXsurf library at the current status of development.
* ``make html .\source_test_doc`` will compile also the test documentation, creating a front page which links to the "official" documentation, but also at a test documentation, where the dubious concepts are tested in details.

In both cases the compiled HTML documentation will be visible in ``docs\build\html\index.html``.

If you modify any file, and you suspect it could be an improvement, please send it to vincenzo.cotroneo@inaf.it, or use regular git functions (that you probably better than I do).

.. note::
    Themes are defined in the ``conf.py`` file of each folder, so there might be a difference in the aspect of the "official" documentation, according to which compilation command is run. TODO: make this point uniform, by making the configuration for tests to recall the configuration for the "official page".


Active Branches
-------------------------------------------------
There a couple of branches active at the moment:

* **documentation** The most recent branching, it is mostly devoted at developing better sphinx documentation
* **readers_dev** Development to Reader routines for I/O of file formats 


Roadmap
---------------

The roadmap is detailed in the following section, main area of work are:

*  documentation
*  testing
*  installation
*  continuous Integration

Status of installer
-------------------------------------------------
I have tried the installation on a few computers and it worked smoothly 
by `setup.py`. The folder ``test_install`` contains tests 
that verify a correct installation.

I can upload the package on test-PyPI, but not on the main PyPI, as I get an error, related to the too large size of the package.

Plan is to achieve a full installable distribution on PyPI, and possibly on conda, and be able to maintain it through Continuous Integration.

Status of code
-------------------------------------------------

Code is quite consistent, and usually commented, but there are still a few modules in early stage of development or which might be removed or integrated somewhere else. Some are included for backwards compatibility, future needs or just to stay on the safe side of breaking dependencies. 
There might be additional features which I lost along the way while experimenting with git and might decide to reintegrate if I recover them for other needs.

The plan here is to add all possible code to a "developer version", and only the relevant to the official distribution. Which packages are installed should be maintained from the distribution. 

Status of docs
-------------------------------------------------

Docstrings are scattered around the code and sometimes even in the right place for self-documentation to work, 
but this is quite non-uniform,
as it mixes different conventions I tried over the time, and it will remain like this
until I find one tool that I can use to maintain the documentation.

I am not very skilled in using sphinx features, but I am studying.
I started a new ``documentation`` branch for the development of documentation, which includes also tests on the different sphinx features. Basic api documentation is very rough, but seems to contain a good amount of documentation, so the obvious step is to obtain a minimally effective, decently clean API documentation, as a starting point for a refinement of the standard in documentation format, which is often not homogeneous.  

Formatting is indeed at the moment very poor, I have tried experimenting with templates (e.g. astropy), ipynb, rst and integration with github, things that I am really not sure how to handle. See more details in the homepage of the documentation branch.

Also, some experiment with jupyter book, that looks a very appealing option on a long term, but implies a shift in paradygm.

Changes
-------------------------------------------------

2022/03/31 started documentation branch. Upgrades to sphinx docs and github readme and readme_developers. did more attempts in separate folder based on astropy and astropy templates. See VS workspace.

2022/03/22 Started restructuring repository. Renamed "source" folder to source (from pyXsurf). Add installation instructions for developers to this file and readme.txt.

2021/07/21 Upgraded installation mechanism.

2020/11/09 Changed repository name from ``pyXTel`` to
``pyXSurf``.

2021/07/21 Really trying to improve installation and release.
Mostly following https://medium.com/free-code-camp/from-a-python-project-to-an-open-source-package-an-a-to-z-guide-c34cb7139a22 and based on astropy template.

References
------------

For users new to Python, the library was developed on anaconda
distribution. Any other distribution should be equivalent, however I
never had any problem with Anaconda and find it very easy to work with,
so I can only recommend it. Same way, I used for developement
notepad++/IPython/jupyter on Windows; Spyder; VScode and feel to
recommend any of these. Of course any other environment or editor works
same way.


if you are not sure of what you are doing (as I am most of the time), feel free to keep it locally and use only ``git`` or to use any expertise you have (and I don't necessarily do have).

This is a good page explaining the basic usage of ``git`` 
https://kbroman.org/github_tutorial/pages/routine.html


See https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install for comments on installation modes.
TODO: summarize here.