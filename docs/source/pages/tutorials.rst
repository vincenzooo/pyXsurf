Tutorials
=========

.. for now I put all notebooks of documentation to notebooks folder, making a copy of the original
    in progetti/pySurf folder will clean later

    I am using pointed list everywhere because I have not clear how toctree works. 

.. TODO: these files in _static etc. are not compiled, so ipynb are not tested.

.. TODO: some of the notebooks have been copied by source in _static or notebooks folder. The first are compiled, but are not in synch with the source.

Overview
-------------------------------

A few introductory presentations to give an overall view of purposes and functionalities:

* Poster (SPIE 2023, San Diego) `pdf <../_static/vcotroneo_SPIE2023.pdf>`_
* Jupyter Notebook converted to `Presentation <../_static/pySurf_NSFE2020.slides.html>`_, or  `report <../_static/pySurf_NSFE2020.html>`_ derived from presentation at NSFE NanoScientific Forum 2020
* Gallery: Examples of different interfaces TODO 🛠️ 

.. TODO: add link to SPIE and NSFE, also recording

Examples
--------

These Jupyter notebooks illustrate practical examples on how to start or to perform specific operations.  

*   `Introduction to the library <../notebooks/pySurf_demo.ipynb>`_ :download:`download source <../notebooks/pySurf_demo.ipynb>`

*  `Usage <../notebooks/basic_usage.ipynb>`_ :download:`download source <../notebooks/basic_usage.ipynb>`

Advanced Examples
-----------------

Some example from real life, Still work in progress. TODO 🛠️ 

Surface Processing
******************

`Example of interactive alignment <../_static/rotate_and_align.html>`_

A more complex analysis `report with toggleable code visualization <../_static/C1S04_PZT_WFS_stress_fit.html>`_ in which difference in two shapes before and after a treatment is measured and fit to a simulated deformation.

Profile Class
******************
Example of class ``Profile``, 1D analogous of ``Data2D``:

`Simple Signal <../notebooks/test_make_signal.ipynb>`_

`Introduction to Profile <../_static/profile_demo_rise.slides.html>`_, created from Jupyter notebook in different formats: `pdf <../_static/profile_demo_rise.slides.pdf>`_ :download:`download source <../_static/profile_demo_rise.ipynb>`

`Profile Tutorial <../_static/Profile_class_tutorial.html>`_  :download:`download source <../_static/Profile_class_tutorial.ipynb>`

`Analysis of PSD and Merging tests <../_static/test_merge_P01.html>`_ `pdf <../_static/test_merge_P01.slides.pdf>`_ :download:`download source <../_static/test_merge_P01.ipynb>` PSDs can be extracted from `Data2D`` surfaces.

`Teoretical and numerical analysis of PSD <../_static/PSDtest.html>`_



Where to find more
****************************

More examples and tests can be found scattered in subfolders of source code repository, see also `Developer Notes <readmedev_link.rst>`_.
	
   




