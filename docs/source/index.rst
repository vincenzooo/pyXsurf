.. pyXsurf documentation master file, created by
   sphinx-quickstart on Mon Mar 14 19:25:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. the "raw" directive below is used to hide the title in favor of just the logo being visible
.. raw:: html

    <style media="screen" type="text/css">
      h1 { display:none; }
    </style>


Welcome to pyXsurf's documentation!
===================================

.. from astropy

.. |logo_svg| image:: _static/astropy_banner.svg

.. |logo_png| image:: _static/astropy_banner_96.png

.. raw:: html

   <img src="_static/astropy_banner.svg" onerror="this.src='_static/astropy_banner_96.png'; this.onerror=null;" width="485"/>

.. only:: latex

    .. image:: _static/astropy_logo.pdf
	
.. 
	a file ipynb linked in a toc, is correctly replaced by its sections. A file rst in outer directories can be called by .. include directive, which works also in external rst files (deleteme_link) if it has a path to a rst file, but doesn't work if it is ipynb, which are included as text content.
	
	


***************
Getting Started
***************

.. toctree::
   :maxdepth: 4
   readme_link


SpectraPy API
=============
.. toctree::
	:maxdepth: 1
    api
   

***************
Second toctree
***************


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   api   
   deleteme/basic_usage.ipynb

***************
Third toctree
***************
   
.. toctree::
   :maxdepth: 2
   basic_usage
   deleteme_link
   deleteme_link2


Prova riferimenti
====================

RAW HTML:

.. raw:: html
  :file: deleteme/basic_usage.html
 
INCLUDE OUTER DIR:
.. include::  ../deleteme2/basic_usage2.rst

Alt
====

basic_usage2.rst

.. automodule:: pySurf
   :members:
  
.. autosummary::
   :toctree: generated
	   
.. autosummary::
    :toctree: _autosummary
    :recursive:
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
