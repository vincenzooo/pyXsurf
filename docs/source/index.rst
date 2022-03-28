.. pyXsurf documentation master file, created by
   sphinx-quickstart on Mon Mar 14 19:25:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyXsurf's documentation!
===================================
.. 
	a file ipynb linked in a toc, is correctly replaced by its sections. A file rst in outer directories can be called by .. include directive, which works also in external rst files (deleteme_link ) if a rst file is included, but doesn't work including ipynb, which are included as text content.
	
	
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api   
   basic_usage
   deleteme_link
   deleteme_link2
   deleteme/basic_usage.ipynb
   deleteme/basic_usage.html
   
Prova riferimenti
====================
.. raw:: html
  :file: deleteme/basic_usage.html
 
.. include:: ../deleteme2/basic_usage2.rst

Alt
====
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
