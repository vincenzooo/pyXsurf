.. pyXsurf documentation master file, created by
   sphinx-quickstart on Mon Mar 14 19:25:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. VC: in source_test, call the original index file related to this folder with experiments about docs and the real documentation


==============================
Page of documentation branch
==============================

Here there are two folders: ``source_test_doc`` containing experimentation on ``sphinx`` and ``rst`` functions and
their usage, in which this file is located, and ``docs``, containing the developing official documentation for pyXsurf
main branch.
The makefile in the parent folder was modify to accept an optional source dir as argument. 
This makes is possible to compile the *official* documentation, with the usual ``make file``,
or the documentation of the *documentation* branch if the folder is passed with ``make file\source_test_docs``,
to render this file.


.. toctree::
   :maxdepth: 1
   
   pySurf documentation <index_docs_link>
   sphinx and rst usage <index_tests>

	
=======================
Automatically generated
=======================


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

