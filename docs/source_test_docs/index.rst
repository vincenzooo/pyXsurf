.. pyXsurf documentation master file, created by
   sphinx-quickstart on Mon Mar 14 19:25:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. VC: in source_test, call the original index file related to this folder with experiments about docs and the real documentation


==============================
Page of documentation branch
==============================

Here there are two folders: ``source_test_doc`` containing experimentation on *sphinx* and *rst* functions and
their usage, in which this file is located, and ``docs``, containing the developing official documentation for pyXsurf
main branch.

The most complete documentation is created by compiling the source in the dev documentation folder ``source_test_doc``.
This can be easily be dome by runnning the modified make file, with an optional source dir as argument.
So, if you put yourself in the ``docs`` folder you can call:

* ``make file``, which as usual compiles the *official* documentation, or
* ``make file\source_test_docs`` which render the same plus the additional documentation of the *documentation* branch.

The last one also renders this "cover page", which allows you to chose between the two:


.. toctree::
   :maxdepth: 1
   
   pySurf "official documentation" (main page) <index_docs_link>
   Testing sphinx and rst usage <index_tests>




=======================
Automatically generated
=======================


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

