.. pyXsurf documentation master file, created by
   sphinx-quickstart on Mon Mar 14 19:25:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. the "raw" directive below is used to hide the title in favor of just the logo being visible
.. raw:: html

    <style media="screen" type="text/css">
      h1 { display:none; }
    </style>


Tests for embedding of different formats in different positions
=============================================================================
..
	.. |logo_svg| image:: _static/astropy_banner.svg

	.. |logo_png| image:: _static/astropy_banner_96.png

	.. raw:: html

	   <img src="_static/astropy_banner.svg" onerror="this.src='_static/astropy_banner_96.png'; this.onerror=null;" width="485"/>

	.. only:: latex

		.. image:: _static/astropy_logo.pdf


Embedding rst directly and indirectly
-------------------------------------

.rst files ``readme.rst``, are linked (even without extension) directly, through a file ``readme_link.rst`` which references (through directive ``.. include::``) to a parent folder (works in same way for any path) with a readme file. In both cases the extension can be omitted:

.. toctree::
   :maxdepth: 1
   
   readme from rst <readme>
   readme from external parent dir <readme_link>
   
  
Embedding ipynb
---------------

If a file ``.ipynb`` is direcly called in ipynb, each of its sections is inserted in the toc (it might depend on how sections are specified in notebook and in rst): 
   
   
.. toctree::
   :maxdepth: 1
   
   doc_files/basic_usage.ipynb

If instead files are included in external .rst:

.. toctree::
   :maxdepth: 1

   embed ipynb in toctree <basic_ipynb_toctree>
   ipynb with include (wrong) <ipynb_include_link>



Ideally one wants to visualize a single link, that, when clicked, brings you to a single jupyter page (it can have internal sections, but they don't have to "spill" in the calling table of content).

If this happens might depend on the levels of titles. Controlling this behavior is the reason for making another page with **section tests*.



Embedding HTML
-----------------

This works an rst which calls RAW HTML:

.. toctree::
   :maxdepth: 1

   call external link to html <basic_html_link>


Non funzionanti
---------------

Include rst from parent toctree with headers of lower level

.. toctree::
   :maxdepth: 2

   put directly HTML file <doc_files/basic_usage.html>
   include rst with link to rst <deleteme_link>
   try include ipynb in parent folder <deleteme_link2>
   doc_files/basic_usage

Comments
--------

The guide at says

   An "empty comment" does not consume following blocks. (An empty comment is ".." with blank lines before and after.)


.. code-block:: rest 

   ..
      comment

   ..

      not comment   

generates:

..
   comment

..

   not comment


RAW HTML:
---------

.. raw:: html
  :file: deleteme/basic_usage.html
 
INCLUDE OUTER DIR:

  .. include::  ../deleteme2/basic_usage2.rst


pyXsurf API
-----------

.. toctree::
    api
	
=======================
Automatically generated
=======================


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
