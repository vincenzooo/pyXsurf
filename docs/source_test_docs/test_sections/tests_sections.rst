.. pyXsurf documentation master file, created by
   sphinx-quickstart on Mon Mar 14 19:25:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

############################################
Test sections (part, level 1, # w overline)
############################################

*******************************************************
A simple root page (chapter, level2, * with overline)
*******************************************************

Tests for sections, this is in the root, without any header.
The header is placed later, this also has the effect to broadcast to upper level as if it
was the title of the embedded file.


Astropy reccommendations are:

   |   # with overline for parts 
   |   * with overline for chapters 
   |   = for sections 
   |   - for subsections 
   |   ^ for subsubsections 
   |   “ for paragraphs

Readme currently is:

   |   = for sections 
   |   - for subsections 

These guidelines follow Sphinx’s recommendation in the Sections chapter of its reStructuredText Primer 
and Python’s convention in the 7.3.6. Sections part of its style guide.

See https://stackoverflow.com/questions/59397527/managing-sphinx-toctrees-combined-with-inline-sections

https://stackoverflow.com/questions/46791625/how-to-separate-sphinx-chapters-within-a-single-part-into-different-files

*******************************************************
Still level2, * with overline
*******************************************************

Here I do a test including files containing headers of any (or none) level, containing a text before and after the level header.

We are now after a level 2 header and this is what happens:

* Files are shown as a continuation of current page
* Level 1 appears in the toc calling this file.

Note you need a white line after include directive, .rst extension is also essential (at least in my case with subfolders).

..
   This is a more complex test:

   .. include:: complex_test


**Here I include a file with header of level 0:**

.. include:: doc_files/rst_level0.rst

**Here I include a file with header of level 1:**

.. include:: doc_files/rst_level1.rst


**Here I include a file with header of level 2:**

.. include:: doc_files/rst_level2.rst

**Here I include a file with header of level 3:**

.. include:: doc_files/rst_level3.rst

**Here I include a file with header of level 4:**

.. include:: doc_files/rst_level4.rst

Here I include a file with header of level 2, followed by a level 3:

.. include:: doc_files/rst_level23.rst

Here I include a file with header of level 2, including a level 3:

.. include:: doc_files/rst_level2inc3.rst

Here is after all include

Here, the same in toctree. All files have internal headers all of same style (level0 doesn't appear in toc, not even with ``level0 <doc_files/rst_level0>`` or  ``level0<doc_files/rst_level0>``):

.. toctree:: 
   doc_files/rst_level0
   doc_files/rst_level1   
   doc_files/rst_level2
   doc_files/rst_level3
   doc_files/rst_level4
   doc_files/rst_level23