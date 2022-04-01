.. pyXsurf documentation master file, created by
   sphinx-quickstart on Mon Mar 14 19:25:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


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

TYPO3 uses

|   = with overline for parts 
|   = for chapters 
|   - for sections 
|   ~ for subsections 
|   " for subsubsections 
|   ' for paragraphs
|   ^, #, *, $, `


See https://stackoverflow.com/questions/59397527/managing-sphinx-toctrees-combined-with-inline-sections

https://stackoverflow.com/questions/46791625/how-to-separate-sphinx-chapters-within-a-single-part-into-different-files


The files linked now, have 

.. toctree::

   part_chapter

	
=======================
After header 1
=======================

Here the same toctree after a section header.


.. toctree::
   
   part_chapter


.. test with include

Now with include:


.. include::   part_chapter

	
=======================
After header 2
=======================

Here the same inclusion after a section header.


.. include::   part_chapter