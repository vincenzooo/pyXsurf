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

..
   .. toctree::

      sphinx_sections
      astropy_sections

	
=======================
After header
=======================

Here the same toctree after a header.

..
   .. toctree::
      
      sphinx_sections
      astropy_sections


