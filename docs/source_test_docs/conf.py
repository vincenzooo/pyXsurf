# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
#import pathlib
#import sys
#sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
## print(pathlib.Path(__file__).parents[2].resolve().as_posix()) #pyXTel
#sys.path.insert(0, r'..\deleteme2')

# -- Project information -----------------------------------------------------

project = 'pyXsurf'
copyright = '2022, Vincenzo Cotroneo'
author = 'Vincenzo Cotroneo'

# The full version, including alpha/beta/rc tags
release = 'v1.5.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [#'myst_parser',
              'sphinx.ext.duration',
              #"guzzle_sphinx_theme",
              'sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.autosummary',
              'nbsphinx',
              'sphinx_automodapi.automodapi',
              'sphinx_automodapi.automodapi',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.inheritance_diagram'
]

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'



#html_theme = 'haiku'  #viene greco
#html_theme = 'traditional'
html_theme = 'bootstrap-astropy'

#import guzzle_sphinx_theme
#html_theme_path = guzzle_sphinx_theme.html_theme_path()
#html_theme = 'guzzle_sphinx_theme'
## Guzzle theme options (see theme.conf for more information)
#html_theme_options = {
#    # Set the name of the project to appear in the sidebar
#    "project_nav_name": "pyXsurf",
#}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

