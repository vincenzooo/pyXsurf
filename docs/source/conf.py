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
import os
import sys

# according to below and https://sphinx-rtd-tutorial.readthedocs.io/en/latest/sphinx-config.html
sys.path.insert(0, os.path.abspath('..\..\source'))

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
import os
sys.path.insert(0, os.path.join(pathlib.Path(__file__).parents[2].resolve().as_posix(),'source','pySurf'))
print("**"+os.path.join(pathlib.Path(__file__).parents[2].resolve().as_posix(),'source','pySurf')+"**") #pyXTel
sys.path.insert(0, r'..\deleteme2')

# -- Project information -----------------------------------------------------

project = 'pyXsurf'
copyright = '2022, Vincenzo Cotroneo'
author = 'Vincenzo Cotroneo'

# The full version, including alpha/beta/rc tags
release = 'v1.5.6'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['myst_parser',
              #"myst_nb",
              'sphinx.ext.duration',
              'nbsphinx',
              #'sphinx_gallery.gen_gallery',
              #'sphinx_gallery.load_style',   #not present in latest documentation
              #"guzzle_sphinx_theme",
              'sphinx.ext.autodoc',
              #'sphinx.ext.doctest',
              'sphinx.ext.autosummary',
              'sphinx_automodapi.automodapi',
              #'sphinx.ext.autosectionlabel',
              'sphinx.ext.inheritance_diagram'
]


autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

#  ----------- VC Sphinx Gallery settings according to https://sphinx-gallery.github.io/stable/getting_started.html#create-simple-gallery
# paths are relative to this file.
# sphinx_gallery_conf = {
     # 'examples_dirs': 'examples',   # path to your example scripts
     # 'gallery_dirs': 'gallery_auto_examples',  # path to where to save gallery generated output
# }

html_logo = 'resources/Transparent Logo.png'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bizstyle'


#html_theme = 'haiku'

# import astropy_sphinx_theme
# html_theme_path = astropy_sphinx_theme.get_html_theme_path()
# html_theme = 'bootstrap-astropy'
# html_theme_options = {
    # 'logotext1':'py',
    # 'logotext2': 'X',  # orange, light
    # 'logotext3': 'surf'   # white,  light,
# }
# html_logo = 'resources/surf-board-wave_small.png'

# Static files to copy after template files
# https://docs.astropy.org/projects/package-template/en/latest/nextsteps.html
#html_static_path = ['_static']
#html_style = 'pyXsurf.css'


# VC
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

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
html_favicon = 'resources/Favicon Original.ico'


'''
import recommonmark
# VC http://tansignariold.opendatasicilia.it/it/latest/ricette/ReadtheDocs/come_fare_leggere_un_file_MD_a_ReadtheDocs.html
from recommonmark.transform import AutoStructify

from recommonmark.parser import CommonMarkParser

source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser',
}

source_suffix = ['.rst', '.md']

extensions = ['sphinx.ext.ifconfig','sphinx_markdown_tables']
'''