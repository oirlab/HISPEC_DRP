# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys, os
sys.path.insert(0, os.path.abspath("../.."))

import hispecdrp


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HISPEC DRP'
copyright = '2025, HISPEC DRP Team'
author = 'HISPEC DRP Team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinxawesome_theme'
from sphinxawesome_theme.postprocess import Icons

html_permalinks_icon = Icons.permalinks_icon
html_static_path = ['_static']


# Version
version = hispecdrp.__version__
# The full version, including alpha/beta/rc tags.
#release = hispecdrp.__version__

# URL
html_baseurl = 'https://oirlab.github.io/HISPEC_DRP/'