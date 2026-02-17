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
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

napoleon_custom_sections = [
    ('Calibrations', 'params_style'),
    ('Returns', 'params_style')
]

autosummary_generate = True  # Generate autosummary stubs automatically
autodoc_default_options = {
    'members': True,          # Include all class members
    'private-members': True, # Include private members
    'undoc-members': False,    # Include undocumented members
    'show-inheritance': True, # Show class inheritance
}

templates_path = ['_templates']
exclude_patterns = []

# Add the custom CSS file
html_css_files = [
    'custom.css'
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://sphinxawesome.xyz/how-to/configure/
html_theme = 'sphinxawesome_theme'
#html_theme = 'sphinx_rtd_theme'
from sphinxawesome_theme.postprocess import Icons
from pygments.styles import get_all_styles
pygments_style = "friendly"
pygments_style_dark = "friendly"
html_permalinks_icon = Icons.permalinks_icon
html_static_path = ['_static']
html_css_files = ["custom.css"]

add_module_names = False

# Version
version = hispecdrp.__version__
# The full version, including alpha/beta/rc tags.
#release = hispecdrp.__version__

# URL
html_baseurl = 'https://oirlab.github.io/HISPEC_DRP/'

#### Sphinx custom formatting ####


# Skip primitive and pipeline calibrations and arguments variables since we explicitly document those.
def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip ARGUMENTS and CALIBRATIONS class variables from documentation."""
    # if what == "class" and name in ("ARGUMENTS", "CALIBRATIONS", "_perform") and issubclass(obj.__class__, hispecdrp.primitives.base_primitive.BasePrimitive) and obj is not hispecdrp.primitives.base_primitive.BasePrimitive:
    if what == "class" and name in ("ARGUMENTS", "CALIBRATIONS", "_perform", "PRIMITIVES"):
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)