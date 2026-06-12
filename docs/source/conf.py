# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
import sys, os
sys.path.insert(0, os.path.abspath("../.."))

import hispecdrp
from hispecdrp import datamodels


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
#from sphinxawesome_theme.postprocess import Icons
from pygments.styles import get_all_styles
pygments_style = "friendly"
pygments_style_dark = "friendly"
#html_permalinks_icon = Icons.permalinks_icon
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

def parse_literal_table(lines):
    """
    Detect a fixed-width table starting with 'HDU Name ... Description'.
    Returns header and rows.
    """
    table_started = False
    rows = []
    header = None
    for line in lines:
        if re.match(r"HDU Name\s+HDU Type\s+Data Type", line):
            table_started = True
            header = re.split(r'\s{2,}', line.strip())
            continue
        if table_started:
            if not line.strip() or re.match(r"-{2,}", line):
                continue
            row = re.split(r'\s{2,}', line.strip())
            rows.append(row)
    return (header, rows) if header and rows else (None, None)

def convert_table_to_csv_directive(header, rows):
    """
    Build a .. csv-table:: directive from header + rows
    """
    lines = ['.. csv-table::',
             '    :header: ' + ', '.join(f'"{h}"' for h in header),
             '']
    for row in rows:
        lines.append("    " + ", ".join(f'"{c}"' for c in row))
    return lines

def replace_literal_table_for_html(app, what, name, obj, options, lines):
    # Only HTML build
    if app.builder.name != "html":
        return
    # Only classes
    if not isinstance(obj, type):
        return
    # Only subclasses of CalibrationModel
    if isinstance(obj, type) and not issubclass(obj, datamodels.DataModel):
        return

    header, rows = parse_literal_table(lines)
    if header and rows:
        csv_lines = convert_table_to_csv_directive(header, rows)
        # Find the start of the literal table
        start_index = next((i for i, l in enumerate(lines) if "HDU Name" in l), None)
        if start_index is not None:
            # Remove original table (header + separator + rows)
            end_index = start_index + len(rows) + 2  # +2 for header & separator
            lines[start_index:end_index] = csv_lines

def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
    app.connect("autodoc-process-docstring", replace_literal_table_for_html)