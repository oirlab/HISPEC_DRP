# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import hashlib
import json
import re
import sys, os
from pathlib import Path
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

# Render example notebooks as static docs (do not execute during the build).
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

napoleon_custom_sections = [
    ('Calibrations', 'params_style'),
    ('Returns', 'params_style')
]

# Resolves :ref:/:doc: targets embedded in third-party (astropy/scipy/numpy)
# docstrings pulled in via autodoc -- without these, those refs have no
# inventory to resolve against and Sphinx reports them as undefined labels.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

autosummary_generate = True  # Generate autosummary stubs automatically
autodoc_default_options = {
    'members': True,          # Include all class members
    'private-members': True, # Include private members
    'undoc-members': False,    # Include undocumented members
    'show-inheritance': True, # Show class inheritance
    # Do not document ``from x import y`` names as if they belonged here (avoids
    # duplicate object descriptions across package re-exports and pipeline imports).
    'imported-members': False,
}
# Avoid pulling third-party / typing objects into the inventory via annotations.
autodoc_typehints = 'none'
autodoc_inherit_docstrings = False


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

# Embed downloadable notebooks as JS so the download button can force a real
# file save (browsers often open ``.ipynb`` as text, especially via ``file://``).
# Content-hash the script URL so a docs rebuild always busts browser caches of
# the previous embed (otherwise "Download as notebook" can serve a stale .ipynb).
_static_dir = Path(__file__).resolve().parent / "_static"
_nb_dir = _static_dir / "notebooks"
_notebook_embeds = _static_dir / "notebook_embeds.js"
_embed_lines = ["window.HISPEC_DRP_NOTEBOOKS = window.HISPEC_DRP_NOTEBOOKS || {};\n"]
if _nb_dir.is_dir():
    for _nb_path in sorted(_nb_dir.glob("*.ipynb")):
        _embed_lines.append(
            f"window.HISPEC_DRP_NOTEBOOKS[{json.dumps(_nb_path.name)}] = "
            f"{json.dumps(_nb_path.read_text(encoding='utf-8'))};\n"
        )
_embed_body = "".join(_embed_lines)
_notebook_embeds.write_text(_embed_body, encoding="utf-8")
_embed_hash = hashlib.sha256(_embed_body.encode("utf-8")).hexdigest()[:12]
_hashed_embed_name = f"notebook_embeds.{_embed_hash}.js"
for _old in _static_dir.glob("notebook_embeds.*.js"):
    if _old.name != _hashed_embed_name:
        _old.unlink(missing_ok=True)
(_static_dir / _hashed_embed_name).write_text(_embed_body, encoding="utf-8")

html_js_files = [
    _hashed_embed_name,
    "download_notebook.js",
]


add_module_names = False

# Version
version = hispecdrp.__version__
# The full version, including alpha/beta/rc tags.
#release = hispecdrp.__version__

# URL
html_baseurl = 'https://oirlab.github.io/HISPEC_DRP/'

#### Sphinx custom formatting ####

def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip ARGUMENTS/CALIBRATIONS, class attributes, and cross-module re-exports."""
    if skip:
        return True
    if name in ("PARAMETERS", "CALIBRATIONS", "_perform", "PRIMITIVES"):
        return True
    # Skip third-party/stdlib objects merely imported into a hispecdrp module
    # (e.g. astropy.table.Table, typing.Any) -- they were never "at home"
    # there, and documenting them under every module that imports them is
    # what produces duplicate-object-description / ambiguous-cross-reference
    # warnings. This check is stateless by design: an earlier version tried
    # to also dedupe *internal* cross-module re-exports (e.g. ConfigFile
    # imported into base_pipeline.py) with an id()-keyed "seen" registry, but
    # autosummary's stub-generation pass fires this same event to decide what
    # to *list* -- before the real rendering pass asks about the same
    # object -- so the registry incorrectly marked real content as "already
    # documented" and stripped it. Not worth the risk for that rarer case.
    if what in ("class", "function", "exception", "data", "module"):
        obj_mod = getattr(obj, "__module__", None)
        if obj_mod and not obj_mod.startswith("hispecdrp"):
            return True
    return False


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
