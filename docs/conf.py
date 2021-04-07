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

# Point to root source dir for API doc, relative to this file:
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "Bellman"
copyright = "2021, The Bellman Contributors"
author = "The Bellman Contributors"

# The full version, including alpha/beta/rc tags
with open(os.path.join("..", "VERSION"), "r") as file:
    release = file.read()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.intersphinx",  # Link to other project's documentation (see mapping below)
    "sphinx.ext.mathjax",  # Render math via Javascript
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    "nbsphinx",  # Integrate Jupyter Notebooks and Sphinx
    "IPython.sphinxext.ipython_console_highlighting",
]

# Mappings for sphinx.ext.intersphinx. Projects have to have a Sphinx-generated doc (.inv file)
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    # Unfort. doesn't work yet! See https://github.com/mr-ubik/tensorflow-intersphinx/issues/1
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "https://raw.githubusercontent.com/mr-ubik/tensorflow-intersphinx/master/tf2_py_objects.inv",
    ),
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = (
    False  # Remove 'view source code' from top of page (for html, not python)
)
nbsphinx_allow_errors = True  # Continue through Jupyter errors
autodoc_inherit_docstrings = False  # If no class summary, *don't* inherit base class summary

# Add any paths that contain Jinja2 templates here, relative to this directory.
templates_path = ["_templates"]

# Exclusions
# To exclude a module, use autodoc_mock_imports. Note this may increase build time:
# autodoc_mock_imports = [
#    'bellman.agents.mpc'
# ]
# To exclude a class, function, method or attribute, use autodoc-skip-member. (Note this can also
# be used in reverse, ie. to include a particular member that would otherwise be excluded.)
# Note that 'private' and 'special' members (_ or __) are already excluded using the Jinja2
# templates, by absenting autoclass directives (ie. :private-members:) and by explicit 'if-not'
# statements (to exclude from summary tables):
# def autodoc_skip_member_callback(app, what, name, obj, skip, options):
#    # This would exclude the TRPOLossInfo and RenderPyEnvironment classes:
#    exclusions = ('TRPOLossInfo', 'RenderPyEnvironment')
#    exclude = name in exclusions
#    return skip or exclude
# def setup(app):
#    # Entry point to autodoc-skip-member
#    app.connect("autodoc-skip-member", autodoc_skip_member_callback)


# -- Options for HTML output -------------------------------------------------

# on_rtd is whether on readthedocs.org, this line of code grabbed from docs.readthedocs.org...
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["color_theme.css"]
