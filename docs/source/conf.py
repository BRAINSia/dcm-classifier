# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinx_rtd_theme
import sys
import os

sys.path.insert(0, os.path.abspath("../../src"))

project = "dcm-classifier"
copyright = "2024, Michal Brzus and Hans Johnson and Cavan Riley"
author = "Michal Brzus and Hans Johnson and Cavan Riley"
release = "0.8.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    # "sphinx.ext.ifconfig",
    "sphinx.ext.autosummary",
    # "sphinx.ext.napoleon",  # Numpy style docstrings
    # "sphinx.ext.linkcode",
]

templates_path = ["_templates"]
exclude_patterns = []

napoleon_google_docstring = True
napoleon_numpy_docstring = False

master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
