"""Sphinx configuration for the dcm-classifier documentation."""

import importlib.metadata

project = "dcm-classifier"
copyright = "2024-2026, Michal Brzus, Hans Johnson, and Cavan Riley"
author = "Michal Brzus, Hans Johnson, and Cavan Riley"
version = release = importlib.metadata.version("dcm_classifier")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = [".rst", ".md"]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
master_doc = "index"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pydicom": ("https://pydicom.github.io/pydicom/stable", None),
}

always_document_param_types = True

nitpicky = True
nitpick_ignore = [
    # autodoc copies instance attribute annotations as written.
    ("py:class", "pd.DataFrame"),
]
nitpick_ignore_regex = [
    # ITK's SWIG-wrapped classes publish no intersphinx inventory.
    ("py:class", r"itk\..*"),
]

html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
