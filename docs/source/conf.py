# pylint: disable=missing-module-docstring
import importlib.metadata
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.pardir, "src")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ProBound"
author = "Lucas Melo"
year = time.gmtime().tm_year
copyright_year = f"2023-{year}" if year > 2023 else "2023"
project_copyright = f"{copyright_year}, {author}"
release = importlib.metadata.version("probound")
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True
autodoc_default_options = {"member-order": "bysource", "undoc-members": True}
nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {"display_version": True}


def skip_member(app, what, name, obj, skip, options):
    del app, what, obj, skip, options
    if name not in ("forward", "tensor") and (
        (name in dir(torch.nn.Module())) or (name in dir(torch.Tensor()))
    ):
        return True


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
