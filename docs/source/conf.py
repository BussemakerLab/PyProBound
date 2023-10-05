# pylint: disable=missing-module-docstring
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.pardir, "src")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyProBound"
author = "Lucas A.N. Melo and Harmen J. Bussemaker"
year = time.gmtime().tm_year
copyright_year = f"2023-{year}" if year > 2023 else "2023"
project_copyright = f"{copyright_year}, {author}"

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
add_module_names = False
autodoc_member_order = "bysource"
napoleon_google_docstring = True
autodoc_default_options = {"member-order": "bysource", "undoc-members": True}
nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {"display_version": True}
html_css_files = ["custom.css"]


def skip_member(app, what, name, obj, skip, options):
    del app, what, obj, skip, options
    if name not in ("forward", "tensor") and (
        (name in dir(torch.nn.Module))
        or (name in dir(torch.Tensor))
        or (name == "training")
    ):
        return True


def process_docstring(app, what, name, obj, options, lines):
    del app, what, name, obj, options
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Alias for field number"):
            del lines[i]


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
    app.connect("autodoc-process-docstring", process_docstring)
