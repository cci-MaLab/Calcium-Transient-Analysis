# Configuration file for the Sphinx documentation builder.
import datetime
import os
import sys
import matplotlib


matplotlib.use('Agg')

sys.path.append(os.path.abspath("../ext"))
sys.path.insert(0, os.path.abspath("../../"))
# -- Project information

project = 'Calcium Transient Analysis'
copyright = '2023-%s, CCI Ma Lab Team' % datetime.datetime.now().year
author = 'Michal Lange'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",

]

napoleon_use_rtype = False
napoleon_use_ivar = True
autodoc_typehints = "none"
autodoc_member_order = "groupwise"
autoclass_content = "both"
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["custom.css"]