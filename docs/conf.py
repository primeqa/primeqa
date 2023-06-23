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

# sys.path.append(os.path.abspath("../primeqa"))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('_ext'))

# -- Project information -----------------------------------------------------

project = 'PrimeQA'
copyright = '2022, IBM Research AI'
author = 'IBM Research AI'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.duration',
    'sphinx.ext.napoleon',
    'sphinx_design',
    # 'recommonmark',
    'edit_on_github',
    'sphinx.ext.coverage',
    'nbsphinx'
]

# Enable myst extensions
myst_enable_extensions = ["colon_fence"]

# Turn on sphinx.ext.autosummary
autosummary_generate = True 

# Configuration of sphinx.ext.coverage
coverage_show_missing_items = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

html_theme_options = {
    "use_edit_page_button": True,
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/primeqa/primeqa/",
            "icon": "https://badgen.net/github/stars/primeqa/primeqa?icon=github",
            "type": "url",
        },
        # {
        #     "name": "GitHub",
        #     "url": "https://github.com/primeqa/primeqa",
        #     "icon": "fab fa-github-square",
        #     "type": "fontawesome",
        # },
        #  {
        #     "name": "Support",
        #     "url": "https://github.com/primeqa/primeqa/discussions",
        #     "icon": "fa fa-comment fa-fw",
        #     "type": "fontawesome",
        # },
        {
            "name": "Slack",
            "url": "https://join.slack.com/t/primeqaworkspace/shared_invite/zt-1edc4fn7n-6aUO0CCvDOMOLb0drROwSw",
            "icon": "https://cdn.bfldr.com/5H442O3W/at/pl546j-7le8zk-6gwiyo/Slack_Mark.svg?auto=webp&format=png",
            "type": "url",
        },
   ],
    "show_prev_next": True,
    "navbar_align": "content",
    "logo":{
        "text": "PrimeQA: The Prime Repository for QA",
    },
    "globaltoc_collapse": True,
    "globaltoc_includehidden": False,
    "globaltoc_maxdepth": 2,
    "favicons": [
        { 
            "rel": "icon",
            "sizes": "16x16",
            "href": "primeqa_logo.png",
        },
        {
            "rel": "icon",
            "sizes": "32x32",
            "href": "primeqa_logo.png",
        },
        {
            "rel": "apple-touch-icon",
            "sizes": "180x180",
            "href": "primeqa_logo.png"
        },
    ],
    # "page_sidebar_items": ["custom-right-section.html", "page-toc", "edit-this-page"],
    "page_sidebar_items": ["custom-right-section.html", "page-toc"],
    # "footer_items": ["copyright", "sphinx-version"],
    "footer_items": [],

}

html_context = {
    "github_user": "primeqa",
    "github_repo": "primeqa",
    "github_version": "main",
    # "edit_page_url_template": "{{ my_vcs_site }}{{ file_name }}{{ some_other_arg }}",
    # "my_vcs_site": "https://github.com/primeqa/primeqa/edit/main/docs/",
    # "file_name": "",
    # "some_other_arg": "",
    "default_mode": "light"
}

edit_on_github_project = 'primeqa/primeqa'
edit_on_github_branch = 'main'

html_css_files = [
    "custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/fontawesome.min.css"
]

html_sidebars = { 
        "index": [
            "search-field.html","custom-left-section.html"
        ],
        "installation": [
            "search-field.html","custom-left-section.html"
        ],
        "development": [
            "search-field.html","custom-left-section.html"
        ],
        "testing": [
            "search-field.html","custom-left-section.html"
        ],
        "tutorials/index": [
            "search-field.html","custom-left-section-tutorials.html"
        ],
        "tutorials/01_Index_Documents_with_DPR.ipynb": [
            "search-field.html","custom-left-section-tutorials.html"
        ],
        "tutorials/02_RAG_Generator": [
            "search-field.html","custom-left-section-tutorials.html"
        ],
        "tutorials/03_ReRank_search_results.ipynb": [
            "search-field.html","custom-left-section-tutorials.html"
        ],
        "tutorials/04_Finetune_LLM_for_QA_with_QLORA.ipynb": [
            "search-field.html","custom-left-section-tutorials.html"
        ],
        "pull_request_template": [
            "search-field.html","custom-left-section.html"
        ],
        "orchestrator": [
            "search-field.html","custom-left-section.html"
        ],
        "tooling_ui": [
            "search-field.html","custom-left-section.html"
        ],
        "README": [
            "search-field.html","custom-left-section.html"
        ],
        "api/index": [
            "search-field.html"
        ],
        "api/boolqa/index": [
            "search-field.html","custom-left-section-api-pkg.html"
        ],
        "api/calibration/index": [
            "search-field.html","custom-left-section-api-pkg.html"
        ],
        "api/distillation/index": [
            "search-field.html","custom-left-section-api-pkg.html"
        ],
        "api/ir/index": [
            "search-field.html","custom-left-section-api-pkg.html"
        ],
        "api/mrc/index": [
            "search-field.html","custom-left-section-api-pkg.html"
        ],
        "api/pipelines/index": [
            "search-field.html","custom-left-section-api-pkg.html"
        ],
        "api/qg/index": [
            "search-field.html","custom-left-section-api-pkg.html"
        ],
        "api/util/index": [
            "search-field.html","custom-left-section-api-pkg.html"
        ],
        "api/tableqa/index": [
            "search-field.html","custom-left-section-api-pkg.html"
        ],
     }

# do not try to provide a backend for jupyter notebooks
nbsphinx_execute = 'never'

# instruct nbsphinx to structure code cells with templates
nbsphinx_templates_path = ['_templates']

