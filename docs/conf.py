# Configuration file for the Sphinx documentation builder.
# For more information on configuration, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import warnings
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Use the installed build_config package to resolve the repo root.
try:
    from accvlab_build_config import get_project_root
except ModuleNotFoundError as exc:
    if exc.name == "accvlab_build_config":
        raise RuntimeError(
            "Sphinx documentation build requires the installed 'accvlab-build-config' package, "
            "but it is not available in this environment. Install it first (for example "
            "`pip install -e build_config` from the repository root."
        ) from exc
    raise

# Use installed packages for autodoc; only add local Sphinx extensions
project_root = get_project_root()
sys.path.insert(0, str(project_root))
# Add local Sphinx extensions
sys.path.insert(0, str(Path(__file__).parent / '_ext'))

from namespace_packages_config import NAMESPACE_PACKAGES

# -- Project information -----------------------------------------------------
project = 'ACCV-Lab'
copyright = '2026, NVIDIA Corporation'
author = 'NVIDIA Corporation'


def _get_docs_version() -> str:
    """Read and validate installed versions for all distributions included in the docs."""
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    dist_names = ["accvlab-build-config", *NAMESPACE_PACKAGES]
    versions: dict[str, str] = {}
    missing: list[str] = []

    for dist_name in dist_names:
        try:
            versions[dist_name] = _dist_version(dist_name)
        except PackageNotFoundError:
            missing.append(dist_name)

    if missing:
        raise RuntimeError(
            "Could not determine installed versions for all packages included in the "
            f"documentation: {', '.join(missing)}."
        )

    unique_versions = sorted(set(versions.values()))
    if len(unique_versions) != 1:
        version_details = "\n".join(
            f"  - {dist_name}: {dist_version}" for dist_name, dist_version in versions.items()
        )
        raise RuntimeError(
            "Version mismatch across pacakges included in the documentation:\n" f"{version_details}"
        )

    return unique_versions[0]


# The docs version must match across all documented ACCV-Lab distributions.
version = _get_docs_version()
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  # Core autodoc functionality
    'sphinx.ext.autosummary',  # Generate summary tables (but no separate files)
    'sphinx.ext.viewcode',  # Add source code links
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx.ext.mathjax',  # Math support
    'sphinx.ext.graphviz',  # Support for Graphviz diagrams
    'sphinx.ext.todo',  # Todo extension
    'sphinx.ext.coverage',  # Coverage checker
    'sphinx_autodoc_typehints',  # Type hints support
    'myst_parser',  # Markdown support
    'note_literalinclude',  # Local extension: note-literalinclude directive
    'module_docstring',  # Local extension: module-docstring directive
    'markdown_note_admonitions',  # Convert Markdown blockquotes to Sphinx admonitions
]

# Optionally enable spelling if available; warn if not, including the original error
try:
    import sphinxcontrib.spelling as _spelling  # noqa: F401
except Exception as _spelling_import_error:
    warnings.warn(
        (
            "sphinxcontrib.spelling is not available and will be disabled. "
            f"Original error: {_spelling_import_error}. "
            "To enable spell checking, install 'sphinxcontrib-spelling' and 'pyenchant', "
            "and ensure the Enchant C library is installed (e.g., 'libenchant-2-2' on Debian/Ubuntu)."
        ),
        category=UserWarning,
    )
else:
    extensions.append('sphinxcontrib.spelling')
    # Basic spelling configuration: allow custom whitelist
    spelling_word_list_filename = 'spelling_wordlist.txt'

# Follow symlinks when discovering source files
followlinks = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'generated']

# The suffix of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst',
}

# The master toctree document.
master_doc = 'index'

# -- MyST Parser Configuration -----------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Generate HTML anchor ids for headings so in-document hash links
# like [Text](#section-slug) are validated by MyST/Sphinx
myst_heading_anchors = 6

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 5,
    'includehidden': True,
    'titles_only': False,
}

# -- Options for LaTeX output -------------------------------------------------

# Use a Unicode-capable engine to support characters like box-drawing (U+250C)
latex_engine = 'xelatex'

# Configure fonts and preamble for better Unicode coverage in PDFs
latex_elements = {
    # With XeLaTeX, inputenc/utf8extra are not needed
    'inputenc': '',
    'utf8extra': '',
    'preamble': (
        r'''
\usepackage{fontspec}
\usepackage{unicode-math}
% Choose widely available system fonts with good Unicode coverage
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
\setmathfont{Latin Modern Math}
% Make quotes in verbatim look correct if available
\IfFileExists{upquote.sty}{\usepackage{upquote}}{}
'''
    ),
}

# -- Extension configuration -------------------------------------------------

# autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

# Concatenate class and __init__ docstrings
autoclass_content = 'both'

# autosummary configuration - disable file generation
autosummary_generate = False
autosummary_generate_overwrite = False

# napoleon configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'nvidia.dali': ('https://docs.nvidia.com/deeplearning/dali/user-guide/docs/', None),
}

# todo configuration
todo_include_todos = True

# -- sphinx_autodoc_typehints configuration -----------------------------------
# Show fully qualified names for types (e.g., `torch.Tensor` instead of `Tensor`)
typehints_fully_qualified = False
# Document return type in the "Returns" section
typehints_document_rtype = True
# Show default values for parameters
typehints_defaults = 'comma'
# Do not show type hints in the function signature (to avoid clutter)
typehints_use_signature = False
# Do not show return type in the signature (to avoid clutter)
typehints_use_rtype = False

# -- Custom Handlers -------------------------------------------------


def process_device_annotation(app, what, name, obj, options, lines):
    # Look for a ':device:' field in the docstring
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        device = None
        if stripped_line.startswith(':gpu:'):
            device = 'GPU'
        elif stripped_line.startswith(':cpu:'):
            device = 'CPU'
        elif stripped_line.startswith(':device:'):
            device = stripped_line.split(':device:')[1].strip().upper()

        if device is not None:
            # Insert a highlighted note at the top of the docstring
            lines.pop(i)
            lines.insert(0, f'.. container:: device-note')
            lines.insert(1, f'    ')
            lines.insert(
                2,
                f'     This function expects input data to be on the **{device}**.',
            )
            lines.insert(3, '')
            break  # Only process the first occurrence


def setup(app):
    """Setup function for Sphinx"""
    # Custom processing can be added here
    app.connect('autodoc-process-docstring', process_device_annotation)
