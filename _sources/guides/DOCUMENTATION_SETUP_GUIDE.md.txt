# Sphinx Documentation Setup Guide

This document explains the Sphinx documentation setup for ACCV-Lab.


## Overview

The documentation system provides:

- **Explicit namespace package configuration** through `namespace_packages_config.py`
- **Dynamic documentation generation** for each configured namespace package
- **Optional package-local asset generation** for generated documentation assets such as plots
- **Comprehensive API reference** with auto-generated content (extracted from docstrings)
- **Referenced directories mirroring** to access files from the individual namespace packages in the 
  documentation by
  - collecting all the needed files from the individual namespace packages
  - without introducing duplicates
  - and while not breaking any relative path references to files inside the package documentation & additional 
    locations such as e.g. an `examples` directory from which files need to be displayed in the documentation.

> **⚠️ Important**: When using links and file paths inside documents (`.md`/`.rst`) and directives 
> (e.g., `include`, `image`, `literalinclude`) inside the documentation of the contained packages, it is 
> recommended to use relative paths (which will be understood as relative to the current document) rather 
> than absolute paths. This keeps package docs portable, and the links are valid both in the original package 
> directory and when mirrored into the main documentation location 
> (`docs/contained_package_docs_mirror/<package>/docs/`; see the 
> [Comprehensive Documentation Structure](#comprehensive-documentation-structure) section).

> **ℹ️ Note**: When using relative paths inside docstrings (e.g. to insert images), these paths are understood 
> as relative to the documentation file which includes the docstring (e.g. the file containing the `autodoc` 
> directive). In this case, absolute paths can be used.

## Key Features

### Explicit Namespace Package Configuration

The system uses explicit namespace package configuration via `namespace_packages_config.py`
(see the [How It Works](DEVELOPMENT_GUIDE.md#how-it-works) section in the Development Guide).

### Dynamic Documentation Generation

The documentation generation makes use of multiple scripts:
- **`generate_new_namespace_package_docs.py`**: Creates documentation structure for new namespace packages
  - **Template-based**: Uses consistent templates for all namespace packages (but generated files may be 
    modified as needed)
  - **Safe regeneration**: Only creates missing files if no `index.rst` is present for the namespace package
- **`generate_package_docs_assets.py`**: Runs optional package-local documentation asset hooks
  - **Package-owned**: Each package can decide whether it needs generated assets and how to create them
  - **Format-agnostic**: The hook can read any package-owned input files and write any output files in the output folder; 
    The core docs system does not prescribe a data format
- **`update_docs_index.py`**: Updates main index file by including references to newly added namespace 
  packages
- **`mirror_referenced_dirs.py`**: Mirrors (symlinks by default) the `docs` directory and other needed 
  directories from the individual packages
- **`sync_root_readme_for_docs.py`**: Copies the project root `README.md` into 
  `docs/project_overview/README.md`, adjusting guide links so that the same overview content works both when
  viewing the original `README.md` file and inside the built documentation

> **ℹ️ Note**: Documentation will only be generated for packages listed in the `NAMESPACE_PACKAGES` 
> configuration. Packages not in this list will be ignored during the documentation build process (also see 
> the [Development Guide](./DEVELOPMENT_GUIDE.md)).

> **ℹ️ Note**: The documentation generation scripts are automatically run during the documentation
> build process (see also the `Generation Scripts` subsection in the *Technical Details* section). You do not
> need to run them manually.

### Comprehensive Documentation Structure

#### Main Documentation Directory (`docs/`)
```
docs/
├── conf.py                         # Sphinx configuration using namespace_packages_config
├── index.rst                       # Main documentation index
├── generate_new_namespace_package_docs.py   # Creates structure for new namespace packages
├── generate_package_docs_assets.py # Runs optional package-local docs asset hooks
├── update_docs_index.py            # Updates navigation and indices
├── mirror_referenced_dirs.py       # Mirrors referenced directories (symlinks by default)
├── sync_root_readme_for_docs.py    # Syncs project root README into docs/project_overview
├── Makefile                        # Build commands
├── requirements.txt                # Documentation dependencies
├── project_overview/               # Synced copy of the project root README used as docs overview
├── contained_package_docs_mirror/  # Mirrored package documentation via symlinks (or copies)
│   ├── example_package/            # Example namespace package docs (representative)
│   │   ├── docs/                   # Documentation files
│   │   │   ├── index.rst           # Namespace package overview
│   │   │   ├── intro.rst           # Introduction (manual content)
│   │   │   └── api.rst             # API reference (auto-generated)
│   │   └── examples/               # Additional mirrored directory (referenced in docs)
│   └── [other_packages]/           # Other configured namespace packages
├── common/                         # Shared documentation resources
├── _static/css/
│           └── custom.css          # Custom styling
└── _build/                         # Built documentation output
```

**Notes**:
- The `docs/contained_package_docs_mirror` structure shows the build-time documentation mirrored via symlinks 
  (or copies)
- While only the `example_package` is shown in detail here, the actual documentation will contain all 
  configured namespace packages
- If you need to refer one contained package documentation from another contained package documentation, you 
  can do this by using relative paths according to this structure or by using absolute paths


#### Source Documentation Structure (`packages/<package_name>/`)
```
packages/
└── example_package/               # Example namespace package (representative)
    ├── docs/                      # Source documentation files
    │   ├── index.rst              # Namespace package overview
    │   ├── intro.rst              # Introduction (manual content)
    │   ├── api.rst                # API reference (auto-generated)
    │   ├── _on_doc_generation.py  # Optional package-local docs asset hook
    │   └── _generated/            # Generated assets created at docs build time
    ├── evaluation_results/        # Optional committed inputs for generated docs assets
    ├── docu_referenced_dirs.txt   # List of additional directories to copy
    ├── examples/                  # Example code (mirrored and referenced by docs)
    └── [other_dirs]/              # Other package directories
```


**Notes**:
- The `packages/example_package/` structure shows the source documentation that gets mirrored during build
- The `example_package` includes a small generated plot example: committed CSV data under
  `packages/example_package/evaluation_results/` is converted into an image under
  `packages/example_package/docs/_generated/` during the docs build
- **⚠️ Important**: Content should be edited in the source locations (`packages/<package_name>/docs/`), not in 
  the mirrored locations
- In case of the `example_package`, the `examples/` directory is mirrored to maintain documentation references 
  to example code. This can be achieved by listing it in the `docu_referenced_dirs.txt` file (see 
  [Referenced Directories Configuration](#referenced-directories-configuration))
- While we focus on the `example_package` package, the documentation setup of other packages is analogous


## How to Work With the Documentation System

> **⚠️ Important**: To ensure correct generation of the API documentation, the `ACCV-Lab` package (i.e. all 
> relevant namespace packages) needs to be installed first before generating the documentation. API 
> documentation for non-installed namespace packages may be empty. 
> 
> For packages installed in **normal**
> (non-editable) mode, changes to code or docstrings are only picked up after reinstalling the affected
> packages (for example via `./scripts/install_local.sh`) and then rebuilding the documentation. When using
> **editable** installs, changes to pure-Python code and docstrings are typically visible on the next docs
> build, but changes coming from compiled extensions still require rebuilding/reinstalling those extensions.

### Adding New Namespace Packages

To add a new namespace package and its documentation, see
[Adding a new Package: Step-by-Step Process](DEVELOPMENT_GUIDE.md#adding-a-new-package-step-by-step-process).
Setting up the documentation for a new namespace package (including how to build it locally) is described in
the [Building Documentation Locally](#building-documentation-locally) section.

> **ℹ️ Note**: New namespace package must be added to the `NAMESPACE_PACKAGES` list in 
> `namespace_packages_config.py` for template generation to work (see 
> [Development Guide](DEVELOPMENT_GUIDE.md) for details).

### Referenced Directories Configuration

Each namespace package can specify additional directories that are referenced by its documentation using a 
`docu_referenced_dirs.txt` file in the package root directory.

**File location**: `packages/<package_name>/docu_referenced_dirs.txt`

**Example** (for `example_package`):
```
# This file lists additional directories (besides docs) that are referenced by documentation
# The docs directory is always mirrored automatically
# Add one directory name per line, without the docs directory
# Lines starting with # are comments and are ignored

examples
```

**Default behavior**: If `docu_referenced_dirs.txt` doesn't exist, only the `docs` directory is mirrored.

**Purpose**: This ensures that e.g. files included in the documentation (images, code examples, test files, 
etc.) can still be found after the documentation is mirrored to the build location.

**Note that**:
- The `docs` directory is **always mirrored automatically** and should not be listed in this file
- Lines starting with `#` are treated as comments and ignored
- Empty lines are ignored
- Only list additional directories that are referenced by your documentation. Note that the API documentation
  does not rely on this mirroring, but is extracted from the installed packages.

### Package-Local Generated Assets

Packages can generate documentation assets during the docs build by adding an optional hook:

```text
packages/<package_name>/docs/_on_doc_generation.py
```

If present, `generate_package_docs_assets.py` imports the hook and calls:

```python
def generate_docs_assets(context):
    ...
```

The hook receives a context with package and documentation paths, including:

- `context.project_root`
- `context.package_root`
- `context.docs_root`
- `context.generated_dir`

The docs asset generator creates `context.generated_dir` before calling the hook. This directory is always:

```text
packages/<package_name>/docs/_generated/
```

It also writes a local `.gitignore` file there so generated assets remain untracked. The hook should write
generated images or other generated files directly into `context.generated_dir`, or into subdirectories below
it if a package needs additional structure.

Source documentation files remain static. For example, an `.rst` file can reference a generated image with a
normal relative path:

```rst
.. figure:: _generated/runtime_plot.png
   :alt: Runtime plot
```

Packages own the input data and generation logic. For example, a package can commit benchmark result tables
under `packages/<package_name>/evaluation_results/` and generate plots from those tables during the docs
build. If a generated asset is required by the static docs, the hook should fail with a clear error when the
required input data is missing or malformed.

> **⚠️ Important**: Documentation asset hooks must not run evaluations, benchmarks, or other measurement
> workflows. They should only regenerate documentation assets, such as plots, from data that is already
> available in the repository. It is recommended to store results in simple formats such as .csv or .md, 
> and use those as the source of truth for the plots.
>
> Keep committed plot or evaluation inputs outside the package `docs/` folder, for example under
> `packages/<package_name>/evaluation_results/`. This prevents Sphinx from discovering e.g. `.md` data tables as
> standalone documentation pages while keeping the inputs package-local.

Package-specific dependencies needed only by the hook should be declared in that package's optional
dependencies in `pyproject.toml`. The default local installation path (`./scripts/install_local.sh`) installs
optional package dependencies. If you build docs after installing packages without optional dependencies,
package-local asset hooks may fail when their optional plotting or parsing dependencies are missing.

### Building Documentation Locally

**Quick build using the script** (can be run from any directory, example shows running from the project 
root directory):
```bash
./scripts/build_docs.sh
```

**Capture Sphinx warnings in a dedicated file**:
```bash
./scripts/build_docs.sh --warning-file docs/_build/sphinx-warnings.log
```
This is useful for CI and other automation that wants to inspect warnings separately from the console output.

Relative warning-file paths are resolved from the project root. The warning file captures Sphinx warnings from
the selected builder (HTML by default, PDF with `--pdf`, or spelling with `--spelling`). In spelling mode, this
is separate from spelling findings, which are still written to `docs/_build/spelling/output.txt`.

**Manual build**:
```bash
cd docs
pip install -r requirements.txt
make html
```

**Development with auto-rebuild**:
```bash
cd docs
make livehtml
```

**Build Process Details**:
- The build process automatically runs the necessary scripts (see [Generation Scripts](#generation-scripts)) 
  in sequence
- The `html` target ensures all scripts run before building
- The `livehtml` target also runs the scripts for development builds
- Package-local docs asset hooks run before package docs are mirrored, so generated assets under
  `packages/<package_name>/docs/_generated/` are available from both the package docs source tree and the
  mirrored docs tree.
- When running spelling via the script, the generation scripts are executed first to ensure mirrored package 
  docs are up to date. Spelling findings are written to `docs/_build/spelling/output.txt`.

> **ℹ️ Note**:
> `make livehtml` watches the `docs/` tree and mirrored package documentation, but:
>  - It does **not** rerun the `sync_root_readme_for_docs.py` script when you edit the project root 
>    `README.md`. This means that if you change the root `README.md` and want to see those changes reflected 
>    in `docs/project_overview/README.md`, you need to restart `make livehtml` (or run `make sync-readme` 
>    once before continuing to work).
>  - It does **not** reinstall or rebuild packages for you. This means that if you change the docstrings in 
>    the source tree of a package, you need to reinstall the package (for example via 
>    `./scripts/install_local.sh`) and then restart `make livehtml` to see updated docstrings.
>  - It does **not** rerun package-local docs asset hooks for you after startup. This means that if you change
>    committed plot data or hook code, you need to restart `make livehtml` (or run `make generate`) to
>    regenerate plots and other generated docs assets.

### Spell-checking

**Spell-checking with sphinxcontrib-spelling**:
- Quick run using the script:
  ```bash
  ./scripts/build_docs.sh --spelling
  ```
  This runs the doc generation steps, and executes the Sphinx spelling builder. Results are written to 
  `docs/_build/spelling/output.txt`.

- Manual run:
  ```bash
  pip install sphinxcontrib-spelling pyenchant
  cd docs
  make generate
  make spelling
  ```
  The spelling report will be at `_build/spelling/output.txt`.

- **Note**: On some Linux systems you may need a system-level Enchant library. If you see errors related to 
Enchant, install it via your package manager (e.g., `sudo apt install enchant-2` on Debian/Ubuntu).

- Whitelisting words (project-specific words or general words which are not part of the used dictionary):
  - Add words (one per line) to `docs/spelling_wordlist.txt` (e.g., `namespace`, `ACCV-Lab`).
  - Re-run the spelling check. These words will no longer be flagged.

### Customizing Documentation

#### For Individual Namespace Packages

Each namespace package documentation is located in `packages/<package_name>/docs/` and is automatically 
mirrored to `docs/contained_package_docs_mirror/<package_name>/docs/`.

**Example structure** (for `example_package`):
- **`intro.rst`**: Manual introduction and overview (customize this)
- **`api.rst`**: Auto-generated API reference (the file can be modified e.g. to add short sections, but the 
  API is auto-extracted)
- **`index.rst`**: Table of contents for the namespace package as well as the "entry point". This file needs 
  to be present in the folder.

> **⚠️ Important**: When editing documentation, work directly in the `packages/<package_name>/docs/` 
> directory, not in the `docs/contained_package_docs_mirror/<package_name>/docs/` directory.
> 
> **Example**: For `example_package`, edit files in `packages/example_package/docs/`, not in 
> `docs/contained_package_docs_mirror/example_package/docs/`.

Note that `index.rst` is the "entry point" which is referenced by the `ACCV-Lab` documentation and needs to 
be present. All other files may be changed (e.g. removed, renamed) and new files may be added. In this case, 
ensure that within the namespace package, all the needed files are referenced from `index.rst` (either 
directly or indirectly).

#### Modifying the Common Documentation & Style

Possible modifications include:
- Modify `docs/conf.py` for Sphinx configuration
- Edit `docs/_static/custom.css` for styling
- Add shared content in `docs/common/`
- Update `docs/index.rst` for main navigation

> **ℹ️ Note**: The high-level project overview used in the documentation is sourced from the project root
> `README.md` and mirrored into `docs/project_overview/README.md` by `sync_root_readme_for_docs.py`. Edit the
> root `README.md` rather than the mirrored copy; the docs build will keep them in sync.

> **⚠️ Important**: When modifying the documentation in these ways, make sure to not break the documentation 
> generation for any of the namespace packages.


## Custom Sphinx Extensions and Styling

The build adds several custom features that improve code inclusion, device notes in API docs, and table 
readability. These are configured in `docs/conf.py`, implemented in:
- `docs/_ext/note_literalinclude.py` (note-literalinclude directive)
- `docs/_ext/module_docstring.py` (module-docstring directive)
- `docs/_ext/markdown_note_admonitions.py` (Markdown “Note/Important” → admonitions)
- `docs/conf.py` (autodoc device-note processing and configuration)
- `docs/_static/css/custom.css` (styling for `.device-note`, `no-scroll`, and autosummary tables)

and styled via `docs/_static/css/custom.css`.

### The `note-literalinclude` Directive

- Include code files and automatically highlight “note blocks” inside the code (e.g., lines starting with a 
  note tag and their continued lines).
- Example:

```rst
.. note-literalinclude:: ../examples/basic_usage.py
   :language: python
   :caption: Simple usage example
   :note-tag: #@NOTE;# @NOTE   # optional, defaults to '#@NOTE;# @NOTE'
   :note-cont: #              # optional, defaults to '#'
   :highlight-notes: true     # optional, defaults to true
```

- Behavior: Lines starting with any tag in `:note-tag:` (default: `#@NOTE` or `# @NOTE`) are highlighted, 
  and highlighting continues on subsequent lines that start with the continuation prefix (default: `#`) 
  until a non-matching line is encountered.
- If `:highlight-notes:` is `false`, it behaves like a normal `literalinclude`.

### The `module-docstring` Directive

- Render the top-level docstring of a Python module directly:

```rst
.. module-docstring:: ../../packages/example_package/accvlab/example_package/__init__.py
```

- Behavior: Reads and parses the given module file and renders only its module-level docstring at the place 
  of the directive. Note that in contrast to the API documentation, the docstring is read directly from the
  indicated file and not from the installed package. This is e.g. useful for docstrings originating from
  examples, which are not part of the installed package.

### Markdown “Note/Important” Conversion

- Converts Markdown blockquoted notes into Sphinx admonitions at build time, while keeping the Markdown source
  readable in GitHub/IDEs.
- Implemented in: `docs/_ext/markdown_note_admonitions.py`
- How to write notes in Markdown (examples):

```md
> **ℹ️ Note**: Short tip for the reader.

> **⚠️ Important**: Crucial warning users must not miss.
```

- Accepted variants:
  - With or without symbols: `ℹ️`, `ⓘ`, `(i)`, `i`, `!`, `⚠️`
  - With or without bold around the word
  - With or without a trailing colon after “Note”/“Important”
  - Examples that also convert:

```md
> Note: ...
> **Note** ...
> ! Important ...
> (i) Note ...
> ⚠️ Important ...
```

> **ℹ️ Note**: Multi-line notes are supported. In this case, all lines must start with `>`, and the whole
> so defined block will be converted to an admonition.

> **ℹ️ Note**: When using the MyST Markdown parser (used in the Sphinx documentation), you can also use the
> built-in fenced admonitions (for example `` ```{note} `` / `` ```{important} `` as opening fences). However, 
> the blockquote-based syntax described here is meant to be compatible with plain Markdown viewers (such as 
> GitHub or IDEs), also highlighting the note/important text if it is not converted to a Sphinx admonition.
> Therefore, this blockquote-based syntax is preferred for files that should work both as standalone `.md` 
> files and as part of the built documentation. If the file is only used as part of the built documentation,
> the fenced admonitions should be used instead.

- Scope:
  - Only blockquotes are converted (lines starting with `>`). Plain list items or inline text containing
    “Note:”/“Important:” are not converted.
  - Nested blockquotes inside lists are supported (the quoted block is converted).

### Device Notes in Autodoc Docstrings

- Autodoc-rendered Python docstrings can show a highlighted “device” note at the top by adding one of the 
  following markers as a separate line in the docstring:
  - `:gpu:` → renders a note that inputs are expected on GPU
  - `:cpu:` → renders a note that inputs are expected on CPU
  - `:device: <TEXT>` → renders a note that inputs are expected on the specified device text (e.g., `GPU`, 
    `CPU`)
- Example (Python):

```python
def my_function(x):
    """Short description.

    :gpu:

    More details about usage...
    """
    ...
```

- Behavior: During autodoc processing, these markers are detected and replaced with an emphasized note at 
  the top of the rendered docstring. The note is styled with the `.device-note` CSS class.

### Table Classes and Wrapping

- To avoid horizontal scrolling and allow wrapping for specific tables, use the `no-scroll` table class:

```rst
.. table:: Example table
   :class: no-scroll

   +------------------+---------------------+
   | Column A         | Column B            |
   +------------------+---------------------+
   | Long content ... | More long content … |
   +------------------+---------------------+
```

- Autosummary tables are styled to wrap cell contents automatically; no additional class is required.


## Technical Details

### Sphinx Configuration (`docs/conf.py`)

The configuration includes:
- **Explicit namespace package import** from `namespace_packages_config.py` (in the helper scripts)
- **Local extension path setup** (`docs/_ext`); autodoc imports installed ACCV-Lab packages
- **Extension configuration** for autodoc, autosummary, and more
- **Theme configuration** with Read the Docs theme
- **Cross-reference setup** with intersphinx

### Build System

#### Makefile (`docs/Makefile`)
- **Standard Sphinx targets**: html, clean, etc.
- **Custom targets**: 
  - `generate`: Runs all documentation generation scripts for namespace packages
  - `sync-readme`: Syncs the project root `README.md` into `docs/project_overview/README.md` with adjusted 
    links
  - `livehtml`: Development build with auto-reload (also runs `sync-readme` and `generate` before starting)
- **Development support**: Live reload and other dev features

#### Build Script (`scripts/build_docs.sh`)
- **Dependency installation**: Installs documentation requirements from `docs/requirements.txt`
- **Complete build process**: Generation + Sphinx build
- **Warning capture**: Can write Sphinx warnings from the build to a dedicated file via `--warning-file`
- **Browser integration**: Optionally opens documentation
- **Error handling**: Provides clear error messages
- **Location**: Can be run from any directory (script automatically determines project root)

### Documentation Dependencies

The `docs/requirements.txt` file contains the Python packages needed for building documentation (separate from
the per-package runtime dependencies defined in each package's `pyproject.toml`):
- Sphinx and related extensions
- Theme packages
- Other documentation-specific dependencies

Package-specific docs asset dependencies belong to the corresponding package's optional dependencies. This keeps the 
global documentation requirements focused on the Sphinx build itself while allowing package-owned hooks to declare their 
own plotting or data-processing dependencies.

### File Descriptions

#### Core Configuration Files

- **`namespace_packages_config.py`**: Shared namespace package configuration (PROJECT ROOT)
- **`docs/conf.py`**: Sphinx configuration using shared config
- **`docs/index.rst`**: Main documentation index with navigation. Note that this file contains a template for
  including the namespace packages, as well as an auto-generated table of contents for the namespace packages
  using the template (see the comments in the file for more details).
- **`docs/requirements.txt`**: Documentation dependencies

#### Generation Scripts

The documentation build relies on the helper scripts described in
the [Dynamic Documentation Generation](#dynamic-documentation-generation) section. These scripts are invoked
automatically as part of the docs build; you normally do not need to run them manually.

#### Documentation Structure

- **`packages/<package_name>/docs/index.rst`**: Namespace package overview (source)
- **`packages/<package_name>/docs/intro.rst`**: Manual introduction content (source)
- **`packages/<package_name>/docs/api.rst`**: Auto-generated API reference (source)
- **`packages/<package_name>/docs/_on_doc_generation.py`**: Optional hook for package-local generated docs
  assets
- **`packages/<package_name>/docs/_generated/`**: Generated documentation assets created during docs
  generation and ignored by Git
- **`packages/<package_name>/evaluation_results/`**: Optional package-owned committed inputs for generated
  docs assets, such as benchmark tables used for plots
- **`packages/<package_name>/docu_referenced_dirs.txt`**: List of directories containing files used in the 
  documentation in addition to `docs` (to mirror into the documentation source directory).
- **`docs/contained_package_docs_mirror/<package_name>/docs/`**: Mirrored documentation (symlink to the 
  individual packages by default)
- **`docs/contained_package_docs_mirror/<package_name>/[other_dirs]/`**: Additional mirrored directories as 
  specified in `docu_referenced_dirs.txt`

The individual `.rst` files for individual namespace packages can be changes (content, filenames, number of 
files, etc.), but `index.rst` is required as it is used as the "entry point" from the overarching 
documentation into the individual namespace package documentation.

#### Supporting Files

- **`docs/common/`**: Shared documentation resources
- **`docs/_static/custom.css`**: Custom styling (applied to all namespace packages)

#### Build and Deployment Files

- **`scripts/build_docs.sh`**: Comprehensive build script (can be run from any directory)
- **`docs/Makefile`**: Standard Sphinx build commands
