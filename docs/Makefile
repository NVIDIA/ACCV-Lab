# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

MAKEFILE_DIR := $(patsubst %/,%,$(abspath $(dir $(lastword $(MAKEFILE_LIST)))))
CURRENT_DIR  := $(patsubst %/,%,$(abspath $(CURDIR)))

ifneq ($(CURRENT_DIR),$(MAKEFILE_DIR))
$(error Please run make from $(MAKEFILE_DIR). Example: 'cd $(MAKEFILE_DIR) && make clean html' or 'make -C $(MAKEFILE_DIR) clean html')
endif

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Generate namespace package documentation before building
generate:
	python3 mirror_referenced_dirs.py
	python3 generate_new_namespace_package_docs.py
	python3 update_docs_index.py

# Sync the root README into the docs tree before building
sync-readme:
	@echo "Syncing root README into docs/project_overview..."
	@python3 sync_root_readme_for_docs.py

# Build HTML documentation with auto-generation
html: sync-readme generate
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean build directory and generated files (full removal to avoid stale sidebar/toctree)
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf $(BUILDDIR)/
	rm -rf api/generated/
	rm -rf ../packages/*/docs/generated/

# Auto-build documentation (watches for changes)
livehtml: sync-readme generate
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 