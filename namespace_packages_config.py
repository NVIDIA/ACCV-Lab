#!/usr/bin/env python3
"""
Shared configuration for ACCV-Lab namespace packages.

This file defines the list of namespace packages that should be:
1. Built and installed by individual package setup.py files
2. Documented by the Sphinx documentation system

Add new namespace packages to the NAMESPACE_PACKAGES list below.
"""

# List of all ACCV-Lab namespace packages
# Each namespace package should:
# - Be a directory under the packages/ subdirectory
# - Have a pyproject.toml and setup.py file for building
# - Be added to this list to be included in builds and documentation
# Please note that:
# - Packages that are not listed here will be ignored when installing all packages, building the
#   documentation, running the tests, etc.
# - The order in which the packages are listed here is the order in which they will be installed, and in which
#   they will appear in the documentation.
NAMESPACE_PACKAGES = [
    # The commented out packages below this line are examples (see the development guide):
    # 'accvlab.example_package',
    # 'accvlab.example_skbuild_package',
    'accvlab.on_demand_video_decoder',
    'accvlab.batching_helpers',
    'accvlab.multi_tensor_copier',
    'accvlab.dali_pipeline_framework',
    'accvlab.draw_heatmap',
    'accvlab.optim_test_tools',
    # Add new namespace packages in the same way as above
]


def get_namespace_packages():
    """Get the list of configured namespace packages."""
    return NAMESPACE_PACKAGES.copy()


def get_package_names():
    """Get just the package names (last part after the dot)."""
    return [pkg.split('.')[-1] for pkg in NAMESPACE_PACKAGES]


if __name__ == "__main__":
    # When run directly, show the configuration
    print(f"Configured namespace packages: {len(NAMESPACE_PACKAGES)}")
    for i, pkg in enumerate(NAMESPACE_PACKAGES, 1):
        print(f"  {i}. {pkg}")
