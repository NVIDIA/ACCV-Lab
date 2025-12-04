# Code Formatting Guide

The ACCV-Lab project uses automated code formatting to maintain consistent code style across all namespace 
packages. The formatting system integrates with the shared configuration system and automatically discovers 
all namespace packages. The project uses `clang-format` for C++/CUDA code formatting and `black` for Python 
code formatting.

## Running Code Formatting

The project provides a unified formatting script that automatically discovers namespace packages and handles 
both Python and C++/CUDA code formatting.

### Main Formatting Script

The primary script for all formatting operations is `scripts/format.sh`:

```bash
# Format everything (common code + all namespace packages)
./scripts/format.sh

# Format Python code only
./scripts/format.sh --python

# Format C++/CUDA code only
./scripts/format.sh --cpp

# Format common code only (root files, docs, shared build_config)
./scripts/format.sh --common-only

# Format common code + all namespace packages
./scripts/format.sh --include-packages

# Format a specific namespace package
./scripts/format.sh --package <package_name>

# Examples:
./scripts/format.sh --package example_package
./scripts/format.sh --package example_skbuild_package
./scripts/format.sh --cpp --package batching_helpers
./scripts/format.sh --python --common-only
```

The `scripts/format.sh` script provides:
- **Unified interface**: Single script for all formatting operations
- **Flexible targeting**: Format everything, common code only, specific packages, or language-specific
- **Automatic discovery**: Finds all namespace packages automatically
- **Combined formatting**: Handles both Python (`black`) and C++/CUDA (`clang-format`) formatting
- **Help system**: Use `--help` for detailed usage information

### Language-Specific Options

The main script supports language-specific formatting:

#### Python Formatting (Black)
```bash
# Format Python code only (common + all packages)
./scripts/format.sh --python

# Format Python code in common areas only
./scripts/format.sh --python --common-only

# Format Python code for specific namespace package
./scripts/format.sh --python --package <package_name>
```

#### C++/CUDA Formatting (clang-format)
```bash
# Format C++ code only (all packages, since no common C++ exists)
./scripts/format.sh --cpp

# Format C++ code for specific namespace package
./scripts/format.sh --cpp --package <package_name>
```

### Advanced Usage

For more granular control, you can also use the individual formatting scripts directly:

```bash
# Individual scripts (located in scripts/formatting/)
./scripts/formatting/black_format_package.sh --include-subpackages
./scripts/formatting/black_format_subpackage.sh <package_name>
./scripts/formatting/clang_format_package.sh --include-subpackages
./scripts/formatting/clang_format_subpackage.sh <package_name>
./scripts/formatting/format_subpackage.sh <package_name>
```

### External Implementation Support

The formatting scripts automatically include external implementation files located in 
`packages/<package>/ext_impl/`. These directories contain namespace package-specific external implementations 
(e.g., custom CUDA kernels) and are formatted alongside the corresponding namespace package code:

- When formatting a specific namespace package, its external implementation in `packages/<package>/ext_impl/` 
  is also formatted
- When using `--include-packages`, all external implementations are formatted along with their respective 
  namespace packages
- Both C++/CUDA and Python files in external implementations are supported

## Style Configuration

The project uses custom style configurations for both Python and C++/CUDA formatters to maintain consistent 
code style across all namespace packages.

### C++/CUDA Style Configuration (clang-format)

**Location**: `.clang-format` file in the project root

The C++/CUDA formatting style is based on Google style with custom overrides for indentation, column limits, 
pointer alignment, and other formatting preferences.

> **ℹ️ Note**: If the configuration needs to be adjusted for a specific directory, a `.clang_format` file can 
> be placed there, and adjustments to the style made there will be applied to that directory (including 
> sub-directories).

### Python Style Configuration (Black)

**Location**: `pyproject.toml` file in the project root

The Python formatting style uses Black formatter with custom settings for line length, Python version targets, 
and exclusion patterns.

> **ℹ️ Note**: The `pyproject.toml` file at the project root is dedicated to defining the formatting style. 
> Black detects the project root by looking for a `.git` directory, `.hg` directory, or a `pyproject.toml` 
> file containing a `[tool.black]` section. It uses the configuration from the root `pyproject.toml` file, 
> even when formatting files in subdirectories. Individual namespace packages have their own `pyproject.toml` 
> files for build configuration, but these do not contain `[tool.black]` sections. This ensures Black 
> consistently uses the root configuration across all namespace packages, maintaining uniform formatting 
>throughout the project.

> **ℹ️ Note**: Unlike `clang-format`, the format is defined in the top-level `pyproject.toml` and cannot be 
> adjusted in sub-directories (except for excluding them).

## Preventing List Collapse with Trailing Commas

For multi-line lists, arrays, or similar structures, you can prevent them from being collapsed into a single 
line by adding a trailing comma after the last element. This is particularly useful when initializing 
multi-dimensional data structures such as tensors or arrays.

**PyTorch tensor initialization:**
```python
# This will stay multi-line due to the trailing comma
tensor = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

# Without trailing comma, this might be collapsed to a single line
tensor = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]  # No trailing comma
])
```

This technique works with both `black` (Python) and `clang-format` (C++) formatters and is often preferable to 
completely disabling formatting for the entire block (see below).

## Exceptions: Disabling Formatting

### Disabling for Code Snippets Inside a File

In rare cases, automated formatting may result in code that is more difficult to read (for example for complex 
macros in C++). In such situations, you can disable formatting for specific code regions to preserve or 
improve readability. Use this sparingly, and only when automated formatting would significantly reduce code 
readability. Always prefer the project's standard formatting unless there is a strong reason to opt out for a 
specific region.

#### C++/CUDA (clang-format)
Wrap the code region you want to exclude from formatting with:

```cpp
// clang-format off
... your code ...
// clang-format on
```

#### Python (black)
Wrap the code region you want to exclude from formatting with:

```python
# fmt: off
... your code ...
# fmt: on
```

### Disabling for Whole Directories

In rare cases, it may be advisable to disable the formatting for whole directories. This may e.g. be the 
case if the directory contains external code which should remain as-is. Note that in these cases, it 
may be advisable to obtain the external dependency as part of the setup instead of committing it as part of 
the repository.

#### C++/CUDA (clang-format)

In the directory which should be ignored, add a `.clang-format` file with the content:

```
DisableFormat: true
```

#### Python (black)

To exclude directories from formatting using the Python `black` formatter, they need to be listed as to 
exclude in the root-level `pyproject.toml` file. Please see the `extend-exclude` argument which is used in 
the `[tool.black]` TOML table.


