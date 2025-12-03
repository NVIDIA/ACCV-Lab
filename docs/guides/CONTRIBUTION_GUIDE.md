# Contribution Guide

This guide contains the guidelines for contributing to ACCV-Lab. It focuses on what should be done rather than 
how to do it, as this is covered in the other guides.

> **ℹ️ Note**: For technical details, see:
> - [Installation Guide](INSTALLATION_GUIDE.md) - How to install the package
> - [Development Guide](DEVELOPMENT_GUIDE.md) - Package structure and build system
> - [Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md) - Documentation creation and management
> - [Formatting Guide](FORMATTING_GUIDE.md) - Code formatting


## Overview

When contributing to ACCV-Lab, your contributions should include comprehensive testing, thorough 
documentation, and examples to both showcase the achieved benefit and to provide guidance on how to use the 
contained functionality.

## Testing Requirements

### Unit Testing

Contributions with new functionality should include comprehensive unit tests that cover:

- **Core Functionality**: Test all major features and functions
- **Edge Cases**: Test boundary conditions, special cases, and error conditions
- **Failure Handling**: Verify that your code properly handles and reports errors

### Testing Approaches

Use appropriate testing strategies based on your contribution:

- **Deterministic Testing**: For algorithms and mathematical functions, test against known inputs and expected 
  outputs.
- **Randomized Testing**: Where appropriate and feasible, test against random inputs and verify expected 
  output (e.g. testing an optimized implementation using a baseline reference implementation).

### Testing Functions which Implement Gradient Propagation

When implementing differentiable building blocks (e.g. as `PyTorch` custom operators), it is important to
test not only the forward pass, but also the gradient computation. 

When testing gradient propagation functionality, it is best to ensure that the gradient which is input to the 
gradient computation to test is not simply 1.0 in each case (or for each element e.g. of a tensor). This can 
e.g. be achieved by applying another function to the output of the function to test in the forward step.

### Test Structure

- For Python
  - Place tests in `packages/<package_name>/tests/` directory (will be automatically discovered by the 
    repository test runner)
  - Use pytest
- For C++/CUDA tests, ensure your build scripts return appropriate error codes on test failure

> **ℹ️ Note**: If you would like to include "manual" tests (i.e. tests which one can run during development but
> which are not part of the unit tests), you can place them in a different directory (not `tests/`). This will
> prevent them from being automatically run by the repository test runner.

## Documentation Requirements

### General Description

Please add documentation to your new namespace package. Provide a clear overview that explains:
- **Problem Statement**: What problem does your contribution solve?
- **Use Cases**: When and why would someone use this functionality?
- **Details of the Approach** from a user perspective: Include info e.g. on
  - Limitations (e.g. which problem sizes are supported)
  - Performance
    - Performance gain of using the optimized implementation (e.g. as part of demo description)
    - Are there cases for which the implementation is not optimal and which may see low performance?
      If so, provide guidance on when to use the implementation and for which cases it is not a good fit.

### Detailed Specifications

Document all aspects of your implementation, e.g.:

- **Input Parameters**: Types, formats, other considerations/constraints, ...
- **Output Values**: Return types, formats, and interpretation
- **Side Effects**: Any modifications to input data or (e.g. object) state
- **Constraints**: Limitations on input sizes, data types, or operating conditions

Note that this kind of documentation should typically be done as part of the API documentation by adding
docstrings and type hints to the code.

### Limitations and Constraints

Be transparent about limitations:

- **Performance Limitations**: When your implementation may not be optimal (e.g., overhead for small problems, 
  memory trade-offs)
- **Functional Limitations**: Size limits, data type restrictions, or unsupported use cases

### API Documentation

- API documentation is automatically extracted from your code
- Ensure your code includes proper docstrings and type hints
- Follow the documentation structure described in the 
  [Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md).

### Documentation Checks

Before submitting your contribution, please ensure that the documentation can be build successfully without 
errors. Please also check any warnings as these may indicate issues which may make the documentation 
hard to read or understand (such as broken links, missing images or code, ...), and fix the warnings where 
feasible. This helps to ensure that the documentation is of high quality, and also helps with pinpointing any
warnings introduced by future changes to the documentation.

Please also check for spelling errors (see 
the [section on spell-checking](DOCUMENTATION_SETUP_GUIDE.md#spell-checking) in the 
[Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md)).
Note that some of the reported errors may not represent actual spelling errors, but rather words picked up 
from the code, file path, specific abbreviations, etc. However, if actual spelling errors are present, they 
should be fixed. If correctly spelled words are picked up as errors, they should be added to the spelling
word list (see the [section on spell-checking](DOCUMENTATION_SETUP_GUIDE.md#spell-checking) in the 
[Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md)).

## Examples and Tutorials

Provide practical demonstrations of your contribution through one or more of the following:

### Examples

Create simple, self-contained examples that demonstrate basic usage:
- **Location**: `packages/<package_name>/examples/`
- **Scope**: Focus on single functions or simple workflows
- **Purpose**: Demonstrate basic usage patterns; potentially show the benefits of the approach

### Tutorials

Develop comprehensive tutorials that guide users through your functionality:
- **Content**: Include both code and detailed explanations
- **Progression**: Start with simple concepts and build to complex applications
- **Variety**: Include both toy examples and real-world scenarios
- **Ordering**: Suggest a logical progression through your tutorials
- **Format**: You may choose a format of your choice (e.g. Sphinx, Jupyter notebooks, ...). 

> **ℹ️ Note**: For the tutorials, the examples and demos can be "re-used". For example, using Sphinx, you can 
> write a description and insert code (snippets) and notes in a structured way.
> Please also refer to the [Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md) for further details.
> The [note-literalinclude](DOCUMENTATION_SETUP_GUIDE.md#the-note-literalinclude-directive) is especially 
> useful for this purpose, as it allows to conveniently highlight specific comment blocks, which can be used 
> as part of the tutorial.

### Demos

Create demonstrations for real-world scenarios that showcase the benefits of the contribution (for example
faster run time, less memory consumption, ...).

Note that demos for the contained packages will be added in the future.


## Code Quality Standards

### Formatting

- Follow the formatting standards outlined in the [Formatting Guide](FORMATTING_GUIDE.md)
- Use the automated formatting tools provided by the project
- Ensure consistent code style

### Structure

- Follow the package structure described in the [Development Guide](DEVELOPMENT_GUIDE.md)
- Organize code logically within the appropriate namespace package
- Use clear, descriptive names for functions, classes, and variables

### Dependencies

- Minimize external dependencies where possible
- Declare required runtime dependencies in each package's `pyproject.toml` under `[project.dependencies]`
- Use `[project.optional-dependencies]` in `pyproject.toml` for non-essential or optional features
- Follow the dependency management patterns established in the project

## Getting Help

If you need assistance with any aspect of contributing, please consider the following options:

- Check the existing guides for technical implementation details
- Review existing contained packages for examples of good practices and patterns
- Consult the project maintainers for further questions and/or guidance on complex contributions

## Signing Your Work & Developer Certificate of Origin (DCO)

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your 
  original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```text
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## Summary Checklist

Before submitting your contribution, please consider the following checklist:

- [ ] **Comprehensive Unit Tests**: Covering core functionality, edge cases, and error handling
- [ ] **Complete Documentation**: General description, detailed specifications, and performance metrics (where relevant)
- [ ] **Limitations Documented**: Clear explanation of constraints and trade-offs
- [ ] **Examples Provided**: Simple demonstrations of basic usage (where relevant)
- [ ] **Tutorials Created**: Detailed guides for learning and understanding (where relevant)
- [ ] **Demos Available**: Real-world showcases of key features (where relevant)
- [ ] **Code Formatted**: Following project formatting standards
- [ ] **Dependencies Managed**: Properly configured in `pyproject.toml`
- [ ] **Repository Test Runner Passing**: Verified functionality and integration across all namespace 
  packages (scripts/run_tests.sh). Note that this will be automated in the future
- [ ] **Documentation Build Checked**: Documentation building successful and the checks outlined in the 
  [Documentation Checks](#documentation-checks) section above are performed.
- [ ] **Signed-Off Commits**: All contained commits are signed-off

Note that the **checklist is a guideline**. Individual points may be omitted if this makes sense for the 
contribution (e.g. no tutorials are provided because the functionality is easy to use and examples are 
sufficient for comprehensive understanding of how to use the package). Please use your best judgment and 
add comments in merge requests to explain why you have omitted certain points where this may be not obvious.

**Remember**: The goal is to create contributions that are not only functional but also well-tested, 
well-documented, and easy to use.



