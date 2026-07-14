#!/usr/bin/env python3
"""
Simple Example for ACCV-Lab Example Package.

This example demonstrates basic usage of the example package functions.
It's designed to show how examples can be included in documentation.
"""

import accvlab.example_package as example_pkg
import torch

# @NOTE
# This note will be highlighted in the documentation using the note-literalinclude directive (see the
# Documentation Setup Guide for more details).


def simple_example():
    """Demonstrate simple function usage."""
    print("=== Simple Example ===")

    # Use the hello function
    message = example_pkg.hello_examples()
    print(f"Message: {message}")

    vector = torch.tensor([1.0, 2.0, 3.0])
    print(f"C++ vector sum: {example_pkg.cpp_vector_sum(vector)}")

    a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    b = torch.tensor([4.0, 5.0, 6.0], device="cuda")
    print(f"CUDA vector product: {example_pkg.cuda_vector_multiply(a, b)}")


if __name__ == "__main__":
    simple_example()
