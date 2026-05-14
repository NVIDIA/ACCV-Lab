#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command batching helpers for formatter invocations."""

import subprocess

DEFAULT_BATCH_SIZE = 100


def iter_file_batches(files, batch_size=DEFAULT_BATCH_SIZE):
    """Yield fixed-size batches from files."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    batch = []
    for file_path in files:
        batch.append(file_path)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def run_formatter_in_batches(formatter_command, files, batch_size=DEFAULT_BATCH_SIZE):
    """Run formatter_command with files appended in fixed-size batches."""
    for batch in iter_file_batches(files, batch_size):
        subprocess.run(formatter_command + batch, check=True)
