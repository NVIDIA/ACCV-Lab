# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("accvlab.dali_pipeline_framework")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Import all child packages
from . import inputs
from . import internal_helpers
from . import operators_impl
from . import pipeline
from . import processing_steps

__all__ = [
    "__version__",
    'inputs',
    'internal_helpers',
    'operators_impl',
    'pipeline',
    'processing_steps',
]
