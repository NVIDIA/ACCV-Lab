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

from pathlib import Path
import sys
from typing import Any

_RESULTS_SUBDIR = Path("evaluation_results") / "polyline_runtime_evaluation"
_GENERATED_IMAGE_SUBDIR = Path("polyline_runtime_evaluation")
_DOC_BATCH_SIZES = [1, 64]
_DOC_REQUIRED_MARKDOWN_METRICS = (
    "runtime_shapely",
    "runtime_cpu",
    "runtime_cuda",
    "speedup_cpu_vs_shapely",
    "speedup_cuda_vs_shapely",
    "speedup_cuda_vs_cpu",
)
_DOC_REQUIRED_IMAGE_NAMES = tuple(
    f"batch_{batch_size}_{plot_kind}_comparison.png"
    for batch_size in _DOC_BATCH_SIZES
    for plot_kind in ("runtime", "speedup")
)


def _required_markdown_paths(input_dir: Path) -> list[Path]:
    return [
        input_dir / f"batch_{batch_size}_{metric_name}.md"
        for batch_size in _DOC_BATCH_SIZES
        for metric_name in _DOC_REQUIRED_MARKDOWN_METRICS
    ]


def _validate_required_markdown_inputs(input_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(
            "Required committed Markdown input directory is missing for lane_helpers docs asset generation: "
            f"{input_dir}."
        )

    missing_inputs = [path for path in _required_markdown_paths(input_dir) if not path.exists()]
    if missing_inputs:
        missing_list = "\n".join(f"  - {path}" for path in missing_inputs)
        raise FileNotFoundError(
            "Missing required committed Markdown input file(s) for lane_helpers docs asset generation:\n"
            f"{missing_list}"
        )


def _validate_required_images(output_dir: Path) -> None:
    missing_outputs = [
        output_dir / image_name
        for image_name in _DOC_REQUIRED_IMAGE_NAMES
        if not (output_dir / image_name).exists()
    ]
    if missing_outputs:
        missing_list = "\n".join(f"  - {path}" for path in missing_outputs)
        raise FileNotFoundError(
            "Polyline runtime docs asset generation did not produce all images referenced by introduction.rst:\n"
            f"{missing_list}"
        )


def generate_docs_assets(context: Any) -> None:
    input_dir = context.package_root / _RESULTS_SUBDIR
    output_dir = context.generated_dir / _GENERATED_IMAGE_SUBDIR

    _validate_required_markdown_inputs(input_dir)

    evaluation_dir = context.package_root / "evaluation"
    sys.path.insert(0, str(evaluation_dir))
    import plot_shapely_evaluation

    plot_shapely_evaluation.plot_from_markdown_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_sizes=_DOC_BATCH_SIZES,
        annotate_plots=True,
    )
    _validate_required_images(output_dir)
