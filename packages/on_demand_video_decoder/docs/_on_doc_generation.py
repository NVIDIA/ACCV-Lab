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

_RESULTS_SUBDIR = Path("evaluation_results")
_GENERATED_IMAGE_SUBDIR = Path("evaluation")

_REQUIRED_CSV_INPUTS = (
    "cross_decoder/hevc_gop30_random_access.csv",
    "cross_decoder/hevc_gop30_sequential.csv",
    "video_config_sweep/gop_random_access.csv",
    "video_config_sweep/gop_sequential.csv",
    "video_config_sweep/bframes_random_access.csv",
    "video_config_sweep/bframes_sequential.csv",
    "video_config_sweep/codec_random_access.csv",
    "video_config_sweep/codec_sequential.csv",
    "streampetr_training/setup_a.csv",
    "streampetr_training/setup_b.csv",
)

_REQUIRED_IMAGE_NAMES = (
    "cross_decoder.png",
    "video_config_gop.png",
    "video_config_bframes.png",
    "video_config_codec.png",
    "streampetr_training.png",
)


def _validate_csv_inputs(input_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(
            "Required committed CSV input directory is missing for on_demand_video_decoder docs asset generation: "
            f"{input_dir}."
        )

    missing = [input_dir / rel for rel in _REQUIRED_CSV_INPUTS if not (input_dir / rel).exists()]
    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Missing required committed CSV input file(s) for on_demand_video_decoder docs asset generation:\n"
            f"{missing_list}"
        )


def _validate_images(output_dir: Path) -> None:
    missing = [output_dir / name for name in _REQUIRED_IMAGE_NAMES if not (output_dir / name).exists()]
    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "on_demand_video_decoder docs asset generation did not produce all images referenced by evaluation.rst:\n"
            f"{missing_list}"
        )


def generate_docs_assets(context: Any) -> None:
    input_dir = context.package_root / _RESULTS_SUBDIR
    output_dir = context.generated_dir / _GENERATED_IMAGE_SUBDIR

    _validate_csv_inputs(input_dir)

    evaluation_dir = context.package_root / "evaluation"
    sys.path.insert(0, str(evaluation_dir))
    import plot_decoder_evaluation

    plot_decoder_evaluation.generate_all(input_root=input_dir, output_dir=output_dir)

    _validate_images(output_dir)
