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

import numpy as np

_LARGE_SPEEDUP_THRESHOLD = 1_000.0


# Helper function for formatting speedup values in result tables.
def _format_speedup_value(value: float) -> str:
    if abs(value) >= _LARGE_SPEEDUP_THRESHOLD:
        return f"{value:.2e}"
    return f"{value:.2f}"


# Helper function for formatting one measured metric as a Markdown table.
def _format_table(
    data: np.ndarray,
    nums_points: list[int],
    nums_distances: list[int],
    *,
    scientific: bool,
) -> str:
    rows = [
        "| # Points (down) / # Distances (right) | " + " | ".join(str(item) for item in nums_distances) + " |"
    ]
    rows.append("| :----- |" + " :-----: |" * len(nums_distances))
    for points_idx, num_points_current in enumerate(nums_points):
        values = []
        for distances_idx in range(len(nums_distances)):
            value = data[points_idx, distances_idx]
            if scientific:
                values.append(np.format_float_scientific(value, precision=3))
            else:
                values.append(_format_speedup_value(value))
        rows.append(f"| {num_points_current} | " + " | ".join(values) + " |")
    table = "\n".join(rows)
    return table


# Helper function for writing one Markdown table to disk.
def _write_markdown(
    data: np.ndarray,
    nums_points: list[int],
    nums_distances: list[int],
    *,
    filename: Path,
    scientific: bool,
) -> None:
    table = _format_table(data, nums_points, nums_distances, scientific=scientific)
    filename.write_text(table + "\n", encoding="utf-8")


# Helper function for writing the Markdown table output for one metric.
def _write_metric_outputs(
    data: np.ndarray,
    nums_points: list[int],
    nums_distances: list[int],
    *,
    filename_stem: Path,
    scientific: bool,
) -> None:
    _write_markdown(
        data,
        nums_points,
        nums_distances,
        filename=filename_stem.with_suffix(".md"),
        scientific=scientific,
    )


# Entry point: write all Markdown tables for one evaluated batch size.
def write_batch_results(
    output_dir: Path,
    batch_size: int,
    nums_points: list[int],
    nums_distances: list[int],
    shapely_runtime_ms: np.ndarray | None,
    cpu_runtime_ms: np.ndarray,
    cuda_runtime_ms: np.ndarray,
    skip_shapely: bool,
    assert_results: bool,
    max_abs_diff_cpu: np.ndarray | None,
    max_abs_diff_cuda: np.ndarray | None,
    max_abs_diff_cuda_vs_cpu: np.ndarray | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cuda_speedup_over_cpu = cpu_runtime_ms / cuda_runtime_ms
    prefix = f"batch_{batch_size}"

    def write_metric(
        metric_name: str,
        data: np.ndarray,
        *,
        scientific: bool,
    ) -> None:
        _write_metric_outputs(
            data,
            nums_points,
            nums_distances,
            filename_stem=output_dir / f"{prefix}_{metric_name}",
            scientific=scientific,
        )

    if not skip_shapely:
        cuda_speedup_over_shapely = shapely_runtime_ms / cuda_runtime_ms
        cpu_speedup_over_shapely = shapely_runtime_ms / cpu_runtime_ms
        write_metric(
            "runtime_shapely",
            shapely_runtime_ms,
            scientific=True,
        )
    # CPU and CUDA outputs are always available; Shapely-related metrics are optional.
    write_metric("runtime_cuda", cuda_runtime_ms, scientific=True)
    write_metric(
        "runtime_cpu",
        cpu_runtime_ms,
        scientific=True,
    )
    if not skip_shapely:
        write_metric(
            "speedup_cuda_vs_shapely",
            cuda_speedup_over_shapely,
            scientific=False,
        )
        write_metric(
            "speedup_cpu_vs_shapely",
            cpu_speedup_over_shapely,
            scientific=False,
        )
    write_metric(
        "speedup_cuda_vs_cpu",
        cuda_speedup_over_cpu,
        scientific=False,
    )
    if assert_results:
        write_metric(
            "max_abs_diff_cuda_vs_cpu",
            max_abs_diff_cuda_vs_cpu,
            scientific=True,
        )
    if assert_results and not skip_shapely:
        write_metric("max_abs_diff", max_abs_diff_cuda, scientific=True)
        write_metric("max_abs_diff_cpu", max_abs_diff_cpu, scientific=True)
