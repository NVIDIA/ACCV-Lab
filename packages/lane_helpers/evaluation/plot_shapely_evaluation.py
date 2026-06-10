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

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as colors
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
import numpy as np

DEFAULT_ANNOTATE_PLOTS = True
_LARGE_SPEEDUP_THRESHOLD = 1_000.0
_PLOT_FIGSIZE = (6.4, 5.2)
_COMPARISON_SUBPLOT_WIDTH = 6.6
_PLOT_SUPTITLE_FONT_SIZE = 22
_PLOT_TITLE_FONT_SIZE = 18
_PLOT_AXIS_LABEL_FONT_SIZE = 16
_PLOT_TICK_LABEL_FONT_SIZE = 14
_PLOT_COLORBAR_TICK_LABEL_FONT_SIZE = 14
_PLOT_ANNOTATION_FONT_SIZE = 16
_PLOT_ANNOTATION_MARKER_SIZE = 52
_PLOT_ANNOTATION_X_OFFSET = 0.25
_PLOT_COLORBAR_FRACTION = 0.046
_PLOT_COLORBAR_PAD = 0.02


@dataclass(frozen=True)
class _MetricPlotConfig:
    title: str
    annotation: str | None = None


_METRIC_PLOT_CONFIGS = {
    "runtime_shapely": _MetricPlotConfig("Shapely", annotation="runtime"),
    "runtime_cuda": _MetricPlotConfig("CUDA", annotation="runtime"),
    "runtime_cpu": _MetricPlotConfig("CPU", annotation="runtime"),
    "speedup_cuda_vs_shapely": _MetricPlotConfig("CUDA vs. Shapely", annotation="speedup"),
    "speedup_cpu_vs_shapely": _MetricPlotConfig("CPU vs. Shapely", annotation="speedup"),
    "speedup_cuda_vs_cpu": _MetricPlotConfig("CUDA vs. CPU", annotation="speedup"),
    "max_abs_diff_cuda_vs_cpu": _MetricPlotConfig("CUDA max abs. difference to CPU"),
    "max_abs_diff": _MetricPlotConfig("CUDA max abs. difference to Shapely"),
    "max_abs_diff_cpu": _MetricPlotConfig("CPU max abs. difference to Shapely"),
}
_SHAPELY_DEPENDENT_METRICS = frozenset(
    {
        "runtime_shapely",
        "speedup_cuda_vs_shapely",
        "speedup_cpu_vs_shapely",
        "max_abs_diff",
        "max_abs_diff_cpu",
    }
)
_RUNTIME_METRICS_WITH_SHAPELY = ("runtime_shapely", "runtime_cpu", "runtime_cuda")
_RUNTIME_METRICS_WITHOUT_SHAPELY = ("runtime_cpu", "runtime_cuda")
_SPEEDUP_METRICS_WITH_SHAPELY = (
    "speedup_cpu_vs_shapely",
    "speedup_cuda_vs_shapely",
    "speedup_cuda_vs_cpu",
)
_SPEEDUP_METRICS_WITHOUT_SHAPELY = ("speedup_cuda_vs_cpu",)


# Helper function for formatting speedup values in tables and annotations.
def _format_speedup_value(value: float) -> str:
    if abs(value) >= _LARGE_SPEEDUP_THRESHOLD:
        return f"{value:.2e}"
    return f"{value:.2f}"


# Helper function for splitting one Markdown table row into stripped cells.
def _split_markdown_table_row(row: str) -> list[str]:
    row = row.strip()
    if not row.startswith("|") or not row.endswith("|"):
        raise ValueError(f"Expected Markdown table row, got: {row}")
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    return cells


# Helper function for loading one metric table written by `_write_markdown`.
def _read_metric_table(filename: Path) -> tuple[list[int], list[int], np.ndarray]:
    table_rows = [
        line.strip()
        for line in filename.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("|")
    ]
    if len(table_rows) < 3:
        raise ValueError(f"Expected a Markdown header, separator, and at least one data row in {filename}")

    header_cells = _split_markdown_table_row(table_rows[0])
    if not header_cells or not header_cells[0].startswith("# Points"):
        raise ValueError(f"Expected first Markdown header cell to describe point counts in {filename}")
    nums_distances = [int(cell) for cell in header_cells[1:]]
    nums_points: list[int] = []
    values: list[list[float]] = []

    for row in table_rows[2:]:
        row_cells = _split_markdown_table_row(row)
        if len(row_cells) != len(nums_distances) + 1:
            raise ValueError(f"Expected {len(nums_distances) + 1} cells in {filename}, got {len(row_cells)}")
        nums_points.append(int(row_cells[0]))
        values.append([float(cell) for cell in row_cells[1:]])

    data = np.asarray(values, dtype=np.float64)
    return nums_points, nums_distances, data


# Helper function for choosing which speedup heatmap cells should show numeric labels.
def _selected_speedup_annotation_cells(
    data: np.ndarray,
    nums_points: list[int],
    nums_distances: list[int],
) -> list[tuple[int, int]]:
    def find_value_index(values: list[int], value: int) -> int | None:
        try:
            index = values.index(value)
        except ValueError:
            return None
        return index

    def add_unique_cell(cells: list[tuple[int, int]], cell: tuple[int, int]) -> None:
        if cell not in cells:
            cells.append(cell)

    def find_first_faster_distance_idx(points_idx: int) -> int | None:
        for distances_idx in range(len(nums_distances)):
            if np.isfinite(data[points_idx, distances_idx]) and data[points_idx, distances_idx] >= 1.0:
                return distances_idx
        return None

    def find_first_faster_points_idx(distances_idx: int) -> int | None:
        for points_idx in range(len(nums_points)):
            if np.isfinite(data[points_idx, distances_idx]) and data[points_idx, distances_idx] >= 1.0:
                return points_idx
        return None

    cells: list[tuple[int, int]] = []

    points_idx = find_value_index(nums_points, 2)
    distances_idx = find_value_index(nums_distances, 1)
    if points_idx is not None and distances_idx is not None:
        add_unique_cell(cells, (points_idx, distances_idx))

    finite_mask = np.isfinite(data)
    if np.any(finite_mask):
        finite_data = np.where(finite_mask, data, -np.inf)
        points_idx, distances_idx = np.unravel_index(np.argmax(finite_data), data.shape)
        add_unique_cell(cells, (int(points_idx), int(distances_idx)))

    finite_values = data[finite_mask]
    if finite_values.size > 0 and not np.all(finite_values > 1.0):
        distances_idx_by_value = {value: idx for idx, value in enumerate(nums_distances)}
        for points_idx, num_points_current in enumerate(nums_points):
            distances_idx = distances_idx_by_value.get(num_points_current)
            if distances_idx is None:
                continue
            if np.isfinite(data[points_idx, distances_idx]) and data[points_idx, distances_idx] > 1.0:
                add_unique_cell(cells, (points_idx, distances_idx))
                if points_idx > 0:
                    faster_distances_idx = find_first_faster_distance_idx(points_idx - 1)
                    if faster_distances_idx is not None:
                        add_unique_cell(cells, (points_idx - 1, faster_distances_idx))
                if distances_idx > 0:
                    faster_points_idx = find_first_faster_points_idx(distances_idx - 1)
                    if faster_points_idx is not None:
                        add_unique_cell(cells, (faster_points_idx, distances_idx - 1))
                break

    return cells


# Helper function for choosing which runtime heatmap cells should get marker labels.
def _selected_runtime_annotation_cells(
    nums_points: list[int],
    nums_distances: list[int],
) -> list[tuple[int, int]]:
    def find_value_index(values: list[int], value: int) -> int | None:
        try:
            index = values.index(value)
        except ValueError:
            return None
        return index

    def add_unique_cell(cells: list[tuple[int, int]], cell: tuple[int, int]) -> None:
        if cell not in cells:
            cells.append(cell)

    cells: list[tuple[int, int]] = []
    if nums_points and nums_distances:
        add_unique_cell(cells, (0, 0))
        add_unique_cell(cells, (len(nums_points) - 1, len(nums_distances) - 1))

    points_idx = find_value_index(nums_points, 100)
    distances_idx = find_value_index(nums_distances, 100)
    if points_idx is not None and distances_idx is not None:
        add_unique_cell(cells, (points_idx, distances_idx))

    return cells


# Helper function for placing numeric labels on selected speedup heatmap cells.
def _speedup_annotation_text_position(
    points_idx: int,
    distances_idx: int,
    nums_points: list[int],
    nums_distances: list[int],
    selected_cells: list[tuple[int, int]],
    data: np.ndarray,
    max_speedup_cell: tuple[int, int] | None,
) -> tuple[float, str]:
    if max_speedup_cell == (points_idx, distances_idx) and distances_idx > 0:
        return distances_idx - _PLOT_ANNOTATION_X_OFFSET, "right"

    is_left_of_value_diagonal = nums_distances[distances_idx] < nums_points[points_idx]
    has_adjacent_above_one_annotation = any(
        (other_points_idx, other_distances_idx) != (points_idx, distances_idx)
        and abs(other_points_idx - points_idx) + abs(other_distances_idx - distances_idx) == 1
        and np.isfinite(data[other_points_idx, other_distances_idx])
        and data[other_points_idx, other_distances_idx] >= 1.0
        for other_points_idx, other_distances_idx in selected_cells
    )
    should_place_left = distances_idx == len(nums_distances) - 1 or (
        distances_idx > 0 and is_left_of_value_diagonal and has_adjacent_above_one_annotation
    )
    if should_place_left:
        return distances_idx - _PLOT_ANNOTATION_X_OFFSET, "right"
    return distances_idx + _PLOT_ANNOTATION_X_OFFSET, "left"


# Helper function for drawing optional numeric labels on selected speedup heatmap cells.
def _annotate_speedup_heatmap(
    ax: Axes,
    data: np.ndarray,
    nums_points: list[int],
    nums_distances: list[int],
) -> None:
    selected_cells = _selected_speedup_annotation_cells(data, nums_points, nums_distances)
    finite_mask = np.isfinite(data)
    max_speedup_cell = None
    if np.any(finite_mask):
        finite_data = np.where(finite_mask, data, -np.inf)
        points_idx, distances_idx = np.unravel_index(np.argmax(finite_data), data.shape)
        max_speedup_cell = (int(points_idx), int(distances_idx))

    for points_idx, distances_idx in selected_cells:
        value = data[points_idx, distances_idx]
        if not np.isfinite(value):
            continue

        ax.scatter(
            [distances_idx],
            [points_idx],
            marker="o",
            s=_PLOT_ANNOTATION_MARKER_SIZE,
            c="black",
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

        text_x, horizontal_alignment = _speedup_annotation_text_position(
            points_idx,
            distances_idx,
            nums_points,
            nums_distances,
            selected_cells,
            data,
            max_speedup_cell,
        )
        ax.text(
            text_x,
            points_idx,
            _format_speedup_value(value),
            ha=horizontal_alignment,
            va="center",
            fontsize=_PLOT_ANNOTATION_FONT_SIZE,
            color="black",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            zorder=4,
        )


# Helper function for drawing fixed reference markers on runtime heatmap cells.
def _annotate_runtime_heatmap(
    ax: Axes,
    data: np.ndarray,
    nums_points: list[int],
    nums_distances: list[int],
) -> None:
    for points_idx, distances_idx in _selected_runtime_annotation_cells(nums_points, nums_distances):
        value = data[points_idx, distances_idx]
        if not np.isfinite(value):
            continue

        ax.scatter(
            [distances_idx],
            [points_idx],
            marker="o",
            s=_PLOT_ANNOTATION_MARKER_SIZE,
            c="black",
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

        if distances_idx == len(nums_distances) - 1:
            text_x = distances_idx - 0.15
            horizontal_alignment = "right"
        else:
            text_x = distances_idx + 0.15
            horizontal_alignment = "left"
        ax.text(
            text_x,
            points_idx,
            f"{value:.1e}",
            ha=horizontal_alignment,
            va="center",
            fontsize=_PLOT_ANNOTATION_FONT_SIZE,
            color="black",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            zorder=4,
        )


# Helper function for drawing one heatmap into an existing subplot.
def _draw_heatmap(
    ax: Axes,
    data: np.ndarray,
    nums_points: list[int],
    nums_distances: list[int],
    *,
    title: str,
    log_scale: bool,
    annotate_speedup: bool = False,
    annotate_runtime: bool = False,
) -> None:
    norm = None
    if log_scale:
        positive_values = data[data > 0]
        if positive_values.size > 0:
            norm = colors.LogNorm(vmin=positive_values.min(), vmax=positive_values.max())

    image = ax.imshow(data, norm=norm)
    ax.set_yticks(list(range(len(nums_points))), labels=nums_points, fontsize=_PLOT_TICK_LABEL_FONT_SIZE)
    ax.set_ylabel("Number of points", fontsize=_PLOT_AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(
        list(range(len(nums_distances))),
        labels=nums_distances,
        rotation=45,
        fontsize=_PLOT_TICK_LABEL_FONT_SIZE,
    )
    ax.set_xlabel("Number of distances", fontsize=_PLOT_AXIS_LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=_PLOT_TITLE_FONT_SIZE, pad=12)
    colorbar = ax.figure.colorbar(
        image,
        ax=ax,
        fraction=_PLOT_COLORBAR_FRACTION,
        pad=_PLOT_COLORBAR_PAD,
    )
    colorbar.ax.tick_params(labelsize=_PLOT_COLORBAR_TICK_LABEL_FONT_SIZE)
    colorbar.ax.yaxis.offsetText.set_fontsize(_PLOT_COLORBAR_TICK_LABEL_FONT_SIZE)
    if annotate_speedup:
        _annotate_speedup_heatmap(ax, data, nums_points, nums_distances)
    if annotate_runtime:
        _annotate_runtime_heatmap(ax, data, nums_points, nums_distances)


# Helper function for writing a multi-subplot comparison plot for one metric group.
def _plot_metric_comparison(
    metric_names: tuple[str, ...],
    metric_data: dict[str, np.ndarray],
    nums_points: list[int],
    nums_distances: list[int],
    *,
    batch_size: int,
    figure_title: str,
    filename: Path,
    annotate_plots: bool,
) -> None:
    available_metric_names = tuple(metric_name for metric_name in metric_names if metric_name in metric_data)
    if not available_metric_names:
        return

    subplot_width = _PLOT_FIGSIZE[0] if len(available_metric_names) == 1 else _COMPARISON_SUBPLOT_WIDTH
    fig, axes = plt.subplots(
        1,
        len(available_metric_names),
        figsize=(subplot_width * len(available_metric_names), _PLOT_FIGSIZE[1]),
        constrained_layout=True,
    )
    fig.suptitle(f"{figure_title} (Batch Size {batch_size})", fontsize=_PLOT_SUPTITLE_FONT_SIZE)
    axes = np.atleast_1d(axes).tolist()

    for ax, metric_name in zip(axes, available_metric_names):
        metric_config = _METRIC_PLOT_CONFIGS[metric_name]
        _draw_heatmap(
            ax,
            metric_data[metric_name],
            nums_points,
            nums_distances,
            title=metric_config.title,
            log_scale=True,
            annotate_speedup=metric_config.annotation == "speedup" and annotate_plots,
            annotate_runtime=metric_config.annotation == "runtime" and annotate_plots,
        )
    fig.savefig(filename)
    plt.close(fig)


# Helper function for writing comparison plots whose subplot counts depend on Shapely availability.
def _write_comparison_outputs(
    output_dir: Path,
    batch_size: int,
    nums_points: list[int],
    nums_distances: list[int],
    metric_data: dict[str, np.ndarray],
    *,
    has_shapely_results: bool,
    annotate_plots: bool,
) -> list[Path]:
    runtime_metric_names = (
        _RUNTIME_METRICS_WITH_SHAPELY if has_shapely_results else _RUNTIME_METRICS_WITHOUT_SHAPELY
    )
    speedup_metric_names = (
        _SPEEDUP_METRICS_WITH_SHAPELY if has_shapely_results else _SPEEDUP_METRICS_WITHOUT_SHAPELY
    )
    prefix = f"batch_{batch_size}"
    comparison_files = [
        output_dir / f"{prefix}_runtime_comparison.png",
        output_dir / f"{prefix}_speedup_comparison.png",
    ]
    _plot_metric_comparison(
        runtime_metric_names,
        metric_data,
        nums_points,
        nums_distances,
        batch_size=batch_size,
        figure_title="Runtime [ms]",
        filename=comparison_files[0],
        annotate_plots=annotate_plots,
    )
    _plot_metric_comparison(
        speedup_metric_names,
        metric_data,
        nums_points,
        nums_distances,
        batch_size=batch_size,
        figure_title="Speedup [x-fold]",
        filename=comparison_files[1],
        annotate_plots=annotate_plots,
    )
    return comparison_files


# Helper function for parsing comma-separated integer lists.
def _parse_int_list(value: str) -> list[int]:
    parsed_values = [int(item) for item in value.split(",") if item]
    return parsed_values


def plot_batch_results_from_markdown(
    input_dir: Path,
    output_dir: Path,
    batch_size: int,
    annotate_plots: bool,
) -> list[Path]:
    prefix = f"batch_{batch_size}_"
    markdown_files = sorted(input_dir.glob(f"{prefix}*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"No Markdown result tables found for batch={batch_size} in {input_dir}")
    available_metric_names = {markdown_file.stem[len(prefix) :] for markdown_file in markdown_files}
    has_shapely_results = "runtime_shapely" in available_metric_names

    metric_data: dict[str, np.ndarray] = {}
    comparison_nums_points: list[int] | None = None
    comparison_nums_distances: list[int] | None = None
    for markdown_file in markdown_files:
        metric_name = markdown_file.stem[len(prefix) :]
        if metric_name not in _METRIC_PLOT_CONFIGS:
            continue
        if metric_name in _SHAPELY_DEPENDENT_METRICS and not has_shapely_results:
            continue

        nums_points, nums_distances, data = _read_metric_table(markdown_file)
        metric_data[metric_name] = data
        comparison_nums_points = nums_points
        comparison_nums_distances = nums_distances

    if comparison_nums_points is not None and comparison_nums_distances is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plotted_files = _write_comparison_outputs(
            output_dir,
            batch_size,
            comparison_nums_points,
            comparison_nums_distances,
            metric_data,
            has_shapely_results=has_shapely_results,
            annotate_plots=annotate_plots,
        )
    else:
        plotted_files = []

    if not plotted_files:
        raise FileNotFoundError(
            f"No known Markdown result tables found for batch={batch_size} in {input_dir}"
        )
    return plotted_files


def plot_from_markdown_directory(
    *,
    input_dir: Path,
    output_dir: Path,
    batch_sizes: list[int],
    annotate_plots: bool = DEFAULT_ANNOTATE_PLOTS,
) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Markdown input directory does not exist: {input_dir}")

    plotted_files: list[Path] = []
    for batch_size in batch_sizes:
        batch_plotted_files = plot_batch_results_from_markdown(
            input_dir=input_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            annotate_plots=annotate_plots,
        )
        plotted_files.extend(batch_plotted_files)
    return plotted_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate polyline runtime plot images from Markdown result tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing Markdown result tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where plot images should be written.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,4,16,64",
        help="Comma-separated batch sizes to plot.",
    )
    no_annotate_plots_action = parser.add_argument(
        "--no-annotate-plots",
        dest="annotate_plots",
        action="store_false",
        help="Disable annotations in generated heatmaps.",
    )
    parser.set_defaults(annotate_plots=DEFAULT_ANNOTATE_PLOTS)
    no_annotate_plots_action.default = argparse.SUPPRESS
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    batch_sizes = _parse_int_list(args.batch_sizes)
    plotted_files = plot_from_markdown_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_sizes=batch_sizes,
        annotate_plots=args.annotate_plots,
    )
    for plotted_file in plotted_files:
        print(f"Generated plot: {plotted_file}")


if __name__ == "__main__":
    main()
