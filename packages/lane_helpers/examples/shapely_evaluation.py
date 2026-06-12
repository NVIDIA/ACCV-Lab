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
from collections.abc import Callable, Iterator
import gc
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
from shapely import get_coordinates, line_interpolate_point, linestrings
import torch

# Import helpers for outputting results and plots
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import _shapely_evaluation_outputs as shapely_evaluation_outputs

# ==================== Default configuration for the evaluation ====================

# These constants are convenient local configuration knobs. However, these configurations can also
# be done with CLI arguments.
# When changing these constants, check the CLI arguments further below in the script,
# because some flags only override the default in one direction.

# Sweep values for the heatmap axes and the batch-size examples.
DEFAULT_NUMS_POINTS = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
DEFAULT_NUMS_DISTANCES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
DEFAULT_BATCH_SIZES = [1, 4, 16, 64]
# Keep the measured work roughly constant across batch sizes.
DEFAULT_NUM_POLYLINES_PER_MEASUREMENT = 64 * 10
# Warm up a representative mid-sized configuration before timing the sweep.
DEFAULT_NUM_WARMUP_RUNS = 3
DEFAULT_WARMUP_NUM_POINTS = 100
DEFAULT_WARMUP_NUM_DISTANCES = 100
# Shapely can be skipped for faster CPU/CUDA-only benchmark sweeps.
DEFAULT_SKIP_SHAPELY = False
# Result checks are optional because they add Shapely reference work to each configuration.
DEFAULT_ASSERT_RESULTS = False
DEFAULT_ASSERT_ATOL = 1e-3
DEFAULT_ASSERT_RTOL = 0.0
# Plot annotations call out representative cells in generated heatmaps.
DEFAULT_ANNOTATE_PLOTS = True
# Plotting mode regenerates images from existing Markdown tables without benchmarking.
DEFAULT_PLOTTING_MODE = False
# ================== End: Default configuration for the evaluation =================


# ================== Constants for the evaluation ==================
DEVICE = "cuda"
DTYPE_NP = np.float32
DTYPE_TORCH = torch.float32
_POLYLINE_MODULE: ModuleType | None = None
# ================ End: Constants for the evaluation ===============


# Helper function for lazily importing the compiled polyline module outside plotting-only mode.
def _get_polyline_module() -> ModuleType:
    global _POLYLINE_MODULE
    if _POLYLINE_MODULE is None:
        from accvlab.lane_helpers import polyline as polyline_module

        _POLYLINE_MODULE = polyline_module
    return _POLYLINE_MODULE


# Helper function for config parsing
def _parse_int_list(value: str) -> list[int]:
    parsed_values = [int(item) for item in value.split(",") if item]
    return parsed_values


# Helper function for computing the batched Shapely reference.
def _compute_batched_shapely_reference(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    line_strings = linestrings(points)
    interpolated_points = line_interpolate_point(line_strings[:, None], distances)
    batched_reference = (
        get_coordinates(interpolated_points)
        .reshape(
            points.shape[0],
            distances.shape[1],
            points.shape[2],
        )
        .astype(DTYPE_NP)
    )
    return batched_reference


# Helper function for computing per-polyline lengths in NumPy.
def _compute_polyline_lengths_np(points: np.ndarray) -> np.ndarray:
    if points.shape[1] <= 1:
        lengths = np.zeros((points.shape[0],), dtype=DTYPE_NP)
        return lengths
    lengths = np.linalg.norm(points[:, 1:] - points[:, :-1], axis=2).sum(axis=1).astype(DTYPE_NP)
    return lengths


# Helper function for comparing CPU and CUDA outputs against Shapely when requested.
def _assert_matches_shapely(
    shapely_result: np.ndarray,
    result: np.ndarray,
    *,
    implementation_name: str,
    batch_size: int,
    num_points: int,
    num_distances: int,
    atol: float,
    rtol: float,
) -> None:
    try:
        np.testing.assert_allclose(result, shapely_result, atol=atol, rtol=rtol)
    except AssertionError as exc:
        max_abs_diff = np.abs(shapely_result - result).max()
        raise AssertionError(
            f"{implementation_name} result differs from Shapely for "
            f"batch={batch_size}, points={num_points}, distances={num_distances}; "
            f"max_abs_diff={max_abs_diff}, atol={atol}, rtol={rtol}"
        ) from exc


# Helper function for constructing one deterministic benchmark input configuration.
def _make_evaluation_case(
    batch_size: int,
    num_points: int,
    num_distances: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed=seed)
    # Set up the polylines
    points = generator.uniform(0.0, 1.0, size=(batch_size, num_points, 2)).astype(DTYPE_NP)
    lengths = _compute_polyline_lengths_np(points)
    # Set up the distances to sample the polyline at
    distances_normalized = generator.uniform(0.0, 1.0, size=(batch_size, num_distances)).astype(DTYPE_NP)
    distances = distances_normalized * lengths[:, None]
    return points, distances


# Helper function for iterating over deterministic benchmark configurations.
def _iter_evaluation_cases(
    batch_size: int,
    nums_points: list[int],
    nums_distances: list[int],
) -> Iterator[tuple[int, int, int, int, int]]:
    for points_idx, num_points_current in enumerate(nums_points):
        for distances_idx, num_distances_current in enumerate(nums_distances):
            seed = batch_size * 1_000_000 + num_points_current * 1_000 + num_distances_current
            yield points_idx, distances_idx, num_points_current, num_distances_current, seed


# Helper function for placing the same NumPy inputs on CUDA and CPU.
def _make_torch_tensors(
    *arrays: np.ndarray,
) -> tuple[torch.Tensor, ...]:
    tensors_gpu = [torch.tensor(array, device=DEVICE, dtype=DTYPE_TORCH) for array in arrays]
    tensors_cpu = [torch.tensor(array, device="cpu", dtype=DTYPE_TORCH) for array in arrays]
    return *tensors_gpu, *tensors_cpu


# Helper function for placing NumPy inputs on one target device.
def _make_torch_tensors_on_device(
    *arrays: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, ...]:
    tensors = tuple(torch.tensor(array, device=device, dtype=DTYPE_TORCH) for array in arrays)
    return tensors


# Helper function for timing repeated calls and synchronizing CUDA work when needed.
def _time_call(
    function: Callable[[], object],
    *,
    num_runs: int,
    synchronize_cuda: bool = False,
) -> float:
    if synchronize_cuda:
        # Ensure previous work is finished before starting the timing.
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        function()
    if synchronize_cuda:
        # Ensure all work is finished before stopping the timing.
        torch.cuda.synchronize()
    runtime = (time.perf_counter() - start) / num_runs
    return runtime


# Helper function for reducing cross-implementation timing interference.
def _cleanup_between_implementation_sweeps() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# Helper function for timing the Shapely reference implementation.
def _time_shapely(
    points: np.ndarray,
    distances: np.ndarray,
    *,
    num_runs: int,
) -> float:
    compute_function = lambda: _compute_batched_shapely_reference(points, distances)
    runtime = _time_call(
        compute_function,
        num_runs=num_runs,
    )
    return runtime


# Helper function for timing the CUDA implementation.
def _time_cuda(
    points: torch.Tensor,
    distances: torch.Tensor,
    *,
    num_runs: int,
) -> float:
    polyline_module = _get_polyline_module()
    compute_function = lambda: polyline_module.interpolate(points, distances)
    runtime = _time_call(
        compute_function,
        num_runs=num_runs,
        synchronize_cuda=True,
    )
    return runtime


# Helper function for timing the CPU implementation.
def _time_cpu(
    points: torch.Tensor,
    distances: torch.Tensor,
    *,
    num_runs: int,
) -> float:
    polyline_module = _get_polyline_module()
    compute_function = lambda: polyline_module.interpolate(points, distances)
    runtime = _time_call(compute_function, num_runs=num_runs)
    return runtime


# Helper function for warming up all selected implementations once before measured runs.
def _run_warmup(
    *,
    batch_size: int,
    num_points: int,
    num_distances: int,
    num_warmup_runs: int,
    skip_shapely: bool,
) -> None:
    if num_warmup_runs <= 0:
        return

    points_np, distances_np = _make_evaluation_case(
        batch_size,
        num_points,
        num_distances,
        seed=0,
    )
    points_gpu, distances_gpu, points_cpu, distances_cpu = _make_torch_tensors(points_np, distances_np)
    polyline_module = _get_polyline_module()

    for _ in range(num_warmup_runs):
        if not skip_shapely:
            _compute_batched_shapely_reference(points_np, distances_np)
        polyline_module.interpolate(points_cpu, distances_cpu)
        polyline_module.interpolate(points_gpu, distances_gpu)

    torch.cuda.synchronize()


# Helper to (optionally) validate the results against the Shapely reference.
def _run_validation_sweep(
    batch_size: int,
    nums_points: list[int],
    nums_distances: list[int],
    *,
    assert_atol: float,
    assert_rtol: float,
    max_abs_diff_cpu: np.ndarray,
    max_abs_diff_cuda: np.ndarray,
    max_abs_diff_cuda_vs_cpu: np.ndarray,
) -> None:
    print(f"Running validation sweep for batch={batch_size}")
    polyline_module = _get_polyline_module()
    for points_idx, distances_idx, num_points_current, num_distances_current, seed in _iter_evaluation_cases(
        batch_size, nums_points, nums_distances
    ):
        print(
            "Running validation "
            f"batch={batch_size}, points={num_points_current}, distances={num_distances_current}"
        )
        points_np, distances_np = _make_evaluation_case(
            batch_size,
            num_points_current,
            num_distances_current,
            seed=seed,
        )
        shapely_result = _compute_batched_shapely_reference(points_np, distances_np)
        points_gpu, distances_gpu, points_cpu, distances_cpu = _make_torch_tensors(points_np, distances_np)
        cpu_result = polyline_module.interpolate(points_cpu, distances_cpu).numpy()
        cuda_result = polyline_module.interpolate(points_gpu, distances_gpu).cpu().numpy()

        max_abs_diff_cpu[points_idx, distances_idx] = np.abs(shapely_result - cpu_result).max()
        max_abs_diff_cuda[points_idx, distances_idx] = np.abs(shapely_result - cuda_result).max()
        max_abs_diff_cuda_vs_cpu[points_idx, distances_idx] = np.abs(cpu_result - cuda_result).max()

        _assert_matches_shapely(
            shapely_result,
            cpu_result,
            implementation_name="CPU",
            batch_size=batch_size,
            num_points=num_points_current,
            num_distances=num_distances_current,
            atol=assert_atol,
            rtol=assert_rtol,
        )
        _assert_matches_shapely(
            shapely_result,
            cuda_result,
            implementation_name="CUDA",
            batch_size=batch_size,
            num_points=num_points_current,
            num_distances=num_distances_current,
            atol=assert_atol,
            rtol=assert_rtol,
        )


# Helper function for evaluating every point-count and distance-count pair for one batch size.
def _evaluate_batch_size(
    batch_size: int,
    nums_points: list[int],
    nums_distances: list[int],
    *,
    num_runs: int,
    assert_results: bool,
    assert_atol: float,
    assert_rtol: float,
    skip_shapely: bool,
) -> tuple[
    np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None
]:
    result_shape = (len(nums_points), len(nums_distances))

    shapely_runtime_ms = None if skip_shapely else np.zeros(result_shape, dtype=np.float64)
    cuda_runtime_ms = np.zeros(result_shape, dtype=np.float64)
    cpu_runtime_ms = np.zeros(result_shape, dtype=np.float64)

    max_abs_diff_cuda = np.zeros_like(cpu_runtime_ms) if assert_results else None
    max_abs_diff_cpu = np.zeros_like(cpu_runtime_ms) if assert_results else None
    max_abs_diff_cuda_vs_cpu = np.zeros_like(cpu_runtime_ms) if assert_results else None

    if not skip_shapely:
        print(f"Running Shapely sweep for batch={batch_size}, runs={num_runs}")
        for (
            points_idx,
            distances_idx,
            num_points_current,
            num_distances_current,
            seed,
        ) in _iter_evaluation_cases(batch_size, nums_points, nums_distances):
            print(
                "Running Shapely evaluation "
                f"batch={batch_size}, points={num_points_current}, distances={num_distances_current}, "
                f"runs={num_runs}"
            )
            points_np, distances_np = _make_evaluation_case(
                batch_size,
                num_points_current,
                num_distances_current,
                seed=seed,
            )

            shapely_runtime = _time_shapely(
                points_np,
                distances_np,
                num_runs=num_runs,
            )
            shapely_runtime_ms[points_idx, distances_idx] = shapely_runtime * 1000
        _cleanup_between_implementation_sweeps()

    print(f"Running CPU sweep for batch={batch_size}, runs={num_runs}")
    for points_idx, distances_idx, num_points_current, num_distances_current, seed in _iter_evaluation_cases(
        batch_size, nums_points, nums_distances
    ):
        print(
            "Running CPU evaluation "
            f"batch={batch_size}, points={num_points_current}, distances={num_distances_current}, "
            f"runs={num_runs}"
        )
        points_np, distances_np = _make_evaluation_case(
            batch_size,
            num_points_current,
            num_distances_current,
            seed=seed,
        )
        points_cpu, distances_cpu = _make_torch_tensors_on_device(
            points_np,
            distances_np,
            device="cpu",
        )

        cpu_runtime_ms[points_idx, distances_idx] = (
            _time_cpu(
                points_cpu,
                distances_cpu,
                num_runs=num_runs,
            )
            * 1000
        )
    _cleanup_between_implementation_sweeps()

    print(f"Running CUDA sweep for batch={batch_size}, runs={num_runs}")
    for points_idx, distances_idx, num_points_current, num_distances_current, seed in _iter_evaluation_cases(
        batch_size, nums_points, nums_distances
    ):
        print(
            "Running CUDA evaluation "
            f"batch={batch_size}, points={num_points_current}, distances={num_distances_current}, "
            f"runs={num_runs}"
        )
        points_np, distances_np = _make_evaluation_case(
            batch_size,
            num_points_current,
            num_distances_current,
            seed=seed,
        )
        points_gpu, distances_gpu = _make_torch_tensors_on_device(
            points_np,
            distances_np,
            device=DEVICE,
        )

        cuda_runtime_ms[points_idx, distances_idx] = (
            _time_cuda(
                points_gpu,
                distances_gpu,
                num_runs=num_runs,
            )
            * 1000
        )
    _cleanup_between_implementation_sweeps()

    if assert_results:
        _run_validation_sweep(
            batch_size,
            nums_points,
            nums_distances,
            assert_atol=assert_atol,
            assert_rtol=assert_rtol,
            max_abs_diff_cpu=max_abs_diff_cpu,
            max_abs_diff_cuda=max_abs_diff_cuda,
            max_abs_diff_cuda_vs_cpu=max_abs_diff_cuda_vs_cpu,
        )
        _cleanup_between_implementation_sweeps()

    return (
        shapely_runtime_ms,
        cpu_runtime_ms,
        cuda_runtime_ms,
        max_abs_diff_cpu,
        max_abs_diff_cuda,
        max_abs_diff_cuda_vs_cpu,
    )


# Helper function for parsing command-line arguments.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate batched CPU/CUDA polyline interpolation against a Shapely LineString reference "
            "over point-count, distance-count, and batch-size sweeps."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-points",
        dest="nums_points",
        default=",".join(str(item) for item in DEFAULT_NUMS_POINTS),
        help="Comma-separated point counts for the polyline-length sweep.",
    )
    parser.add_argument(
        "--num-distances",
        dest="nums_distances",
        default=",".join(str(item) for item in DEFAULT_NUMS_DISTANCES),
        help="Comma-separated sample-distance counts for the interpolation sweep.",
    )
    parser.add_argument(
        "--batch-sizes",
        default=",".join(str(item) for item in DEFAULT_BATCH_SIZES),
        help="Comma-separated batch sizes to evaluate or replot.",
    )
    parser.add_argument(
        "--num-polylines-per-measurement",
        type=int,
        default=DEFAULT_NUM_POLYLINES_PER_MEASUREMENT,
        help="Target number of polylines measured per configuration; divided by batch size to get runs.",
    )
    parser.add_argument(
        "--num-warmup-runs",
        type=int,
        default=DEFAULT_NUM_WARMUP_RUNS,
        help="Number of untimed warmup runs before the measured sweep.",
    )
    parser.add_argument(
        "--warmup-num-points",
        type=int,
        default=DEFAULT_WARMUP_NUM_POINTS,
        help="Point count used for warmup inputs.",
    )
    parser.add_argument(
        "--warmup-num-distances",
        type=int,
        default=DEFAULT_WARMUP_NUM_DISTANCES,
        help="Sample-distance count used for warmup inputs.",
    )
    parser.add_argument(
        "--skip-shapely",
        action="store_true",
        default=DEFAULT_SKIP_SHAPELY,
        help="Skip Shapely reference timing and Shapely-based speedup plots.",
    )
    parser.add_argument(
        "--assert-results",
        action="store_true",
        default=DEFAULT_ASSERT_RESULTS,
        help="Compare CPU and CUDA outputs against Shapely using the configured tolerances.",
    )
    parser.add_argument(
        "--assert-atol",
        type=float,
        default=DEFAULT_ASSERT_ATOL,
        help="Absolute tolerance used when asserting results against Shapely.",
    )
    parser.add_argument(
        "--assert-rtol",
        type=float,
        default=DEFAULT_ASSERT_RTOL,
        help="Relative tolerance used when asserting results against Shapely.",
    )
    parser.add_argument(
        "--plotting",
        action="store_true",
        default=DEFAULT_PLOTTING_MODE,
        help="Only regenerate plots from existing Markdown result tables.",
    )
    no_annotate_plots_action = parser.add_argument(
        "--no-annotate-plots",
        dest="annotate_plots",
        action="store_false",
        help="Disable annotations in generated heatmaps.",
    )
    parser.set_defaults(annotate_plots=DEFAULT_ANNOTATE_PLOTS)
    no_annotate_plots_action.default = argparse.SUPPRESS
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("polyline_shapely_eval_results"),
        help="Directory for Markdown result tables and generated plot images.",
    )
    args = parser.parse_args()
    return args


# Helper function for recreating plots from existing Markdown result tables.
def _run_plotting_mode(
    *,
    output_dir: Path,
    batch_sizes: list[int],
    annotate_plots: bool,
) -> None:
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    print("Regenerating plots from existing Markdown result tables...")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Annotate plots: {annotate_plots}")
    print(f"Output directory: {output_dir}")

    for batch_size in batch_sizes:
        plotted_files = shapely_evaluation_outputs.plot_batch_results_from_markdown(
            output_dir,
            batch_size,
            annotate_plots,
        )
        print(f"Regenerated {len(plotted_files)} plot(s) for batch={batch_size}")


# Main entry point for the full benchmark sweep.
def main() -> None:
    args = _parse_args()
    nums_points = _parse_int_list(args.nums_points)
    nums_distances = _parse_int_list(args.nums_distances)
    batch_sizes = _parse_int_list(args.batch_sizes)
    # Make relative output paths independent of the caller's working directory.
    if not args.output_dir.is_absolute():
        args.output_dir = SCRIPT_DIR / args.output_dir

    if args.plotting:
        _run_plotting_mode(
            output_dir=args.output_dir,
            batch_sizes=batch_sizes,
            annotate_plots=args.annotate_plots,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires a CUDA-capable PyTorch installation.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Result assertions require Shapely, so disabling Shapely also disables assertions.
    assert_results = args.assert_results and not args.skip_shapely

    print("Performing runtime evaluation...")
    print(f"Numbers of points: {nums_points}")
    print(f"Numbers of distances: {nums_distances}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Measured polylines per configuration: {args.num_polylines_per_measurement}")
    print(
        "Warmup configuration: "
        f"batch={max(batch_sizes)}, points={args.warmup_num_points}, "
        f"distances={args.warmup_num_distances}, runs={args.num_warmup_runs}"
    )
    print(f"Use Shapely reference: {not args.skip_shapely}")
    print(f"Assert results against Shapely: {assert_results}")
    print(f"Annotate plots: {args.annotate_plots}")
    print(f"Output directory: {args.output_dir}")

    _run_warmup(
        batch_size=max(batch_sizes),
        num_points=args.warmup_num_points,
        num_distances=args.warmup_num_distances,
        num_warmup_runs=args.num_warmup_runs,
        skip_shapely=args.skip_shapely,
    )

    for batch_size in batch_sizes:
        # Keep (roughly) the same number of measured polylines per configuration across batch sizes.
        num_runs = max(1, args.num_polylines_per_measurement // batch_size)
        print(f"Using {num_runs} measured runs for batch={batch_size}")

        # Run evaluation & get results for one batch size (number of polylines in single call).
        (
            shapely_runtime_ms,
            cpu_runtime_ms,
            cuda_runtime_ms,
            max_abs_diff_cpu,
            max_abs_diff_cuda,
            max_abs_diff_cuda_vs_cpu,
        ) = _evaluate_batch_size(
            batch_size,
            nums_points,
            nums_distances,
            num_runs=num_runs,
            assert_results=assert_results,
            assert_atol=args.assert_atol,
            assert_rtol=args.assert_rtol,
            skip_shapely=args.skip_shapely,
        )

        # Write results to disk.
        shapely_evaluation_outputs.write_batch_results(
            args.output_dir,
            batch_size,
            nums_points,
            nums_distances,
            shapely_runtime_ms,
            cpu_runtime_ms,
            cuda_runtime_ms,
            args.skip_shapely,
            assert_results,
            max_abs_diff_cpu,
            max_abs_diff_cuda,
            max_abs_diff_cuda_vs_cpu,
            args.annotate_plots,
        )

        # Print info.
        cuda_speedup_over_cpu = cpu_runtime_ms / cuda_runtime_ms
        if not args.skip_shapely:
            cuda_speedup_over_shapely = shapely_runtime_ms / cuda_runtime_ms
            cpu_speedup_over_shapely = shapely_runtime_ms / cpu_runtime_ms
            print(f"Average Shapely runtime [ms], batch={batch_size}:\n{shapely_runtime_ms}")
        print(f"Average CPU runtime [ms], batch={batch_size}:\n{cpu_runtime_ms}")
        print(f"Average CUDA runtime [ms], batch={batch_size}:\n{cuda_runtime_ms}")
        if not args.skip_shapely:
            print(f"CPU speedup over Shapely, batch={batch_size}:\n{cpu_speedup_over_shapely}")
            print(f"CUDA speedup over Shapely, batch={batch_size}:\n{cuda_speedup_over_shapely}")
        print(f"CUDA speedup over CPU, batch={batch_size}:\n{cuda_speedup_over_cpu}")
        if assert_results:
            print(f"CUDA max absolute difference to CPU, batch={batch_size}:\n{max_abs_diff_cuda_vs_cpu}")
            print(f"CPU max absolute difference to Shapely, batch={batch_size}:\n{max_abs_diff_cpu}")
            print(f"CUDA max absolute difference to Shapely, batch={batch_size}:\n{max_abs_diff_cuda}")


if __name__ == "__main__":
    main()
