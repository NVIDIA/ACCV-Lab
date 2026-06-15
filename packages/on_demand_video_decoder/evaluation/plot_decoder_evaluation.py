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

"""Generate benchmark chart images for the on_demand_video_decoder evaluation docs."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

_GPU_COLORS = {
    "A100": "#4C72B0",
    "H200 NVL": "#55A868",
    "B200": "#C44E52",
    "B300": "#8172B2",
    "RTX PRO 6000": "#CCB974",
}

_DECODER_COLORS = {
    "accvlab_gpu": "#55A868",
    "pynvc_gpu": "#4C72B0",
    "decord_gpu": "#C44E52",
    "decord_cpu": "#8172B2",
    "opencv_cpu": "#CCB974",
}

_DECODER_LABELS = {
    "accvlab_gpu": "accvlab_gpu (ours)",
    "pynvc_gpu": "pynvc_gpu",
    "decord_gpu": "decord_gpu",
    "decord_cpu": "decord_cpu",
    "opencv_cpu": "opencv_cpu",
}

_NA_HATCH = "////"

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return (header_row, data_rows) from a CSV file."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows[0], rows[1:]


def _parse_float(value: str):
    """Return float or None for empty/missing cells."""
    value = value.strip()
    if value == "":
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str = "FPS (6-camera aggregate)") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Plot: cross-decoder grouped bar chart
# ---------------------------------------------------------------------------


def plot_cross_decoder(
    random_csv: Path,
    sequential_csv: Path,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    fig.suptitle(
        "HEVC GOP=30 — Cross-Decoder Comparison\n(6-camera aggregate FPS, higher is better)", fontsize=12
    )

    for ax, csv_path, access_label in zip(
        axes,
        [random_csv, sequential_csv],
        ["Random Access", "Sequential (Stream) Access"],
    ):
        header, rows = _read_csv(csv_path)
        # header: [gpu, decoder1, decoder2, ...]
        gpu_names = [r[0] for r in rows]
        decoders = header[1:]
        n_gpus = len(gpu_names)
        n_dec = len(decoders)

        x = np.arange(n_gpus)
        width = 0.8 / n_dec
        offsets = np.linspace(-(n_dec - 1) / 2, (n_dec - 1) / 2, n_dec) * width

        for i, decoder in enumerate(decoders):
            vals = [_parse_float(r[i + 1]) for r in rows]
            bar_x = x + offsets[i]
            color = _DECODER_COLORS.get(decoder, "#888888")
            label = _DECODER_LABELS.get(decoder, decoder)

            for j, v in enumerate(vals):
                if v is None:
                    # draw an N/A hatched bar using the max height for reference
                    ax.bar(
                        bar_x[j],
                        1,
                        width=width * 0.9,
                        color="white",
                        edgecolor=color,
                        linewidth=1,
                        hatch=_NA_HATCH,
                        alpha=0.6,
                        label="_nolegend_",
                    )
                else:
                    ax.bar(
                        bar_x[j],
                        v,
                        width=width * 0.9,
                        color=color,
                        edgecolor="white",
                        linewidth=0.5,
                        label=label if j == 0 else "_nolegend_",
                    )
                    ax.text(
                        bar_x[j],
                        v + max(vals[k] for k in range(n_gpus) if vals[k]) * 0.01,
                        str(int(v)),
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(gpu_names, fontsize=9)
        _apply_style(ax, access_label, "GPU")

        # legend only on first subplot
        if ax is axes[0]:
            handles = [
                mpatches.Patch(color=_DECODER_COLORS.get(d, "#888"), label=_DECODER_LABELS.get(d, d))
                for d in decoders
            ]
            na_patch = mpatches.Patch(
                facecolor="white", edgecolor="#888", hatch=_NA_HATCH, label="N/A (decoder failed)"
            )
            ax.legend(handles=handles + [na_patch], fontsize=8, loc="upper right")

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot: line chart for a continuous config sweep (GOP, B-frames)
# ---------------------------------------------------------------------------


def plot_config_line(
    random_csv: Path,
    sequential_csv: Path,
    output_path: Path,
    xlabel: str,
    title: str,
    xscale: str = "linear",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    fig.suptitle(title + "\n(6-camera aggregate FPS, higher is better)", fontsize=12)

    for ax, csv_path, access_label in zip(
        axes,
        [random_csv, sequential_csv],
        ["Random Access", "Sequential (Stream) Access"],
    ):
        header, rows = _read_csv(csv_path)
        x_vals = [_parse_float(r[0]) for r in rows]
        gpu_names = header[1:]

        for gpu in gpu_names:
            col_idx = header.index(gpu)
            y_vals = [_parse_float(r[col_idx]) for r in rows]
            color = _GPU_COLORS.get(gpu, "#888888")

            # split into segments at None gaps so line breaks at N/A
            xs, ys = [], []
            for xv, yv in zip(x_vals, y_vals):
                if yv is None:
                    if xs:
                        ax.plot(xs, ys, marker="o", color=color, linewidth=1.8, markersize=5)
                        xs, ys = [], []
                else:
                    xs.append(xv)
                    ys.append(yv)
            if xs:
                ax.plot(xs, ys, marker="o", color=color, linewidth=1.8, markersize=5, label=gpu)

        if xscale == "log":
            ax.set_xscale("log")
            ax.set_xticks(x_vals)
            ax.set_xticklabels([str(int(v)) for v in x_vals], fontsize=9)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.set_xticks(x_vals)
            ax.set_xticklabels([str(int(v)) for v in x_vals], fontsize=9)

        _apply_style(ax, access_label, xlabel)

        if ax is axes[0]:
            ax.legend(fontsize=8, loc="upper right")

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot: codec grouped bar chart
# ---------------------------------------------------------------------------


def plot_codec_bars(
    random_csv: Path,
    sequential_csv: Path,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    fig.suptitle(
        "Effect of Codec — GOP=30, B-frames=0\n(6-camera aggregate FPS, higher is better)", fontsize=12
    )

    for ax, csv_path, access_label in zip(
        axes,
        [random_csv, sequential_csv],
        ["Random Access", "Sequential (Stream) Access"],
    ):
        header, rows = _read_csv(csv_path)
        codec_names = [r[0] for r in rows]
        gpu_names = header[1:]
        n_codecs = len(codec_names)
        n_gpus = len(gpu_names)

        x = np.arange(n_codecs)
        width = 0.8 / n_gpus
        offsets = np.linspace(-(n_gpus - 1) / 2, (n_gpus - 1) / 2, n_gpus) * width

        for i, gpu in enumerate(gpu_names):
            col_idx = header.index(gpu)
            vals = [_parse_float(r[col_idx]) for r in rows]
            color = _GPU_COLORS.get(gpu, "#888888")
            bar_x = x + offsets[i]

            bars = ax.bar(
                bar_x,
                vals,
                width=width * 0.9,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=gpu,
            )
            for bar, v in zip(bars, vals):
                if v is not None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + max(v for v in vals if v) * 0.01,
                        str(int(v)),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(codec_names, fontsize=10)
        _apply_style(ax, access_label, "Codec")

        if ax is axes[0]:
            ax.legend(fontsize=8, loc="upper right")

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot: StreamPETR training horizontal bars
# ---------------------------------------------------------------------------


def plot_streampetr(
    setup_a_csv: Path,
    setup_b_csv: Path,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle("StreamPETR Training Performance — Iteration Time (lower is better)", fontsize=12)

    setup_configs = [
        (
            axes[0],
            setup_a_csv,
            "Hardware Setup A\n(8× RTX 6000D, 2× EPYC 7742)",
            ["Image", "Video: OpenCV", "Video: Ours"],
            ["image_ms", "video_opencv_ms", "video_ours_ms"],
            ["#4C72B0", "#CCB974", "#55A868"],
        ),
        (
            axes[1],
            setup_b_csv,
            "Hardware Setup B\n(8× H20, 2× Xeon Platinum 8468V)",
            ["Image", "Video: Ours"],
            ["image_ms", "video_ms"],
            ["#4C72B0", "#55A868"],
        ),
    ]

    for ax, csv_path, title, series_labels, col_names, colors in setup_configs:
        header, rows = _read_csv(csv_path)
        configs = [r[0] for r in rows]
        n_configs = len(configs)
        n_series = len(series_labels)

        y = np.arange(n_configs)
        height = 0.7 / n_series
        offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * height

        for i, (col, label, color) in enumerate(zip(col_names, series_labels, colors)):
            col_idx = header.index(col)
            vals = [_parse_float(r[col_idx]) for r in rows]
            bar_y = y + offsets[i]
            bars = ax.barh(bar_y, vals, height=height * 0.9, color=color, label=label, edgecolor="white")
            for bar, v in zip(bars, vals):
                if v is not None:
                    ax.text(
                        v + max(v for v in vals if v) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{int(v)} ms",
                        va="center",
                        fontsize=8,
                    )

        # annotate speedup for "Video: Ours" vs Image
        image_col = "image_ms"
        ours_col = col_names[-1]
        if image_col in header and ours_col in header:
            image_idx = header.index(image_col)
            ours_idx = header.index(ours_col)
            for j, row in enumerate(rows):
                img_v = _parse_float(row[image_idx])
                our_v = _parse_float(row[ours_idx])
                if img_v and our_v:
                    speedup = img_v / our_v
                    ax.text(
                        0.98,
                        y[j] + offsets[-1],
                        f"×{speedup:.2f} vs image",
                        transform=ax.get_yaxis_transform(),
                        ha="right",
                        va="center",
                        fontsize=7.5,
                        style="italic",
                        color="#333333",
                    )

        ax.set_yticks(y)
        ax.set_yticklabels(configs, fontsize=9)
        ax.set_xlabel("Iteration time [ms]", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="lower right")

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_all(input_root: Path, output_dir: Path) -> None:
    cross = input_root / "cross_decoder"
    sweep = input_root / "video_config_sweep"
    streampetr = input_root / "streampetr_training"

    plot_cross_decoder(
        random_csv=cross / "hevc_gop30_random_access.csv",
        sequential_csv=cross / "hevc_gop30_sequential.csv",
        output_path=output_dir / "cross_decoder.png",
    )

    plot_config_line(
        random_csv=sweep / "gop_random_access.csv",
        sequential_csv=sweep / "gop_sequential.csv",
        output_path=output_dir / "video_config_gop.png",
        xlabel="GOP size",
        title="Effect of GOP Size — HEVC, B-frames=0",
        xscale="log",
    )

    plot_config_line(
        random_csv=sweep / "bframes_random_access.csv",
        sequential_csv=sweep / "bframes_sequential.csv",
        output_path=output_dir / "video_config_bframes.png",
        xlabel="Number of B-frames",
        title="Effect of B-frames — HEVC, GOP=30",
    )

    plot_codec_bars(
        random_csv=sweep / "codec_random_access.csv",
        sequential_csv=sweep / "codec_sequential.csv",
        output_path=output_dir / "video_config_codec.png",
    )

    plot_streampetr(
        setup_a_csv=streampetr / "setup_a.csv",
        setup_b_csv=streampetr / "setup_b.csv",
        output_path=output_dir / "streampetr_training.png",
    )
