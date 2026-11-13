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

"""Generate the auto-labeling throughput figure from committed measurements."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def generate_docs_assets() -> None:
    docs_root = Path(__file__).resolve().parent
    input_file = docs_root.parents[2] / "applications/auto_labeling/evaluation_results/throughput_by_bs.csv"
    with input_file.open(encoding="utf-8", newline="") as handle:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(line for line in handle if line.strip() and not line.startswith("#"))
            if float(row["slot_utilization"]) >= 0.999
        ]
    if not rows:
        raise ValueError(f"No fully occupied batches in {input_file}")

    generated_dir = docs_root / "_generated"
    generated_dir.mkdir(exist_ok=True)
    header = "# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    (generated_dir / ".gitignore").write_text(header + "*\n", encoding="utf-8")
    output_file = generated_dir / "throughput_by_bs.svg"
    with plt.rc_context(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "accvlab-autolabel-throughput",
        }
    ):
        figure, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
        for variant, label, color, marker in [
            ("jpeg_parallel", "CPU JPEG (parallel)", "#52677f", "o"),
            ("nvdec_full", "ACCV-Lab", "#76b900", "s"),
        ]:
            median = [row[variant + "_median"] for row in rows]
            errors = [
                [row[variant + "_median"] - row[variant + "_minimum"] for row in rows],
                [row[variant + "_maximum"] - row[variant + "_median"] for row in rows],
            ]
            axis.errorbar(
                [row["batch_size"] for row in rows],
                median,
                yerr=errors,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=2,
                capsize=4,
                label=label,
            )
        best = max(rows, key=lambda row: row["nvdec_full_median"])
        batch_size = int(best["batch_size"])
        axis.axvline(batch_size, color="#76b900", linestyle="--", linewidth=1, alpha=0.7)
        axis.annotate(
            f"ACCV-Lab at BS={batch_size}\n{best['nvdec_full_median']:.2f} samples/s",
            (batch_size, best["nvdec_full_median"]),
            xytext=(-125, 18),
            textcoords="offset points",
            fontsize=10,
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks([row["batch_size"] for row in rows], [str(int(row["batch_size"])) for row in rows])
        axis.set_xlabel("Batch size (parallel scenes, log2 scale)")
        axis.set_ylabel("Useful samples / second")
        axis.set_title(
            "Auto-labeling throughput on one H100\nSix-camera HEVC input, 1600 x 900 pixels per camera",
            loc="left",
            fontweight="bold",
        )
        axis.set_ylim(bottom=0, top=max(row["nvdec_full_maximum"] for row in rows) * 1.25)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(loc="lower right")
        figure.text(
            0.015,
            -0.025,
            "Both pipelines use the same GPU model inference and GPU preprocessing.",
            fontsize=9,
        )
        figure.text(
            0.015,
            -0.06,
            "One sample = six camera views; HEVC/MP4 input at 10 fps with GOP 30.",
            fontsize=9,
        )
        figure.text(
            0.015,
            -0.095,
            "Median and min-max over 3 runs; same 1,024 samples. Only full batches are plotted.",
            fontsize=9,
        )
        figure.savefig(
            output_file,
            bbox_inches="tight",
            metadata={
                "Date": None,
                "Creator": "ACCV-Lab",
                "Title": "CPU JPEG versus ACCV-Lab throughput by batch size",
                "Description": (
                    "Six-camera 1600x900 HEVC/MP4 input, 10 fps, GOP 30; median and min-max over three runs on one H100."
                ),
            },
        )
        plt.close(figure)

    # Keep the generated asset's copyright notice without recording a build date.
    svg = output_file.read_text(encoding="utf-8")
    notice = "<!-- Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->"
    output_file.write_text(svg.replace("\n", "\n" + notice + "\n", 1), encoding="utf-8")


if __name__ == "__main__":
    generate_docs_assets()
