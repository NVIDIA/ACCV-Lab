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

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_DATA_FILE = Path("evaluation_results") / "simple_plot.csv"
_OUTPUT_FILE = "simple_plot.png"


def _read_plot_data(input_file: Path) -> tuple[list[float], list[float]]:
    if not input_file.exists():
        raise FileNotFoundError(f"Required example plot input data is missing: {input_file}")

    with input_file.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != ["x", "y"]:
            raise ValueError(f"Expected CSV columns 'x,y' in {input_file}")
        x_values: list[float] = []
        y_values: list[float] = []
        for row in reader:
            x_values.append(float(row["x"]))
            y_values.append(float(row["y"]))

    if not x_values:
        raise ValueError(f"Expected at least one data row in {input_file}")
    return x_values, y_values


def generate_docs_assets(context: Any) -> None:
    input_file = context.package_root / _DATA_FILE
    output_file = context.generated_dir / _OUTPUT_FILE
    x_values, y_values = _read_plot_data(input_file)

    figure, axis = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
    axis.plot(x_values, y_values, marker="o")
    axis.set_title("Generated Example Plot")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(True)
    figure.savefig(output_file)
    plt.close(figure)
