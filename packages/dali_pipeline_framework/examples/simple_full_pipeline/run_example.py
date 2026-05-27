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

import os
import sys

try:
    from accvlab.optim_test_tools import TensorDumper
except ImportError as exc:
    print(
        "ERROR: Could not import `accvlab.optim_test_tools`, which provides helper utilities used by this "
        "example (for example, dumping pipeline outputs for inspection). The `optim_test_tools` package is "
        "part of this ACCV-Lab repository, so this usually means not all packages of this project are "
        "installed. Please see `docs/guides/INSTALLATION_GUIDE.md` or the HTML documentation for installation "
        "details.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from simple_pipeline_setup import setup_simple_pipeline

_current_dir = os.path.dirname(os.path.abspath(__file__))

# @NOTE
# Set up `TensorDumper` before running the pipeline. The dumper is not required for training; it is used here
# so the tutorial output can be inspected visually and structurally after the example has run.
dump_dir = os.path.join(_current_dir, "dump_simple_pipeline")
dumper = TensorDumper()
dumper.enable(dump_dir)


def _print_batch(batch: dict, batch_idx: int):
    '''Print a human-readable summary of one batch (nested dict of tensors).'''
    print(f"\n===== Batch {batch_idx} =====")

    # @NOTE
    # The `DALIStructuredOutputIterator` reconstructs the nested sample structure from the flat DALI output.
    # The batch can therefore be accessed like the original `SampleDataGroup` hierarchy: cameras -> annotation
    # -> label/is_active, plus sample-level scene label fields.
    cameras = batch["cameras"]
    num_cameras = len(cameras)
    for cam_idx in range(num_cameras):
        cam = cameras[cam_idx]
        img = cam["image"]
        annotation = cam["annotation"]
        label = annotation["label"]
        is_active = annotation["is_active"]

        print(
            f"  Camera {cam_idx}:"
            f"  image shape={tuple(img.shape)}  dtype={img.dtype}"
            f"  |  label={label.flatten().tolist()}"
            f"  is_active={is_active.flatten().tolist()}"
        )

    scene_label = batch["scene_label"]
    scene_label_as_str = batch["scene_label_as_str"]
    print("  Scene label:" f"  mapped={scene_label.flatten().tolist()}" f"  original={scene_label_as_str}")


if __name__ == "__main__":
    batch_size = 2
    num_iterations = 4

    # ===== Set up the pipeline =====
    # @NOTE
    # `setup_simple_pipeline()` performs all framework-specific setup: provider creation, input callable
    # wrapping, processing-step composition, DALI pipeline build, and output iterator wrapping.
    print("Setting up the simple pipeline...")
    data_loader = setup_simple_pipeline(batch_size=batch_size)
    data_iter = iter(data_loader)

    # ===== Iterate over batches =====
    # @NOTE
    # Iterate like a regular PyTorch DataLoader. The example resets the iterator on `StopIteration` so it can
    # run for a fixed number of demonstration iterations even if the small synthetic dataset is exhausted.
    for i in range(num_iterations):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter.reset()
            batch = next(data_iter)

        _print_batch(batch, i)

        # ===== Dump using TensorDumper (for illustration purposes) =====

        # @NOTE
        # Dump the structured batch. Images are overridden to be written as RGB image files, while the scalar
        # labels and nested metadata are dumped as JSON-compatible values.
        dumper.add_tensor_data(
            "output",
            batch,
            TensorDumper.Type.JSON,
            dump_type_override={"image": TensorDumper.Type.IMAGE_RGB},
        )
        dumper.dump()

        print(f"\nDumped to: {dump_dir}")
