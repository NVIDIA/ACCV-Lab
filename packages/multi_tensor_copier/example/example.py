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

import torch

import accvlab.multi_tensor_copier as mtc

NUM_CAMERAS = 6
# The config contains 4 samples. This means that the batch size is
# 4 * NUM_SAMPLE_CONFIG_REPEATS (= 16 samples for the default value).
NUM_SAMPLE_CONFIG_REPEATS = 4


def create_sample_meta_data(num_gt_objects: int, num_visible_objects_per_cam: list[int]) -> dict:
    """Create a single sample's meta-data dict, simulating a multi-camera 3D detection pipeline."""

    # @NOTE
    # Each sample contains two top-level entries:
    # - "cams_gt": a list with one dict per camera, holding variable-size 2D bounding boxes,
    #   class IDs, active flags, depths, and a projection matrix
    # - "gt_data": a dict with 3D ground truth bounding boxes, class IDs, and active flags
    # Several tensors have a variable first dimension (number of visible / GT objects),
    # which is the typical scenario that benefits from multi_tensor_copier: many small,
    # variable-size tensors that need to be transferred together.
    cams = []
    for cam_idx in range(NUM_CAMERAS):
        n_visible = num_visible_objects_per_cam[cam_idx]
        cam_data = {
            "bounding_boxes": torch.rand(n_visible, 4),
            "class_ids": torch.randint(0, 10, (n_visible,)),
            "active": torch.randint(0, 2, (n_visible,), dtype=torch.bool),
            "depths": torch.rand(
                n_visible,
            ),
            "proj_mat": torch.rand(3, 4),
        }
        cams.append(cam_data)

    gt_data = {
        "bounding_boxes_3d": torch.rand(num_gt_objects, 7),
        "class_ids": torch.randint(0, 10, (num_gt_objects,)),
        "active": torch.randint(0, 2, (num_gt_objects,), dtype=torch.bool),
    }

    sample_data = {"cams_gt": cams, "gt_data": gt_data}

    return sample_data


def create_batch_meta_data() -> list[dict]:
    """Assemble a batch of per-sample meta-data dicts."""

    # @NOTE
    # In a real training loop the meta-data originates from the DataLoader. Here we create
    # synthetic data where each sample has a different number of GT objects and a different
    # number of visible objects per camera, reflecting realistic variability.
    base_sample_configs = [
        {"num_gt": 60, "visible_per_cam": [30, 20, 40, 10, 50, 20]},
        {"num_gt": 100, "visible_per_cam": [60, 40, 30, 70, 20, 50]},
        {"num_gt": 45, "visible_per_cam": [10, 20, 10, 30, 20, 10]},
        {"num_gt": 120, "visible_per_cam": [80, 50, 60, 40, 70, 30]},
    ]
    sample_configs = base_sample_configs * NUM_SAMPLE_CONFIG_REPEATS

    batch = []
    for cfg in sample_configs:
        sample = create_sample_meta_data(cfg["num_gt"], cfg["visible_per_cam"])
        batch.append(sample)
    return batch


def dummy_compute():
    """Placeholder for useful work that can overlap with the CPU-to-GPU copy."""
    _ = torch.ones(256, 256) @ torch.ones(256, 256)


def dummy_process(gpu_meta_data: list[dict]):
    """Placeholder for a function that consumes the copied meta-data on the GPU."""
    for sample_idx, sample in enumerate(gpu_meta_data):
        num_cams = len(sample["cams_gt"])
        num_gt = sample["gt_data"]["bounding_boxes_3d"].shape[0]
        device = sample["gt_data"]["bounding_boxes_3d"].device
        print(f"  Sample {sample_idx}: {num_cams} cameras, {num_gt} GT objects  (device: {device})")


def main():

    # ----------------------- Create the batch meta-data -----------------------

    # @NOTE
    # Here, the per-sample meta-data tensors are not combined into per-batch meta-data tensors. This is due to
    # the fact that e.g. the number of visible objects per camera varies per sample, making combination &
    # handling of combined tensors cumbersome.
    batch_meta_data = create_batch_meta_data()

    # ------------------------ Start the asynchronous copy ------------------------

    # @NOTE
    # `start_copy()` traverses the nested structure of lists, tuples, and dicts, and
    # asynchronously copies all contained tensors to the target device. It returns a handle
    # before the copy is complete so that the calling thread can continue with other work while the transfer
    # is in progress.
    # Under the hood, the package applies several optimizations automatically (all enabled by
    # default): tensors are staged into pinned host memory for truly non-blocking H2D copies,
    # and small tensors are packed into (one or more) staging buffers to reduce per-tensor overhead. All of
    # this runs on a background thread, so this call returns before the copies complete.
    #
    # Note that this copy can be started anywhere the data is needed (i.e. not only when obtaining the data
    # from a DataLoader), so that it can be used with only local modifications to the training loop. For
    # example, if the meta-data on the GPU is only needed for loss computation, the copy can
    # be started inside the loss computation implementation (ideally with some work done in the meantime to
    # overlap with the asynchronous copy).
    handle = mtc.start_copy(batch_meta_data, "cuda:0")

    # @NOTE
    # IMPORTANT: Because the copy runs asynchronously, the input tensors must not be freed or
    # modified in-place until the copy has completed (i.e. until `handle.get()` returns or
    # `handle.ready()` returns `True`). See the `start_copy()` function documentation for details.

    # -------------------- Overlap with other work --------------------

    # @NOTE
    # Because `start_copy()` is asynchronous, we can overlap the CPU-to-GPU transfer with
    # other computation. Note that running asynchrounously with the copy is not the only (and not the most
    # important) optimization that is applied, so that this is beneficial but optional.
    dummy_compute()

    # -------------------- Retrieve and use the results --------------------

    # @NOTE
    # `handle.get()` blocks until the copy is complete and returns the same nested structure
    # with all tensors now residing on the target device. Non-tensor leaves (if any) are
    # passed through unchanged.
    gpu_meta_data = handle.get()

    # @NOTE
    # The copied data can now be consumed by downstream GPU operations (e.g. the detection
    # head, loss computation, etc.).
    #
    # Note on performance: For this simplified example, multi_tensor_copier achieves a
    # significant speedup over naive per-tensor .to() calls (see the evaluation script for
    # measurements). However, the absolute overhead of meta-data copying is moderate here.
    # In more complex real-world pipelines, the meta-data can be more extensive (e.g. additional
    # variable-length lane geometry where multiple lanes per sample cannot be combined into
    # single tensors due to variable size; or additional sensor modalities), which increases the number of
    # tensors and thus the per-tensor overhead. In such cases -- and with larger batch sizes -- the absolute
    # time savings grow accordingly.
    print("GPU meta-data ready:")
    dummy_process(gpu_meta_data)


if __name__ == "__main__":
    main()
