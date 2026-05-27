# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import sys

# Ensure content of 'dali_pipeline_example_extensions' is importable
_current_dir = os.path.dirname(os.path.abspath(__file__))
_dali_extensions_root = os.path.abspath(os.path.join(_current_dir, "..", "dali_pipeline_example_extensions"))
if _dali_extensions_root not in sys.path:
    sys.path.insert(0, _dali_extensions_root)


try:
    from accvlab.optim_test_tools import Stopwatch, NVTXRangeWrapper, TensorDumper
except ImportError as exc:
    print(
        "ERROR: Could not import `accvlab.optim_test_tools`, which provides helper utilities used by this "
        "example (for example, dumping pipeline outputs for inspection and profiling the example loop). "
        "The `optim_test_tools` package is part of this ACCV-Lab repository, so this usually means not all "
        "packages of this project are installed. Please see `docs/guides/INSTALLATION_GUIDE.md` or the HTML "
        "documentation for installation details.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from pipeline_setup import *

import input_definitions as defs

# Note that we do not activate the stopwatch here (and therefore also do not need
# to obtain the singleton object here). In this example, we provide command-line
# arguments to control the number of warmup iterations.
# We will activate the stopwatch in main block below using these arguments.

nvtx_wrp = NVTXRangeWrapper()
nvtx_wrp.enable(sync_on_push=True, sync_on_pop=True, keep_track_of_range_order=False)

# We do activate the dumper here, but it is used only at the end of the script
# and not inside the main loop. It is used to show the data of the last
# performed iteration.
dumper = TensorDumper()
dumper.enable(os.path.join(_current_dir, "dump_obj_det_2d_pipeline"))


# Draw bounding boxes for both the image and the heatmap.
# For the heatmap, the bounding boxes also contain the centers.
# Note that while this is a relatively complex function, it is only
# executed if the dumper is enabled (see `dumper.run_if_enabled()`-call in the main
# block).
# This is an example of how to use the dumper to do complex visualizations
# without introducing overhead outside of debugging (as the dumper is enabled
# for debugging only).
def draw_bboxes(data, use_single_images):
    import torch
    import numpy as np
    import cv2

    def normalize_image(image):
        img_min = image.min()
        img_max = image.max()
        img_range = img_max - img_min
        if img_range < 1e-6:
            img_range = 1.0
        normalized_image = (image - img_min) / img_range * 255
        return normalized_image.astype(np.uint8)

    def draw_heatmap(heatmap):
        # Create a separate random number generator with fixed seed for consistent heatmap colors
        heatmap_rng = np.random.RandomState(42)
        res = np.zeros((*heatmap.shape[1:], 3), dtype=np.float32)
        for i in range(heatmap.shape[0]):
            color = heatmap_rng.uniform(0.2, 1.0, 3)
            res[..., 0] += heatmap[i] * color[0]
            res[..., 1] += heatmap[i] * color[1]
            res[..., 2] += heatmap[i] * color[2]
        return res

    def draw_bboxes(image, is_active, bboxes, centers=None):
        # Create a separate random number generator with fixed seed for consistent bbox colors
        bbox_rng = np.random.RandomState(123)
        image = np.ascontiguousarray(image)
        image = normalize_image(image)
        bboxes = (bboxes + 0.5).astype(int)
        if centers is not None:
            centers = (centers + 0.5).astype(int)
        for j in range(bboxes.shape[0]):
            if not is_active[j]:
                continue
            color = bbox_rng.randint(50, 255, 3).tolist()
            image = cv2.rectangle(
                image,
                bboxes[j, :2],
                bboxes[j, 2:],
                color=color,
            )
            if centers is not None:
                image = cv2.circle(image, centers[j], 1, color, -1)
        return image

    for i in range(args.batch_size):
        num_elements = 1 if use_single_images else len(data)
        for j in range(num_elements):
            # Get the current camera. If multiple image are used, it will be a child of `data`. Otherwise, it will be `data` itself.
            curr_cam_data = data[j] if not use_single_images else data

            image = np.array(curr_cam_data["image"][i].cpu())
            annotation = curr_cam_data["annotation"]

            bboxes = np.array(annotation["bboxes"][i].cpu())
            is_active = np.array(annotation["is_active"][i].cpu())
            bboxes_heatmap = np.array(annotation["bboxes_heatmap"][i].cpu())
            centers_heatmap = np.array(annotation["center"][i].cpu())
            heatmap = np.array(annotation["heatmap"][i].cpu())

            bbox_image = draw_bboxes(image, is_active, bboxes)
            dumper.add_tensor_data(
                f"data.bbox_image_{i}_{j}", torch.tensor(bbox_image), dumper.Type.IMAGE_RGB
            )

            hm_bbox_image = draw_heatmap(heatmap)
            hm_bbox_image = draw_bboxes(hm_bbox_image, is_active, bboxes_heatmap, centers_heatmap)
            dumper.add_tensor_data(
                f"data.bbox_heatmap_{i}_{j}", torch.tensor(hm_bbox_image), dumper.Type.IMAGE_RGB
            )


if __name__ == "__main__":

    # Simulationg a larger dataset
    os.environ["DATASET_REPEAT_FACTOR"] = str(2000)

    # from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser(description="DALI pipeline test")
    args = add_object_detection_2d_arguments_to_parser(parser).parse_args()

    args.num_iterations = 10
    args.num_warmup_iters = 1
    # args.num_workers = 2
    # args.prefetch_queue_depth = 1
    # args.num_iterations = 1000
    # args.num_warmup_iters = 100
    # args.batch_size = 1

    print(args)

    stopwatch = Stopwatch()
    stopwatch.enable(args.num_warmup_iters, None, True)
    stopwatch.set_cpu_usage_meas_name("train_loop")

    # Get the pipeline and the output data format blueprint (i.e. SampleDataGroup with the correct nesting & data fields, but with the actual data empty, i.e. set to None)
    config = get_default_object_detection_2d_pipeline_config(
        1, args.num_iterations, use_single_images=bool(args.use_single_images)
    )
    config = add_object_detection_2d_config_to_arguments(config, args)

    # Get the DALI pipeline wrapped so that it can be used as a drop-in replacement for a PyTorch DataLoader
    dali_data_provider = setup_dali_pipeline_2d_object_detection(
        defs.nuscenes_root_dir,
        defs.nuscenes_version,
        config,
        batch_size=args.batch_size,
        use_gpu=bool(args.pipeline_use_gpu),
        repeatable_seed=True,
    )
    dali_iter = iter(dali_data_provider)

    for i in range(args.num_iterations):
        stopwatch.start_meas("train_loop")
        nvtx_wrp.range_push("batch" + str(i))
        # When one epoch is finished, the iterator will raise `StopIteration` to indicate that.
        try:
            # Get the next batch
            batch_structured = next(dali_iter)
        except StopIteration:
            # In this simple example, we do not handle the end of an epoch in any way except to reset the iterator
            # to continue obtaining batches (for a new epoch)
            dali_iter.reset()

            # After starting a new epoch, get the next batch
            batch_structured = next(dali_iter)
        nvtx_wrp.range_pop()
        print('.', end="", flush=(i % 10 == 9))

        stopwatch.end_meas("train_loop")
        stopwatch.finish_iter()
    # Print a newline to separate the progress dots from the next output
    print()

    stopwatch.print_eval_times()

    dumper.add_tensor_data(
        "data",
        batch_structured,
        dumper.Type.JSON,
        dump_type_override={"image": dumper.Type.IMAGE_RGB, "heatmap": dumper.Type.BINARY},
    )
    dumper.run_if_enabled(lambda: draw_bboxes(batch_structured, args.use_single_images))
    dumper.dump()

    exit()
