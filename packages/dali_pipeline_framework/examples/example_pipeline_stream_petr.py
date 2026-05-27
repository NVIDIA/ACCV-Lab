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
import sys
import os

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

_current_dir = os.path.dirname(os.path.abspath(__file__))

nvtx_wrp = NVTXRangeWrapper()
nvtx_wrp.enable(sync_on_push=True, sync_on_pop=True, keep_track_of_range_order=False)

# We do activate the dumper here, but it is used only at the end of the script
# and not inside the main loop. It is used to show the data of the last
# performed iteration.
dumper = TensorDumper()
from pipeline_setup.stream_petr_pipeline import LiDARInstance3DBoxes, DataContainer

dumper.enable(os.path.join(_current_dir, "dump_stream_petr_pipeline"))
# Register custom converters for the external types
dumper.register_custom_converter(LiDARInstance3DBoxes, lambda x: {"tensor": x.tensor})
dumper.register_custom_converter(DataContainer, lambda x: {"data": x.data})


if __name__ == "__main__":

    # Simulationg a larger dataset
    os.environ["DATASET_REPEAT_FACTOR"] = str(2000)

    # from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser(description="DALI StreamPETR pipeline test")
    args = add_stream_petr_arguments_to_parser(parser).parse_args()

    args.num_iterations = 10
    args.num_warmup_iters = 1
    args.num_workers = 1
    args.prefetch_queue_depth = 1
    args.batch_size = 4
    # args.load_as_for_eval = 0
    # Needs to be true for now, no alternative implemented
    args.convert_format_for_training = False

    print(args)

    stopwatch = Stopwatch()
    stopwatch.enable(args.num_warmup_iters, None, True)
    stopwatch.set_cpu_usage_meas_name("train_loop")

    # Get the pipeline and the output data format blueprint (i.e. SampleDataGroup with the correct nesting & data fields, but with the actual data empty, i.e. set to None)
    config = get_default_stream_petr_pipeline_config(1, args.num_iterations)
    # Override config with command line arguments
    config = add_stream_petr_config_to_arguments(config, args)

    # `setup_dali_pipeline_stream_petr_train()` returns the iterator described in the StreamPETR example docs
    # page "StreamPETR Pipeline Setup", section "Wrap as Structured Iterator", including the PyTorch
    # DataLoader masking.
    dali_data_provider = setup_dali_pipeline_stream_petr_train(
        defs.nuscenes_root_dir, defs.nuscenes_version, sys.maxsize, config, batch_size=args.batch_size
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
            # In this simple example, we do not handle the end of an epoch in any way except to reset the iterator to continue obtaining batches (for a new epoch).
            dali_iter.reset()

            # After starting a new epoch, do the same as in the try block.
            batch_structured = next(dali_iter)

        nvtx_wrp.range_pop()
        print('.', end="", flush=(i % 10 == 9))

        stopwatch.end_meas("train_loop")
        stopwatch.finish_iter()
    # Print a newline to separate the progress dots from the next output
    print()

    stopwatch.print_eval_times()

    # For `image` and `img` we set up dumping as images. Note that
    #   - `image` are images as used inside the pipeline (i.e. RGB, color channel last)
    #   - `img` are images as used in the training loop, i.e.
    #     - Color channel before the spatial dimensions of the image
    #     - Has additional dimensions for the points in time & camera views
    #     - Has a batch dimension (this is the same as the batch dimension of the `image` tensor)
    #   - Due to colors not being the last dimension, we need to permute the axes to get the correct image format.
    #     This is why the `permute_axes_override` is set for `img`, but not for `image`.
    #   - Depending on whether the conversion is enabled or not, only `img` or `image` is available. However,
    #     it is ok to add both to the override dictionaries, as they may contain non-existing elements (will be ignored).
    dumper.add_tensor_data(
        "data",
        batch_structured,
        dumper.Type.JSON,
        dump_type_override={"image": dumper.Type.IMAGE_RGB, "img": dumper.Type.IMAGE_RGB},
        permute_axes_override={"img": (0, 1, 2, 4, 5, 3)},
    )
    dumper.dump()
