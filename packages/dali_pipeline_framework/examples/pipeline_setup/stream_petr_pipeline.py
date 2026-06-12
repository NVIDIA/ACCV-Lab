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

'''
StreamPETR training pipeline using DALI.

This pipeline reads NuScenes samples in a sequence-based way, applies image decoding and augmentations,
filters and prepares both 2D and 3D ground-truth annotations, and converts outputs into the
training format expected by StreamPETR via a dedicated post-processing function.

.. seealso::

    Please also see :doc:`../2d_object_detection/pipeline_setup`. Here, we assume that the reader is familiar
    with that example and refer to it for some details which apply to both pipelines.
'''

from functools import partial

from typing import Any

import numpy as np
import torch

from accvlab.dali_pipeline_framework.inputs import SequenceSampler, SamplerInputIterable, SamplerInputCallable

from accvlab.dali_pipeline_framework.processing_steps import (
    ImageDecoder,
    AffineTransformer,
    ImageToTileSizePadder,
    ImageMeanStdDevNormalizer,
    VisibleBboxSelector,
    DataGroupArrayWithNameElementsAppliedStep,
    PointsInRangeCheck,
    AnnotationElementConditionEval,
    ConditionalElementRemover,
    UnneededFieldRemover,
    CoordinateCropper,
    BEVBBoxesTransformer3D,
    PaddingToUniform,
)
from accvlab.dali_pipeline_framework.pipeline import PipelineDefinition, DALIStructuredOutputIterator

from .additional_impl.data_loading import *
from .additional_impl.processing_steps.stream_petr_data_combiner import StreamPETRDataCombiner

# @NOTE
# Use the external types as needed in the real training implementation. However, for the purpose of only
# running the example,also provide a dummy implementation if the external types are not available.
try:
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
except:
    import warnings

    warnings.warn(
        "`mmdet3d.core.bbox.LiDARInstance3DBoxes` could not be imported. Using dummy implementation instead."
    )

    class LiDARInstance3DBoxes:
        def __init__(self, tensor, *args, **kwargs):
            self._tensor = tensor

        @property
        def tensor(self):
            return self._tensor


try:
    from mmcv.parallel import DataContainer
except:
    import warnings

    warnings.warn("mmcv.parallel.DataContainer could not be imported. Using dummy implementation instead.")

    class DataContainer:
        def __init__(self, data, *args, **kwargs):
            self._data = data

        @property
        def data(self):
            return self._data


from accvlab.optim_test_tools import Stopwatch

stopwatch = Stopwatch()


def dali_structured_to_torch(batch, for_training) -> Any:
    '''Method used as a post-processing step to align data format to the format used in StrreamPETR training.

    The conversion includes the conversion to OpenMM data formats (where needed) and re-structuring of the
    data where the samples are combined into batches different than in the original collation implementation
    used in the PyTorch DataLoader data pipeline.

    Args:
        batch: DALI output batch. The can be either a :class:`dict` or a
            :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup`, depending on whether the used
            :class:`~accvlab.dali_pipeline_framework.pipeline.DALIStructuredOutputIterator` was configured to
            convert the output to a dictionary or not.
        for_training: Whether the data is for training. Note that this parameter is not part of the
            convention for the post-processing function, and e.g. :func:`func_tools.partial` can be used to
            fix this parameter and remove it from the function signature.

    Returns:
        The converted batch. There are no restrictions on the format, but typically, this will be a
        :class:`dict`.
    '''

    # ===== Helper and input shape =====

    # @NOTE Helper to convert tensors to LiDARInstance3DBoxes (used for 3D GT in the training implementation);
    def tensor_to_lidar_inst(tensor):
        return LiDARInstance3DBoxes(tensor, box_dim=tensor.size(-1))

    # Get the shape of the (concatenated) images as well as the batch size (corresponds to the first dimension
    # of the images)
    img_shape = tuple(batch["img"].shape)
    batch_size = img_shape[0]

    # ===== Wrap core tensors as DataContainer objects =====
    # @NOTE Add a sequence dimension (length 1, as we use the streaming video training mode) and wrap tensors
    # in DataContainer objects as expected by StreamPETR training code.
    #
    # Convert data fields by unsqueezing (the new dimension represents time steps, but only single time steps
    # are used in the relevant setups) and wrapping as DataContainer, as this is what
    # the training expects.
    for key in [
        'lidar2img',
        'intrinsics',
        'extrinsics',
        'timestamp',
        'img_timestamp',
        'ego_pose',
        'ego_pose_inv',
        'img',
        'prev_exists',
    ]:
        # @NOTE Get as one tensor and add a dimension for the sequence length (always 1)
        tensor_to_store = batch[key]
        tensor_with_seq_dim = torch.unsqueeze(tensor_to_store, axis=1)
        batch[key] = DataContainer([tensor_with_seq_dim], cpu_only=False, stack=True, pad_dims=None)

    # ===== Process ground truth (training only) =====
    # @NOTE For training: trim padded regions, wrap GT in DataContainer, and add sequence/batch nesting to match
    # consumer expectations.
    # If the data is for training ...
    if for_training:
        # ----- Process 2D ground truth -----
        num_gt_obj = batch["num_gt_objects"]
        for key in ['gt_bboxes', 'gt_labels', 'centers2d', 'depths']:
            num_cams = batch[key][0].shape[0]
            # Note that the following line performs the following functions
            # - In the innermost dimension (iterating over individual objects), only as many elements are
            #   retained as there are objects (indexed as `[0:num_gt_obj[s][c]]`). This is needed because the
            #   individual samples were padded to the largest number of objects encountered for any camera in
            #   the batch (to be filled into a single tensor). The padded regions are removed here and
            #   `num_gt_obj` (from data field `batch["num_gt_objects"]`) stores the actual number of objects
            #   for each sample and camera.
            # - The data is wrapped in a DataContainer, as this is what the training
            #   expects.
            # - Dimensions are added: Note that the list comprehensions are enclosed by another pair of `[]`.
            #   For the inner list comprehension, the added dimension corresponds to the sequence of time
            #   steps (always length 1).
            batch[key] = DataContainer(
                [
                    [
                        [[batch[key][s][c][0 : num_gt_obj[s][c]] for c in range(num_cams)]]
                        for s in range(batch_size)
                    ]
                ],
                cpu_only=False,
            )

        # ----- Process 3D ground truth -----
        # The following lines perform the following functions
        # - Make sure that only as many elements are retained as there are objects (stored in
        #   `batch["num_gt_objects_3d"]`). Similar to the 2D GT data, the data was padded to the maximum
        #   number of objects in the batch, and this needs to be reversed here.
        # - The data is wrapped in a DataContainer, as this is what the training
        #   expects
        # - For the ground truth bounding boxes, the format is converted from PyTorch tensors to
        #   `LiDARInstance3DBoxes` (as this format is expected in the training)
        # - Dimensions are added (similar to the other tensors, described above)
        num_gt_obj_3d = batch["num_gt_objects_3d"]
        batch["gt_labels_3d"] = DataContainer(
            [[[batch["gt_labels_3d"][s][0 : num_gt_obj_3d[s]]] for s in range(batch_size)]], cpu_only=False
        )
        batch["gt_bboxes_3d"] = DataContainer(
            [
                [
                    [tensor_to_lidar_inst(batch["gt_bboxes_3d"][s][0 : num_gt_obj_3d[s]])]
                    for s in range(batch_size)
                ]
            ],
            cpu_only=True,
        )

    # ===== Set image metas =====
    # @NOTE Populate img_metas to align with training expectations; much of this information exists elsewhere,
    # but this is done to replaicate the training data format from the original implementation.
    img_shape_to_set = [(img_shape[3], img_shape[4], img_shape[2])] * img_shape[1]
    img_metas = [None] * batch_size
    for s in range(batch_size):
        image_metas_sample = {}
        image_metas_sample["img_shape"] = img_shape_to_set
        image_metas_sample["pad_shape"] = img_shape_to_set
        if for_training:
            # Here, `batch["gt_bboxes_3d"]` and `batch["gt_labels_3d"]` are already wrapped in a DataContainer.
            # Access accordingly (with `.data[0]`)
            image_metas_sample["gt_bboxes_3d"] = DataContainer(
                batch["gt_bboxes_3d"].data[0][s][0], cpu_only=False
            )
            image_metas_sample["gt_labels_3d"] = DataContainer(
                batch["gt_labels_3d"].data[0][s][0], cpu_only=False
            )
        img_metas[s] = [image_metas_sample]

    batch["img_metas"] = DataContainer([img_metas], cpu_only=True)

    # ===== Cleanup =====
    # @NOTE Remove fields used only internally to reverse padding; they are used in this function to reverse
    # the padding, and are not needed downstream.
    del batch["num_gt_objects"]
    del batch["num_gt_objects_3d"]

    return batch


def setup_dali_pipeline_stream_petr_train(
    nuscenes_root_dir: str,
    nuscenes_version: str,
    num_batches_in_epoch: int,
    config,
    device_id: int = 0,
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = 1,
    val_sequences=['scene-0103', 'scene-0916'],
) -> DALIStructuredOutputIterator:
    '''Setup DALI pipeline for StreamPETR training.

    Params
    ------
    nuscenes_root_dir : str
        Root directory of the nuscenes dataset
    nuscenes_version : str
        Version of the nuscenes dataset
    num_batches_in_epoch : int
        Number of batches in an epoch. Note that the data sampling used means that
        there are no real epochs. However, an epoch length is still used to define
        how often certain actions are performed (e.g. print out training info, store checkpoint etc.)
    config
        Configuration for the DALI pipeline. See example configuration files for details.
        Example file: "projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e_playground_dali.py"
    device_id : int = 0
        GPU device ID used for the pipeline (and the training)
    rank : int = 0
        Rank of the current training process (should be 0 for single-GPU training)
    world_size : int = 1
        World size (should be 1 for single-GPU training)
    batch_size : int = 1
        Batch size. In case of multi-GPU training, this is the batch size for a single GPU
    val_sequences = ['scene-0103', 'scene-0916']
        Scene names in the dataset to be not used during training. This is used with the NuScenes mini version to
        reserve some datasets for validation.

    Returns
    -------
    dali_structures_output_iterator : DALIStructuredOutputIterator
        Iterator for getting training data. Mimics a PyTorch DataLoader.
    '''

    # ===== Input data handling =====

    # @NOTE
    # Goal: Create an input implementation (iterable or callable) that provides sequence-based samples for
    # StreamPETR. See the 2D object detection example for general input callable mechanics and
    # sharding/shuffling notes.

    # @NOTE
    # Try to load previously converted data. If this is not possible, read in the original format & convert.
    nuscenes_preproc_file_name = '{}_preproc_{}.pkl'.format(nuscenes_version, "streampetr")
    input_data = NuScenesReader.load_data_if_available_else_create_and_store(
        nuscenes_root_dir,
        nuscenes_preproc_file_name,
        nuscenes_version,
        add_projected_bboxes_annotations=True,
        image_size=config.image_size,
    )

    # @NOTE
    # Remove dedicated validation scenes and perform basic dataset reshaping (sort, split sequences) for
    # training.
    input_data = input_data.get_subset_without_sequences(val_sequences)
    # Sort scenes by timestamps
    input_data = input_data.get_sequences_sorted_by_start_time()
    # Split each scene into 2 sequences. This increases the number of sequences, and is also done in the
    # original StreamPETR pipeline.
    input_data = input_data.get_with_sequences_split(2)

    # @NOTE Create a data provider for the StreamPETR use-case.
    input_provider = NuscenesStreamPETRDataProvider(
        nuscenes_root_dir,
        input_data,
        image_size=config.image_size,
        return_ground_truth=True,
    )

    # @NOTE Total batch size for multi-GPU training (used by the sampler).
    total_batch_size = batch_size * world_size
    # @NOTE
    # Sampler defines per-batch sequence indices; can randomize selection based on config.
    # Note that `input_data.get_sequence_lengths()` is used to get the sequence lengths, so that the sampler
    # knows how the dataset is divided into the individual sequences.
    sequence_sampler = SequenceSampler(
        total_batch_size, input_data.get_sequence_lengths(), 21, randomize=config.randomize_sample_selection
    )
    # @NOTE Create either an input iterable or input callable using the sampler and data provider.
    if config.use_input_iterable:
        input_impl = SamplerInputIterable(input_provider, sequence_sampler, rank, world_size)
    else:
        input_impl = SamplerInputCallable(
            input_provider,
            sequence_sampler,
            config.max_iterations_input_callable,
            config.prefetch_queue_depth,
            rank,
            world_size,
        )

    # ===== Define processing steps =====
    # @NOTE The following pre-processing steps mirror the 2D object detection example in spirit, adapted to
    # StreamPETR specifics. Refer to the 2D object detection example for details on common concepts
    # (e.g. name-based data selection, optional steps, padding, etc.).

    decoder = ImageDecoder(
        "image",
        use_device_mixed=True,
        hw_decoder_load=config.decoder_config.hw_decoder_load,
        as_bgr=not config.decoder_config.as_rgb,
    )

    # @NOTE
    # Affine augmentation: define the sequence of transforms (flip, scaling, alignment) and then create the
    # step.
    image_flip_prob = 0.5 if config.affine_trafo_config.rand_flip else 0.0
    transformation_steps = [
        # Perform a horizontal image flip (i.e. scale along the x-axis with a fixed factor of -1.0, and a
        # factor of 1.0 (i.e. unchanged) for the y-axis) with a probability of 0.5 if random flips are enabled
        # in the configuration, and with a probability of 0.0 otherwise
        AffineTransformer.NonUniformScaling(image_flip_prob, [-1.0, 1.0]),
        # Scaling in a configured range
        AffineTransformer.UniformScaling(
            1.0, config.affine_trafo_config.resize_lim[0], config.affine_trafo_config.resize_lim[1]
        ),
        # NOTE
        # Align bottom border of the viewport with the original image. Note that such alignment operations are
        # also part of the `AffineTransformer` step (and not only random augmentations).
        AffineTransformer.ShiftToAlignWithOriginalImageBorder(
            1.0, AffineTransformer.ShiftToAlignWithOriginalImageBorder.Border.BOTTOM
        ),
    ]
    # @NOTE
    # Create the actual affine transformation step. Apart from the affine transformation itself, the step
    # additionally applies resizing to needed output image size and ensures that the image remains aligned
    # to the bottom border of the viewport, cropping on the top if needed.
    image_trafo = AffineTransformer(
        output_hw=config.affine_trafo_config.output_hw,
        resizing_mode=AffineTransformer.ResizingMode.CROP,
        resizing_anchor=AffineTransformer.ResizingAnchor.BOTTOM_OR_RIGHT,
        image_field_names="image",
        projection_matrix_field_names=["lidar2img", "intr_lidar2img"],
        point_field_names=["centers", "bboxes"],
        transformation_steps=transformation_steps,
        transform_image_on_gpu=True,
    )
    # @NOTE Pad images so width/height are divisible by 32 (as expected by training implementation).
    image_padder = ImageToTileSizePadder("image", 32)
    # @NOTE Normalize images based on pre-defined mean & std.dev. values.
    normalizer = ImageMeanStdDevNormalizer("image", config.img_norm_cfg.mean, config.img_norm_cfg.std)

    # @NOTE
    # Select visible 2D bboxes (not completely occluded by closer bboxes and not too small)
    # Note that this step is applied per camera, as it expects to find exactly one data field for each
    # role (e.g. bonuding boxes, depths etc.), so that the correspondences between these fields are clear.
    visible_bbox_selector_per_cam = VisibleBboxSelector(
        "bboxes",
        resulting_mask_field_path=("gt_boxes_2d", "is_visible"),
        image_field_name="image",
        check_for_bbox_occlusion=True,
        check_for_minimum_size=True,
        depths_field_name="depths",
        minimum_bbox_size=2.0,
    )
    # @NOTE Ensure the step is applied per camera using an access modifier wrapper step.
    visible_bbox_selector = DataGroupArrayWithNameElementsAppliedStep(visible_bbox_selector_per_cam, "cams")

    # @NOTE Check whether 3D bbox centers are within a configured range.
    point_cloud_range_min = config.point_cloud_range.min
    point_cloud_range_max = config.point_cloud_range.max
    # @NOTE None denotes no border → replaced by -inf/inf for computation.
    for i in range(3):
        if point_cloud_range_min[i] is None:
            point_cloud_range_min[i] = -np.inf
        if point_cloud_range_max[i] is None:
            point_cloud_range_max[i] = np.inf
    bbox_inside_range_selector = PointsInRangeCheck(
        "translations", "is_bbox_in_range", point_cloud_range_min, point_cloud_range_max
    )

    # @NOTE
    # Mark active objects (3D) based on multi-sensor evidence and range checks; keep inputs for later steps.
    # Please see the 2D object detection example for more details on the `AnnotationElementConditionEval` step.
    active_value_selector = AnnotationElementConditionEval(
        "gt_boxes",
        condition="is_active = (num_lidar_points >= 1 or num_radar_points >= 1) and categories >= 0 and is_bbox_in_range",
        remove_data_fields_used_in_condition=False,
    )

    # @NOTE Mark active objects (2D) using category and visibility flags from the previous step.
    active_value_selector_2d = AnnotationElementConditionEval(
        "gt_boxes_2d",
        condition="is_active = categories >= 0 and is_visible",
        remove_data_fields_used_in_condition=False,
    )

    # @NOTE
    # Remove inactive elements (3D and 2D) based on the active masks (generated by the
    # `AnnotationElementConditionEval` steps).
    inactive_remover = ConditionalElementRemover(
        "gt_boxes",
        "is_active",
        ["categories", "sizes", "rotations", "translations", "orientations", "velocities"],
        [0, 0, 0, 0, 0, 0],
        [1, 2, 2, 2, 1, 2],
        True,
    )
    inactive_remover_2d = ConditionalElementRemover(
        "gt_boxes_2d",
        "is_active",
        ["bboxes", "centers", "depths", "categories"],
        [0, 0, 0, 0],
        [2, 2, 1, 1],
        True,
    )

    # @NOTE Remove fields that are no longer needed for further processing.
    unneded_fields_remover_post_check = UnneededFieldRemover(
        ["num_lidar_points", "num_radar_points", "is_visible", "is_bbox_in_range"]
    )

    # @NOTE
    # Crop 2D centers and bboxes to the output image size (after cropping/padding).
    # First, compute tiled output dims expected downstream ...
    out_img_size_x = config.affine_trafo_config.output_hw[1]
    out_img_size_y = config.affine_trafo_config.output_hw[0]
    out_img_xize_x_tiled = ((out_img_size_x + 31) // 32) * 32
    out_img_xize_y_tiled = ((out_img_size_y + 31) // 32) * 32
    # @NOTE
    # ... Then, crop bbox centers ...
    bbox_cropper_centers = CoordinateCropper(
        "centers", [0.0, 0.0], [out_img_xize_x_tiled, out_img_xize_y_tiled]
    )
    # @NOTE
    # ... Then, crop bboxes (2 points per bbox, same limits apply)
    bbox_cropper_bboxes = CoordinateCropper(
        "bboxes",
        [0.0, 0.0, 0.0, 0.0],
        [out_img_xize_x_tiled, out_img_xize_y_tiled, out_img_xize_x_tiled, out_img_xize_y_tiled],
    )

    # @NOTE Optional BEV-space 3D bbox augmentations.
    if config.perform_3d_augmentation:
        transformer_3d = BEVBBoxesTransformer3D(
            "translations",
            "velocities",
            "sizes",
            "orientations",
            ["lidar2img", "extr_lidar2img"],
            "lidar_ego_pose",
            "lidar_ego_pose_inv",
            config.bev_bboxes_trafo_config.rotation_range,
            2,
            config.bev_bboxes_trafo_config.scaling_range,
            config.bev_bboxes_trafo_config.translation_range,
        )
    else:
        transformer_3d = None

    # @NOTE
    # Combine and convert intermediate outputs into a format close to StreamPETR training expectations.
    # Note that:
    #   - The final format cannot be exactly matched, as it contains some framework-specific containers and
    #     batching conventions that cannot be represented in DALI
    #   - At the same time, the format is brought as close as feasible to the final format within the DALI
    #     pipeline to minimize the amount of post-processing needed outside the DALI pipeline (see
    #     `dali_structured_to_torch()` in this file). This is advisable for performance reasons, as the
    #     post-processing is done in the main training thread.
    #
    # Also, please note that this step is very specific to the StreamPETR training use-case. Therefore,
    # it is part of the example, and not included in the core DALI pipeline package. You can find it in
    # in the repository under
    # `packages/dali_pipeline_framework/examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py`.
    training_data_format_converter = StreamPETRDataCombiner(True)

    # @NOTE
    # Apply padding so variable-length arrays become uniform across the batch. Note that only specific fields
    # are padded. This is similar to the 2D object detection example, but here, the fields are defined by
    # name (instead of specifying a root field, and padding everything it contains).
    padding_to_uniform = PaddingToUniform(
        field_names=["gt_bboxes_3d", "gt_labels_3d", "gt_bboxes", "depths", "centers2d", "gt_labels"]
    )

    # @NOTE Define the pipeline steps sequence; `None` entries are ignored.
    pre_processing_steps = [
        decoder,
        image_trafo,
        image_padder,
        normalizer,
        visible_bbox_selector,
        bbox_inside_range_selector,
        active_value_selector,
        active_value_selector_2d,
        inactive_remover,
        inactive_remover_2d,
        unneded_fields_remover_post_check,
        bbox_cropper_centers,
        bbox_cropper_bboxes,
        transformer_3d,
        training_data_format_converter,
        padding_to_uniform,
    ]

    # ===== Pipeline definition & Output data format =====
    # @NOTE
    # Define the pipeline by wiring the input implementation and the pre-processing steps.
    #
    # IMPORTANT: Note that `check_data_format` is set to `False` here. This is done to avoid the overhead of
    # checking the data format during pipeline execution. During development, it is recommended to set it
    # to `True` to catch potential issues early, and later set it to `False` to avoid the overhead in
    # production.
    pipeline_def = PipelineDefinition(
        input_impl,
        pre_processing_steps,
        check_data_format=False,
        print_sample_data_group_format=True,
        # Enable the DALI external-source pass-through issue copy workaround only for final fields that are or
        # may be passed through the pipeline unchanged. See the API docs for this constructor argument for
        # details about the underlying issue and when a copy is needed.
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["timestamp", "ego_pose", "ego_pose_inv"],
    )

    # @NOTE
    # Check and capture the output data structure blueprint. See the 2D object detection example for an
    # in-depth description.
    res_data_setup = pipeline_def.check_and_get_output_data_structure()

    # ===== Create DALI pipeline =====
    # @NOTE:
    # Seed for augmentation reproducibility (rank-specific). Please also see the 2D object detection example
    # for more details.
    if config.use_reproducible_seed:
        seed = rank * 10 + 1
    else:
        seed = -1

    # Start measuring creation time
    stopwatch.start_one_time_measurement("pipe_create")
    # @NOTE Obtain the actual DALI Pipeline object (see DALI getting started docs).
    pipe = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=batch_size,
        prefetch_queue_depth=config.pipeline_config.prefetch_queue_depth,
        py_num_workers=config.pipeline_config.num_workers,
        py_start_method="spawn",
        num_threads=config.pipeline_config.num_threads,
        device_id=device_id,
        seed=seed,
    )
    # End measuring creation time
    stopwatch.end_one_time_measurement("pipe_create")

    # Start measuring build time
    stopwatch.start_one_time_measurement("pipe_build")

    # For details on preparing the pipeline see:
    # case spawn:
    # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/parallel_external_source.html
    # case fork:
    # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/parallel_external_source_fork.html
    pipe.build()
    # End measuring build time
    stopwatch.end_one_time_measurement("pipe_build")

    # ===== Wrap as iterator =====
    # @NOTE
    # Wrap the pipeline as an DALIStructuredOutputIterator, which can be used as a drop-in replacement for a
    # PyTorch DataLoader.
    #
    # Note that apart from the pipeline, the info on the epoch size and the data format of the output
    # (`res_data_setup`) need to be set explicitly in the iterator wrapper.
    #
    # Also, we use the `CreateAsDataLoaderObject()` method rather than the constructor directly. This ensures
    # that the iterator object is masked as a PyTorch DataLoader object, so that checks such as
    # `assert isinstance(iterator_object, DataLoader)` inside the training implementation pass.
    dali_structured_to_torch_used = partial(dali_structured_to_torch, for_training=True)
    # @NOTE: Set up the iterator
    res_iterator = DALIStructuredOutputIterator.CreateAsDataLoaderObject(
        num_batches_in_epoch, pipe, res_data_setup, post_process_func=dali_structured_to_torch_used
    )

    # Print measured times
    stopwatch.print_eval_times()

    # Return the wrapped pipeline.
    return res_iterator
