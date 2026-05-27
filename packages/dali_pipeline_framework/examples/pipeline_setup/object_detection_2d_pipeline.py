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
This 2D object detection pipeline reads NuScenes samples, applies image decoding and augmentations,
and processed the ground truth bounding boxes (generating gaussian heatmaps and providing related
information).

The pipeline is configured here (see function ``setup_dali_pipeline_2d_object_detection()`` below).
It is meant as an example for the use of the DALI pipeline framework.

'''

from accvlab.optim_test_tools import Stopwatch

stopwatch = Stopwatch()

from accvlab.dali_pipeline_framework.inputs import *
from accvlab.dali_pipeline_framework.processing_steps import *
from accvlab.dali_pipeline_framework.pipeline import PipelineDefinition
from accvlab.dali_pipeline_framework.pipeline import DALIStructuredOutputIterator

from .additional_impl.data_loading import *


def setup_dali_pipeline_2d_object_detection(
    nuscenes_root_dir: str,
    nuscenes_version: str,
    config,
    device_id: int = 0,
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = 1,
    use_gpu: bool = True,
    repeatable_seed: bool = False,
) -> DALIStructuredOutputIterator:

    # ===== Input data handling =====

    # @NOTE
    # Goal: Create an input callable (functor) which is responsible for providing the input one sample at a
    # time.

    # @NOTE
    # Here, we use a NuScenesReader to read the metadata and store it in memory in a format facilitating fast
    # & easy sample ratrieval.
    #
    # Try to load previously converted data. If this is not possible, read in the original format & convert
    nuscenes_preproc_file_name = '{}_preproc_{}.pkl'.format(nuscenes_version, "det2d")
    input_data = NuScenesReader.load_data_if_available_else_create_and_store(
        nuscenes_root_dir,
        nuscenes_preproc_file_name,
        nuscenes_version,
        # Note that to use `add_image_annotations=True`, the image annotations need to be present.
        # They can be generated using the `prepare_dataset.py` script in the DALI pipeline framework examples.
        add_image_annotations=True,
    )

    # @NOTE
    # Create a data provider for this use case.
    #
    # Note that this data provider can output either a single image or all camera images as one single sample.
    # (set as `config.image_config.use_single_images`). The use of single images is the main use-case,
    # and the use of multiple images is for demonstration purposes.
    input_provider = Nuscenes2DDetectionDataProvider(
        nuscenes_root_dir, input_data, not config.image_config.use_single_images
    )

    # @NOTE
    # The actual callable uses the `input_provider` to provide the input data  Here, we use the general
    # `ShuffledShardedInputCallable` provided by the package, and pass the data provider to it to
    # configure it according to our use-case.
    #
    # As a side note, note that if sharding is used, the `seed` needs to be the same for all shards to ensure
    # consistent splits of the dataset into the shards among the input callable objects of all shards.
    input_callable = ShuffledShardedInputCallable(
        input_provider, batch_size, shuffle=True, shard_id=rank, num_shards=world_size, seed=21
    )

    # ===== Define processing steps =====
    # @NOTE: The procesing steps are defined here and composed into a pipeline afterwards.

    # @NOTE
    # Note that all the used steps work both for single-image input and for samples containing all camera
    # images.
    #
    # Please see the introduction in the `dali-pipeline` documentation for details on the
    # input data structure and how the individual steps are designed to achieve this flexibility.
    #
    # Also, note that there are wrapper classes available which allow to modify the steps to define which
    # data is processed independently (e.g. different augmentations for different images) or consistently
    # (e.g. same augmentation for multiple images). These wrapper classes are not used in this example
    # pipeline, but are needed in other use-cases (see the documentation of the package, especially the
    # introduction to the `dali-pipeline` package, and the API documentation for the `PipelineStepBase`
    # class).

    # @NOTE
    # Image decoding step: Decode jpeg-images (CPU or GPU, depending on the configuration).
    decoder = ImageDecoder(
        "image",
        use_device_mixed=config.decoder_config.use_device_mixed,
        hw_decoder_load=config.decoder_config.hw_decoder_load,
        as_bgr=False,
    )

    # @NOTE: Affine augmentation step

    # @NOTE: First, we need to define the transformations to perform.
    transformation_steps = [
        # Perform a horizontal image flip (i.e. scale along the x-axis with a fixed factor of -1.0, and a
        # factor of 1.0 (i.e. unchanged) for the y-axis) with a probability of 0.5
        AffineTransformer.NonUniformScaling(config.augmentation_config.flip_probability, [-1.0, 1.0]),
        # Select one of two options with respective probabilities and perform all steps in the chosen option
        AffineTransformer.Selection(
            1.0,
            # Probabilities for the different options. Note that exactly one option will be selected according
            # to the probabilities, and that therefore, the probabilities have to sum up to 1
            config.augmentation_config.scaling_probabilities,
            # Options
            [
                # Option 1: Scaling in the range [0.6; 1.4] followed by a translation in the range
                # [(-100, -100); (100, 100)] followed by a.
                [
                    AffineTransformer.UniformScaling(
                        1.0,
                        config.augmentation_config.scaling_ranges[0][0],
                        config.augmentation_config.scaling_ranges[0][1],
                    ),
                    AffineTransformer.Translation(
                        1.0,
                        [
                            config.augmentation_config.translation_ranges[0][0],
                            config.augmentation_config.translation_ranges[0][1],
                        ],
                        [
                            config.augmentation_config.translation_ranges[0][2],
                            config.augmentation_config.translation_ranges[0][3],
                        ],
                    ),
                ],
                # Option 2: Larger scaling with a fixed factor of 2.0. As we scale more, also allow for
                # larger translations in the range [(-300, -300); (300, 300)]
                [
                    AffineTransformer.UniformScaling(
                        1.0,
                        config.augmentation_config.scaling_ranges[1][0],
                        config.augmentation_config.scaling_ranges[1][1],
                    ),
                    AffineTransformer.Translation(
                        1.0,
                        [
                            config.augmentation_config.translation_ranges[1][0],
                            config.augmentation_config.translation_ranges[1][1],
                        ],
                        [
                            config.augmentation_config.translation_ranges[1][2],
                            config.augmentation_config.translation_ranges[1][3],
                        ],
                    ),
                ],
            ],
        ),
    ]

    # @NOTE
    # Then, we create the actual affine transformation pipeline step. Note how multiple data fields are
    # processed in the same step (images, projection matrices, point sets). All the occurences of all the data
    # fields will be processed consistently, i.e. with the same transformation (could be configured to do
    # otherwise by using `GroupToApplyToSelectedStepBase`-derived wrapper classes).
    affine_transformer = AffineTransformer(
        output_hw=config.image_config.output_hw,
        resizing_mode=AffineTransformer.ResizingMode.PAD,
        resizing_anchor=AffineTransformer.ResizingAnchor.CENTER,
        image_field_names="image",
        projection_matrix_field_names=None,
        point_field_names="bboxes",
        transformation_steps=transformation_steps,
        transform_image_on_gpu=bool(config.decoder_config.use_device_mixed),
    )

    # @NOTE: Image normalization (applied to all images with name "image")
    image_normalizer = ImageRange01Normalizer("image")

    # @NOTE
    # Valid annotation selection step: Create the processing step evaluating which objects are valid. Note
    # that as "annotation" is used here as the data field group name for which the processing step is applied.
    # All the names defined in the condition are expected to be children of any "annotation" field
    # encountered in the data structure (as the step is applied to each annotation field).
    # The result will also be stored as a child to the "annotation" fields, in this case with the name
    # "is_valid" (defined in the condition).
    #
    # All the data fields used in the comparisons (corresponding to the names contained in the condition) will
    # be deleted (as `remove_data_fields_used_in_condition=True`). This is convenient here, but in other cases
    # some of the data fields may still be needed later. In that case, we can set
    # `remove_data_fields_used_in_condition=False` and delete the unneeded fields manually (using
    # `UnneededFieldRemover`).
    valid_value_selector = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="is_valid = (num_lidar_points >= 1 or num_radar_points >= 1) and visibility_levels > 0",
        remove_data_fields_used_in_condition=True,
    )

    # @NOTE
    # Heatmap conversion step: Generate heatmaps and related data from bounding boxes. This step expects some
    # specific data fields to be available as children for each data group field with the name "annotation"
    # in the input data.
    heatmap_converter = BoundingBoxToHeatmapConverter(
        image_field_name="image",
        annotation_field_name="annotation",
        bboxes_in_name="bboxes",
        categories_in_name="categories",
        heatmap_out_name="heatmap",
        num_categories=config.heatmap_config.num_categories,
        heatmap_hw=config.heatmap_config.heatmap_hw,
        is_valid_opt_in_name="is_valid",
        center_opt_in_name=None,  # Compute the center from the bounding box.
        is_active_opt_out_name="is_active",
        center_opt_out_name="center",
        center_offset_opt_out_name="center_offset",
        height_width_bboxes_heatmap_opt_out_name="height_width_bboxes_heatmap",
        bboxes_heatmap_opt_out_name="bboxes_heatmap",
        max_radius=config.heatmap_config.max_radius,
    )

    # @NOTE
    # Cleanup step: This processing step does not perform any actual computations. It is instead responsible
    # for removing unneeded fields from the input data structure. The fields with the given names are removed
    # no matter where in the input data structure they appear.
    #
    # This is done here as they are not needed anymore and outputting them from the pipeline would potentially
    # lead to unneccesary copies (to make the individual samples continuous in memory). Note that this step is
    # completely performed at the pipeline construction (i.e. DALI graph construction time), and does not add
    # any runtime overhead when running the pipeline.
    unneeded_fields_remover = UnneededFieldRemover(["image_hw", "is_valid"])

    # @NOTE
    # Pad all children of data group fields with the name "annotation".
    # Note that the padding is performed only to the annotations:
    #   - As it is known that non-uniform sizes are only present there, so no need to check other fields
    #   - As this step expects all contained data to be non-scalar, and thus, care must be taken that no
    #     scalar data fields are present.
    padding_to_uniform = PaddingToUniform(field_names="annotation", fill_value=0.0)

    # @NOTE
    # Store the steps as a sequence in the order in which they are going to be processed by the pipeline. Note
    # that that 'image_normalizer' is an optional step. If it is not needed, 'None' is used instead. This is
    # interpreted by the pipeline as a no-op and is ignored.
    pre_processing_steps = [
        decoder,
        affine_transformer,
        (image_normalizer if config.image_config.normalize else None),
        valid_value_selector,
        heatmap_converter,
        unneeded_fields_remover,
        padding_to_uniform,
    ]

    # ===== Pipeline definition & Output data format =====
    # @NOTE
    # Define the pipeline consisting of the 'input_callable' and 'pre_processing_steps'.
    #
    # IMPORTANT: Note how `check_data_format` is set to `False` here. This is done to avoid the overhead of
    # checking the data format during pipeline execution. During development, it is recommended to set it
    # to `True` to catch potential issues early, and later set it to `False` to avoid the overhead in
    # production.
    pipeline_def = PipelineDefinition(
        input_callable, pre_processing_steps, check_data_format=False, print_sample_data_group_format=True
    )

    # @NOTE
    # This is the output data format blueprint (also documentation of `SampleDataGroup` for a description of
    # the blueprint concept). This means it is a `SampleDataGroup` with the data format as the actual output
    # (i.e. same (nested) data group fields & data fields, data types), but without the actual data. As the
    # output of the pipeline will be flattened into a sequence, this blueprint can be used to fill it back to
    # the hierarchical `SampleDataGroup` structure.
    res_data_setup = pipeline_def.check_and_get_output_data_structure()

    # ===== Create DALI pipeline =====
    # @NOTE
    # Set a pre-defined seed used in augmentation if the results need to be repeatable. Note that the seed is
    # different for each GPU (rank) in case of multi-GPU training to avoid performing the exact same
    # augmentations. A value of -1 means a unique seed will be used every time.
    #
    # This is different from the shuffling seed, which needs to be the same for all GPUs to enable consistent
    # sharding.
    if repeatable_seed:
        seed = rank * 10 + 1
    else:
        seed = -1

    # Start measuring the time for creating the pipeline.
    stopwatch.start_one_time_measurement("pipe_create")

    # @NOTE
    # This step is for obtaining the actual DALI Pipeline object (see DALI getting started tutorial:
    # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting_started.html)
    pipe = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=batch_size,
        prefetch_queue_depth=config.pipeline_config.prefetch_queue_depth,
        py_num_workers=config.pipeline_config.num_workers,
        py_start_method="spawn",
        num_threads=config.pipeline_config.num_threads,
        device_id=device_id if use_gpu else None,
        seed=seed,
    )

    # End measuring the time for creating the pipeline.
    stopwatch.end_one_time_measurement("pipe_create")

    # Start measuring the time for building the pipeline.
    stopwatch.start_one_time_measurement("pipe_build")

    # @NOTE
    # For details on preparing the pipeline see:
    # case spawn:
    # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/parallel_external_source.html
    #
    # case fork:
    # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/parallel_external_source_fork.html
    pipe.build()

    # End measuring the time for building the pipeline.
    stopwatch.end_one_time_measurement("pipe_build")

    # Print the times.
    stopwatch.print_eval_times()

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
    res_iterator = DALIStructuredOutputIterator.CreateAsDataLoaderObject(
        input_callable.length, pipe, res_data_setup
    )

    return res_iterator
