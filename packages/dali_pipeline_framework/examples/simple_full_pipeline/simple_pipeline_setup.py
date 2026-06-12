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

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable
from accvlab.dali_pipeline_framework.pipeline import PipelineDefinition, DALIStructuredOutputIterator
from accvlab.dali_pipeline_framework.processing_steps import (
    PhotoMetricDistorter,
    DataGroupArrayInPathElementsAppliedStep,
    ImageRange01Normalizer,
    AnnotationElementConditionEval,
)

from simple_data_provider import ExampleDataProvider


def setup_simple_pipeline(
    batch_size: int = 2,
    num_samples: int = 8,
    num_cameras: int = 2,
    image_height: int = 64,
    image_width: int = 64,
    device_id: int = 0,
    seed: int = 42,
) -> DALIStructuredOutputIterator:
    '''Build and return a ready-to-iterate DALI pipeline for the simple example.

    Args:
        batch_size: Number of samples per batch.
        num_samples: Total number of samples available.
        num_cameras: Number of cameras per sample.
        image_height: Height of the generated images in pixels.
        image_width: Width of the generated images in pixels.
        device_id: GPU device id.
        seed: Random seed for augmentation reproducibility.

    Returns:
        An iterator (drop-in replacement for a PyTorch DataLoader) that yields
        nested dicts with the processed data.
    '''

    # ===== Input data handling =====

    # @NOTE
    # Goal: Create an input callable which provides one sample at a time to DALI's external source. The
    # provider defines the sample structure and fills that structure with data on request. This example uses a
    # provider that generates synthetic samples so it can run without dataset preparation; in a real use case,
    # the same setup pattern can be used with a provider that loads the actual training data.
    data_provider = ExampleDataProvider(
        num_samples=num_samples,
        image_height=image_height,
        image_width=image_width,
        num_cameras=num_cameras,
        seed=seed,
    )

    # @NOTE
    # The general `ShuffledShardedInputCallable` wraps the provider and adds common input behavior: batching,
    # shuffling, and optional sharding for multi-GPU setups. This keeps the provider focused on describing and
    # returning individual samples.
    input_callable = ShuffledShardedInputCallable(
        data_provider,
        batch_size,
        shuffle=True,
        seed=seed,
    )

    # ===== Define processing steps =====

    # @NOTE
    # The processing steps are defined as separate objects and then composed into a pipeline. This mirrors the
    # larger examples, but uses only a small set of steps to avoid bloating the example with unnecessary complexity.

    # @NOTE
    # Step 1 -- Photometric augmentation for images. `PhotoMetricDistorter` searches for image fields with the
    # configured name and applies random brightness, contrast, saturation, and hue changes.
    #
    # When used directly on the full sample, the same random decisions would be shared across matching fields (i.e. the images of
    # all cameras in the sample). In the next step, we wrap it with an access modifier wrapper to apply it independently per
    # camera.
    photometric_step = PhotoMetricDistorter(
        image_name="image",
        min_max_brightness=[-32.0, 32.0],
        min_max_contrast=[0.5, 1.5],
        min_max_saturation=[0.5, 1.5],
        min_max_hue=[-18.0, 18.0],
        prob_brightness_aug=1.0,
        prob_contrast_aug=1.0,
        prob_saturation_aug=1.0,
        prob_hue_aug=1.0,
        prob_swap_channels=0.0,
        enforce_process_on_gpu=True,
    )

    # @NOTE
    # The access modifier wrapper applies the wrapped step separately to each element of the "cameras" array.
    # Each camera therefore receives its own random augmentation.
    #
    # More generally, access modifiers are useful whenever related fields form a sub-tree that should be
    # processed consistently, while other such sub-trees should be randomized independently. For example, a
    # spatial image transform may need to update an image, its segmentation mask, projected points or bounding
    # boxes, and the corresponding projection matrix with the same sampled transformation. Grouping all fields
    # for one camera into one selected sub-tree keeps these correspondences consistent, while still allowing a
    # different random transformation for each camera. Please see the API documentation for the available access modifier
    # wrappers.
    independent_photometric_per_camera = DataGroupArrayInPathElementsAppliedStep(
        photometric_step,
        path_to_array_to_apply_to="cameras",
    )

    # @NOTE
    # Step 2 -- Normalize all images (i.e. all fields named "image" in the data structure) to the [0, 1] float range.
    image_normalizer = ImageRange01Normalizer("image")

    # @NOTE
    # Step 3 -- Evaluate a condition on the per-camera annotations. For each "annotation" data group, the
    # expression creates a new "is_active" field alongside "label". Keeping `label` in the output makes it easy
    # to compare the original mapped label with the derived flag when inspecting the result.
    condition_eval = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="is_active = label >= 3 and label < 8",
        remove_data_fields_used_in_condition=False,
    )

    # @NOTE
    # Store the processing steps in execution order. This list defines the steps which are performed in the
    # pipeline and their order.
    processing_steps = [
        independent_photometric_per_camera,
        image_normalizer,
        condition_eval,
    ]

    # ===== Pipeline definition =====

    # @NOTE
    # Define the pipeline from the input callable and the processing steps.
    #
    # `check_data_format` controls type/format checks inside the DALI pipeline. It is useful to enable during
    # development, when adding custom steps, or when changing the input data structure, as it catches mistakes
    # such as assigning a field with the wrong DALI type. After the pipeline and data format have been
    # validated, it is usually disabled in production to avoid the related overhead.
    pipeline_def = PipelineDefinition(
        input_callable,
        preprocess_functors=processing_steps,
        check_data_format=False,  # Enable during development/debugging, disable in production.
        print_sample_data_group_format=True,  # This is useful to inspect the data format after each step (for debugging).
        # Disable the DALI external-source pass-through issue copy workaround: callable inputs with
        # batch_size > 1 are outside the affected case. See the API docs for this constructor argument for
        # details about the underlying issue and when a copy is needed.
        copy_external_source_passthrough_outputs=False,
    )

    # @NOTE
    # Infer the output data format blueprint. The resulting `SampleDataGroup` blueprint has the same structure and field types as
    # the pipeline output and is (further down in the code) used to reconstruct nested dictionaries from DALI's flat output
    # sequence.
    output_data_structure = pipeline_def.check_and_get_output_data_structure()

    # @NOTE
    # Create the underlying DALI `Pipeline` object.
    pipe = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,  # <- this needs to be enabled for the pipelines build with this framework.
        batch_size=batch_size,
        num_threads=2,
        device_id=device_id,
        seed=seed,
        py_num_workers=2,
        py_start_method="spawn",
    )

    # @NOTE
    # Build the DALI pipeline.
    pipe.build()

    # ===== Wrap as iterator =====

    # @NOTE
    # Wrap the DALI pipeline so consumers receive nested dictionaries matching the `SampleDataGroup` structure
    # instead of the flat tuple returned by the raw DALI pipeline. The wrapper can be used as a drop-in
    # replacement for a PyTorch DataLoader in training code.
    num_batches = input_callable.length
    result_iterator = DALIStructuredOutputIterator.CreateAsDataLoaderObject(
        num_batches, pipe, output_data_structure
    )

    return result_iterator
