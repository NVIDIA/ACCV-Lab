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

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np
import nvidia.dali.types as types

from accvlab.dali_pipeline_framework.inputs import DataProvider
from accvlab.dali_pipeline_framework.pipeline import SampleDataGroup

# @NOTE
# These mappings demonstrate how a `SampleDataGroup` data field can accept human-readable strings while storing
# numeric IDs in the DALI pipeline. When a mapped field is assigned a string key, the framework converts it to
# the corresponding integer value. This is e.g. useful for class labels (e.g. "vehicle", "pedestrian", "cyclist").
CAMERA_LABEL_TO_ID = {
    "vehicle": 1,
    "pedestrian": 3,
    "cyclist": 6,
    "construction": 8,
}
SCENE_LABEL_TO_ID = {
    "indoor": 0,
    "urban": 1,
    "highway": 2,
    "tunnel": 3,
}
CAMERA_ID_TO_LABEL = {label_id: label for label, label_id in CAMERA_LABEL_TO_ID.items()}
SCENE_ID_TO_LABEL = {label_id: label for label, label_id in SCENE_LABEL_TO_ID.items()}


# @NOTE
# A data provider is responsible for two things:
#   - describing the sample data structure via a `SampleDataGroup` blueprint
#   - returning a filled sample for a requested sample index
class ExampleDataProvider(DataProvider):
    '''Data provider that generates synthetic multi-camera data for demonstration purposes.

    Each sample consists of:
      - An array of cameras, each containing a synthetic image and an annotation
        with a string label that is mapped to an integer id by
        :class:`SampleDataGroup`.
      - A global scene label shared across all cameras in the sample, also
        provided as a string and converted via a user-defined mapping.

    Images are deterministic color gradients so that the output is reproducible
    and visually distinguishable across cameras and samples.
    '''

    def __init__(
        self,
        num_samples: int = 8,
        image_height: int = 64,
        image_width: int = 64,
        num_cameras: int = 2,
        seed: int = 42,
    ):
        '''

        Args:
            num_samples: Number of samples available in this provider.
            image_height: Height of the generated images in pixels.
            image_width: Width of the generated images in pixels.
            num_cameras: Number of cameras per sample.
            seed: Base seed used for deterministic data generation.
        '''

        self._num_samples = num_samples
        self._image_height = image_height
        self._image_width = image_width
        self._num_cameras = num_cameras
        self._seed = seed
        self._camera_label_names = tuple(CAMERA_LABEL_TO_ID)
        self._scene_label_names = tuple(SCENE_LABEL_TO_ID)

        # @NOTE
        # The blueprint contains the field names, nesting, and DALI data types, but no actual sample values.
        # It also carries the optional string-to-ID mappings used when values are assigned to mapped fields.
        # It is cached because it is reused both for reporting the input format and for creating empty samples
        # with the same format in `get_data()`.
        self._blueprint = self._build_blueprint(num_cameras)

    @override
    def get_data(self, sample_index: int) -> SampleDataGroup:
        # @NOTE
        # `DataProvider.get_data()` is called by the input callable to obtain one sample by index. The returned
        # `SampleDataGroup` must match `sample_data_structure` and contain the actual values for that sample.

        # @NOTE
        # Start each sample from an empty copy of the blueprint. This keeps the data format fixed while
        # allowing the actual values to be filled independently for each requested sample.
        sample = self._blueprint.get_empty_like_self()
        rng = np.random.RandomState(self._seed + sample_index)

        # @NOTE
        # The synthetic images are deterministic for a given sample index. This makes the example easy to
        # inspect while still providing different values across samples.
        base_color = rng.randint(80, 256, size=3).astype(np.float32)

        # @NOTE
        # Fill the array field "cameras". Each camera entry is a data group with an image and an annotation
        # subgroup. Here, the label field is mapped to an integer ID on assignment. String fields can also be
        # passed through directly, as shown below with `scene_label_as_str`.
        for cam_idx in range(self._num_cameras):
            image = self._generate_image(base_color, cam_idx)
            sample["cameras"][cam_idx]["image"] = image
            label_name_as_str = self._camera_label_names[rng.randint(len(self._camera_label_names))]
            sample["cameras"][cam_idx]["annotation"]["label"] = label_name_as_str

        # @NOTE
        # The same scene label is stored twice to show the difference between a mapped integer field and a raw
        # string field. `scene_label` is converted via `SCENE_LABEL_TO_ID`, while `scene_label_as_str` is passed
        # through as string data.
        scene_label_name_as_str = self._scene_label_names[rng.randint(len(self._scene_label_names))]
        sample["scene_label"] = scene_label_name_as_str
        sample["scene_label_as_str"] = scene_label_name_as_str

        return sample

    @override
    def get_number_of_samples(self) -> int:
        # @NOTE
        # `DataProvider.get_number_of_samples()` tells the input callable how many samples are available for
        # indexing, shuffling, and epoch-length computation.
        return self._num_samples

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        # @NOTE
        # `DataProvider.sample_data_structure` is the provider's format contract. The pipeline asks for this
        # blueprint before execution to infer the input and output formats. Return an (empty) copy so callers
        # cannot accidentally fill or mutate the cached blueprint.
        return self._blueprint.get_empty_like_self()

    def _generate_image(self, base_color: np.ndarray, cam_idx: int) -> np.ndarray:
        '''Create a synthetic HxWx3 uint8 image.

        Both cameras in a sample share the same ``base_color`` but use
        different gradient directions (horizontal for camera 0, vertical for
        camera 1).  This makes the effect of independent per-camera photometric
        augmentation clearly visible in the output.
        '''
        h, w = self._image_height, self._image_width

        # @NOTE
        # Use different gradient directions for neighboring cameras so that the images are visually distinguishable.
        if cam_idx % 2 == 0:
            gradient = np.linspace(0, 1, w, dtype=np.float32)
            gradient = np.tile(gradient[np.newaxis, :], (h, 1))
        else:
            gradient = np.linspace(0, 1, h, dtype=np.float32)
            gradient = np.tile(gradient[:, np.newaxis], (1, w))

        image = np.stack(
            [
                np.clip(gradient * base_color[0], 0, 255),
                np.clip(gradient * base_color[1], 0, 255),
                np.clip(gradient * base_color[2], 0, 255),
            ],
            axis=-1,
        ).astype(np.uint8)

        return image

    @staticmethod
    def _build_blueprint(num_cameras: int) -> SampleDataGroup:
        '''Construct the hierarchical SampleDataGroup blueprint.

        Structure::

            root
            +-- cameras (array of ``num_cameras`` data group fields)
            |   +-- 0
            |   |   +-- image: UINT8
            |   |   +-- annotation
            |   |       +-- label: INT32, with string-to-id mapping
            |   +-- 1
            |       +-- ...
            +-- scene_label: INT32, with string-to-id mapping
            +-- scene_label_as_str: STRING
        '''
        # @NOTE
        # Build the annotation subgroup first. The `label` field is stored as INT32, but the mapping allows the
        # provider to assign strings such as "vehicle" or "pedestrian" when filling samples.
        annotation = SampleDataGroup()
        annotation.add_data_field(
            "label",
            types.DALIDataType.INT32,
            mapping=CAMERA_LABEL_TO_ID,
        )

        # @NOTE
        # A camera is represented as a data group containing the image and the nested annotation group. The
        # group is later reused as the template for every element of the "cameras" array field.
        cam = SampleDataGroup()
        cam.add_data_field("image", types.DALIDataType.UINT8)
        cam.add_data_group_field("annotation", annotation)

        # @NOTE
        # The root sample contains the multi-camera array and two sample-level scene labels. `scene_label` is a
        # mapped INT32 field; `scene_label_as_str` is a STRING field that shows how string data can pass through
        # the pipeline directly.
        root = SampleDataGroup()
        root.add_data_group_field_array("cameras", cam, num_cameras)
        root.add_data_field(
            "scene_label",
            types.DALIDataType.INT32,
            mapping=SCENE_LABEL_TO_ID,
        )
        root.add_data_field(
            "scene_label_as_str",
            types.DALIDataType.STRING,
        )

        return root
