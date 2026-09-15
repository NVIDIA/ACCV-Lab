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

# Used to enable type hints using a class type inside the implementation of that class itself.
from __future__ import annotations

from abc import ABC, abstractmethod

from enum import Enum

import numpy as np

import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.types as types
import nvidia.dali.data_node as node
from nvidia.dali.pipeline import do_not_convert

from typing import Tuple, Union, Sequence, List, Tuple, Set, Optional

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..pipeline.sample_data_group import SampleDataGroup
from ..operators_impl.python_operator_functions import apply_transform_to_points
from ..operators_impl.numba_operators import apply_matrix
from ..internal_helpers import get_as_data_node

from . import PipelineStepBase


class AffineTransformer(PipelineStepBase):
    '''Apply affine augmentations (translation, scaling, rotation, shearing) to images, and update
    associated geometry (points, projection matrices) consistently.

    This step can process one or multiple images, as well as point sets and projection matrices. It
    expects image data fields and sibling image-size fields in the input (see :class:`SampleDataGroup`).
    Optionally, names of point-set and projection-matrix fields can be provided. Multiple instances may
    be present; all matching occurrences are processed. If multiple images are found, each must have a
    sibling size field, and the sizes must match.

    The same transformation is applied to all matched images. If different images require different
    transformations, create multiple instances of this step and apply them to different sub-trees (see
    :class:`GroupToApplyToSelectedStepBase`).

    Projection geometry represented as intrinsics and extrinsics should be handled by passing only the
    intrinsics matrix to this step; extrinsics are unaffected by an image-plane affine transform.
    Note that apart from true projection matrices, any matrices can be handled which transform points from
    a different coordinate system into the image coordinate system.

    The affine transform conceptually moves image content within a fixed viewport. For example, a
    translation to the right shifts the content rightward and exposes a border on the left. Scaling does
    not change the viewport size (pixel resolution), so upscaling reveals only the center region, while
    downscaling fills only part of the viewport.

    After augmentation, a resize to the requested output resolution is applied if needed. When aspect
    ratios differ, the adjustment is controlled by :class:`AffineTransformer.ResizingMode` and
    :class:`AffineTransformer.ResizingAnchor`. Note that this resizing is independent of the affine
    transformation (where scaling leaves the viewport unchanged), and can be used to change the resolution
    and aspect ratio of the image.

    The overall transform is built as a chain of steps (see :class:`AffineTransformer.TransformationStep`
    and subclasses). :class:`AffineTransformer.Selection` allows probabilistic branching. Some steps that
    depend on alignments cannot follow incompatible steps (e.g., rotation or shearing). These constraints
    are validated at construction, and include incompatible steps anywhere in the chain before the step
    (including potentially applied probabilistic branches).

    All steps that require a reference point (e.g., rotation, scaling) use the viewport center.

    The composed augmentation and resize are combined to a single image resampling step to minimize,
    which is advantageous both for quality of the final image and runtime.

    '''

    class TransformationStep(ABC):
        '''Step used to build up the overall affine transformation to apply. Each step is processed in sequence and with a given probability.

        Probabilistic branching possible by using the ``AffineTransformer.Selection`` (also see documentation for that step).
        '''

        def __init__(self, prob: float):
            '''

            Args:
                prob: Probability with which this step is applied
            '''

            self.prob = prob

        def __call__(self, prior_trafo: Union[node.DataNode, None], image_hw: node.DataNode):
            # Note: This docstring should not be shown, but `:meta private` does not work for private methods.
            # Comment it out, but do not delete (for in-code documentation).
            # '''
            # Update current transformation matrix (corresponding to previous steps) with this step.
            #
            # Args:
            #     prior_trafo: Transforamtion matrix for previous steps or None if this is the first transformation
            #     image_hw: Input image height and width
            #
            # Returns:
            #     Transformation matrix corresponding to previous and current steps
            #
            # :meta private:
            # '''

            if prior_trafo is None:
                trafo = fn.transforms.translation(offset=[0.0, 0.0])
            else:
                trafo = prior_trafo
            # trafo = fn.cast([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=types.)
            draw = self._get_random_in_range(0.0, 1.0)
            if draw < self.prob:
                trafo = self._apply(trafo, image_hw)
            return trafo

        @abstractmethod
        def check_prev_types_compatible_and_add_current_type(self, prev_types: Set[type]) -> Set[type]:
            '''Validate that no incompatible prior steps exist and record this step's type.

            If a step itself aggregates other steps (e.g., :class:`AffineTransformer.Selection`), the
            types of all potentially applied steps must be included.

            Args:
                prev_types: Types of previously (potentially) applied steps.

            Returns:
                Types of potentially taken steps up to and including this step.

            :meta private:
            '''
            pass

        @abstractmethod
        def _apply(self, prior_trafo: node.DataNode, image_hw: node.DataNode):
            '''Apply this step to update the current transformation matrix.

            Note:
                The :meth:`AffineTransformer.TransformationStep.__call__` method initializes the matrix (if
                needed) and handles probabilistic execution, so here
                  - ``prior_trafo`` is always provided
                  - the step should be executed without any further checks regarding the execution probability

            '''
            pass

        @staticmethod
        def _get_random_in_range(min, max):
            if min == max:
                res = min
            else:
                min = fn.cast(min, dtype=types.DALIDataType.FLOAT)
                max = fn.cast(max, dtype=types.DALIDataType.FLOAT)
                res = fn.random.uniform(range=fn.stack(min, max))
            return res

        @staticmethod
        def _get_center_xy(image_hw):
            res = fn.stack(image_hw[1] * 0.5, image_hw[0] * 0.5)
            return res

    class Translation(TransformationStep):
        '''Perform a randomized translation (in a given range).'''

        def __init__(self, prob: float, min_xy: Sequence[float], max_xy: Union[Sequence[float], None] = None):
            '''

            Args:
                prob: Probability to apply step.
                min_xy: Minimum shift in x and y. If ``max_xy`` is not set, a shift of exactly ``min_xy`` is
                    performed instead of selecting at random from a range.
                max_xy: Maximum shift in x and y.
            '''
            super().__init__(prob)
            self.min_xy = min_xy
            self.max_xy = max_xy

        def _apply(self, prior_trafo, image_hw):
            if self.max_xy is None:
                transformation = fn.transforms.translation(prior_trafo, offset=self.min_xy)
            else:
                translation_x = self._get_random_in_range(self.min_xy[0], self.max_xy[0])
                translation_y = self._get_random_in_range(self.min_xy[1], self.max_xy[1])
                transformation = fn.transforms.translation(
                    prior_trafo, offset=fn.stack(translation_x, translation_y)
                )
            return transformation

        def check_prev_types_compatible_and_add_current_type(self, prev_types: Set[type]) -> Set[type]:
            res = prev_types.copy()
            res.add(self.__class__)
            return res

    class ShiftInsideOriginalImage(TransformationStep):
        '''Perform a random translation. The shift is selected so that the viewport is filled with the image.

        This is only possible if the image is larger (i.e. previously scaled up) or equal to the viewport.
        If this is not the case, this step does nothing.

        The shift is computed and performed independently for x- and y-directions. This means that if the
        image is larger than the viewport in one dimension and smaller in the other one (e.g. due to
        non-uniform scaling), this step will be performed in the dimension where the image is larger than
        the viewport.

        Also, if the image is larger than the viewport, this step will bring back the image
        to cover the whole viewport if it was previously moved out of it.

        This step cannot be performed if a rotation and/or shearing was potentially performed before.

        Args:
            prob: Probability to apply step.
            shift_x: Whether to apply in x-direction.
            shift_y: Whether to apply in y-direction.

        '''

        def __init__(self, prob: float, shift_x: bool, shift_y: bool):
            '''

            Args:
                prob: Probability to apply step.
                shift_x: Whether to apply in x-direction.
                shift_y: Whether to apply in y-direction.
            '''
            super().__init__(prob)
            self.shift_x = shift_x
            self.shift_y = shift_y

        def _apply(self, prior_trafo, image_hw):
            @do_not_convert
            def get_min_max_shifts(prior_trafo, image_hw):
                upper_left_orig = (prior_trafo @ np.array([0.0, 0.0, 1.0]))[0:2]
                lower_right_orig = (prior_trafo @ np.array([image_hw[1], image_hw[0], 1.0]))[0:2]
                min_shift = np.zeros(2, dtype=np.float)
                max_shift = np.zeros(2, dtype=np.float)

                # For each dimension, check which point is the lower and which the higher cordinate (may be flipped)
                min_coords = np.zeros(2, dtype=np.float)
                max_coords = np.zeros(2, dtype=np.float)
                for d in range(2):
                    if upper_left_orig[d] < lower_right_orig[d]:
                        min_coords[d] = upper_left_orig[d]
                        max_coords[d] = lower_right_orig[d]
                    else:
                        min_coords[d] = lower_right_orig[d]
                        max_coords[d] = upper_left_orig[d]

                    min_shift[d] = -min_coords[d]
                    max_shift[d] = image_hw[1 - d] - max_coords[d]

                    if min_shift[d] > max_shift[d]:
                        temp = min_shift[d]
                        min_shift[d] = max_shift[d]
                        max_shift[d] = temp

                return min_shift, max_shift

            min_shift, max_shift = fn.python_function(
                prior_trafo, image_hw, function=get_min_max_shifts, num_outputs=2
            )

            if self.shift_x and min_shift[0] < max_shift[0]:
                x_shift = self._get_random_in_range(min_shift[0], max_shift[0])
            else:
                x_shift = 0.0
            if self.shift_y and min_shift[1] < max_shift[1]:
                y_shift = self._get_random_in_range(min_shift[1], max_shift[1])
            else:
                y_shift = 0.0
            transformation = fn.transforms.translation(prior_trafo, offset=fn.stack(x_shift, y_shift))
            return transformation

        def check_prev_types_compatible_and_add_current_type(self, prev_types: Set[type]) -> Set[type]:
            if AffineTransformer.Rotation in prev_types or AffineTransformer.Shearing in prev_types:
                raise ValueError(
                    "Cannot perform `ShiftInsideOriginalImage` if rotation or shearing are (potentially) performed before."
                )
            res = prev_types.copy()
            res.add(self.__class__)
            return res

    class ShiftToAlignWithOriginalImageBorder(TransformationStep):
        '''Translate the image so that it is aligned to a border of the viewport.

        The border to align to can be selected on construction.

        This step cannot be performed if a rotation and/or shearing was potentially performed before.

        '''

        class Border(Enum):
            '''Enumeration for viewport borders to align to'''

            TOP = 0
            LEFT = 1
            BOTTOM = 2
            RIGHT = 3

        def __init__(self, prob: float, border: AffineTransformer.ShiftToAlignWithOriginalImageBorder):
            '''

            Args:
                prob: Probability to perform step.
                border: Border of the viewport to align image to.

            '''

            super().__init__(prob)
            self._border = border

        def _apply(self, prior_trafo, image_hw):
            @do_not_convert
            def get_min_max_coords(prior_trafo, image_hw):
                upper_left_orig = prior_trafo @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
                lower_right_orig = prior_trafo @ np.array([image_hw[1], image_hw[0], 1.0], dtype=np.float32)
                # For each dimension, check which point is the lower and which the higher cordinate (may be flipped)
                min_coords = np.zeros(2, dtype=np.float32)
                max_coords = np.zeros(2, dtype=np.float32)
                for d in range(2):
                    if upper_left_orig[d] < lower_right_orig[d]:
                        min_coords[d] = upper_left_orig[d]
                        max_coords[d] = lower_right_orig[d]
                    else:
                        min_coords[d] = lower_right_orig[d]
                        max_coords[d] = upper_left_orig[d]

                return min_coords, max_coords

            min_coords, max_coords = fn.python_function(
                prior_trafo, image_hw, function=get_min_max_coords, num_outputs=2
            )

            if self._border == self.Border.TOP:
                translation = fn.stack(0.0, -min_coords[1])
            elif self._border == self.Border.LEFT:
                translation = fn.stack(-min_coords[0], 0.0)
            elif self._border == self.Border.BOTTOM:
                translation = fn.stack(0.0, image_hw[0] - max_coords[1])
            elif self._border == self.Border.RIGHT:
                translation = fn.stack(image_hw[1] - max_coords[0], 0.0)
            else:
                raise NotImplementedError(f"Border type {self._border} not supported")
            transformation = fn.transforms.translation(prior_trafo, offset=translation)
            return transformation

        def check_prev_types_compatible_and_add_current_type(self, prev_types: Set[type]) -> Set[type]:
            if AffineTransformer.Rotation in prev_types or AffineTransformer.Shearing in prev_types:
                raise ValueError(
                    "Cannot perform `ShiftToAlignWithOriginalImageBorder` if rotation or shearing are (potentially) performed before."
                )
            res = prev_types.copy()
            res.add(self.__class__)
            return res

    class Rotation(TransformationStep):
        '''Perform a rotation.'''

        def __init__(self, prob: float, min_rot: float, max_rot: Optional[float] = None):
            '''

            Args:
                prob: Probability to perform step.
                min_rot: Minimum rotation to perform. If ``max_rot`` is not set, this rotation is performed
                    instead of selecting a rotation value randomly from the range.
                max_rot: Maximum rotation to perform.

            '''

            super().__init__(prob)
            self.prob = prob
            self.min_rot = min_rot
            self.max_rot = max_rot

        def _apply(self, prior_trafo, image_hw):
            center = self._get_center_xy(image_hw)
            # Note that in both of the following cases, the angle is negated to ensure that positive angles
            # correspond to anti-clockwise rotation in the image. Due to the coordinate system used for
            # images, rotation with positive angle will rotate the image clockwise (due to y pointing down).
            # This is not the common convention when rotating images. To ensure that positive angles
            # correspond to anti-clockwise rotation, the angle is negated. This is also done for the case
            # of random angles to ensure that the minimum and maximum angles are always in the expected
            # direction.
            if self.max_rot is None:
                angle = -self.min_rot
                transformation = fn.transforms.rotation(prior_trafo, angle=angle, center=center)
            else:
                angle = -self._get_random_in_range(self.min_rot, self.max_rot)
                transformation = fn.transforms.rotation(prior_trafo, angle=angle, center=center)
            return transformation

        def check_prev_types_compatible_and_add_current_type(self, prev_types: Set[type]) -> Set[type]:
            res = prev_types.copy()
            res.add(self.__class__)
            return res

    class UniformScaling(TransformationStep):
        '''Perform uniform scaling (i.e. identical scaling factor in both x- and y-dimensions).'''

        def __init__(self, prob: float, min_scaling: float, max_scaling: Optional[float] = None):
            '''

            Args:
                prob: Probability to perform step.
                min_scaling: Minimum scaling factor. If ``max_scaling`` is not set, this factor is always
                    applied instead of selecting a random factor from the range.
                max_scaling: Maximum scaling factor.
            '''

            super().__init__(prob)
            self.min_scaling = min_scaling
            self.max_scaling = max_scaling

        def _apply(self, prior_trafo, image_hw):
            center = self._get_center_xy(image_hw)
            if self.max_scaling is None:
                transformation = fn.transforms.scale(
                    prior_trafo, scale=[self.min_scaling, self.min_scaling], center=center
                )
            else:
                scale = self._get_random_in_range(self.min_scaling, self.max_scaling)
                transformation = fn.transforms.scale(prior_trafo, scale=fn.stack(scale, scale), center=center)
            return transformation

        def check_prev_types_compatible_and_add_current_type(self, prev_types: Set[type]) -> Set[type]:
            res = prev_types.copy()
            res.add(self.__class__)
            return res

    class NonUniformScaling(TransformationStep):
        '''Perform non-uniform scaling (i.e. scaling factors in x- and y-dimensions are independent).'''

        def __init__(
            self,
            prob: float,
            min_scaling_xy: Sequence[float],
            max_scaling_xy: Optional[Sequence[float]] = None,
        ):
            '''

            Args:
                prob: Probability to perform step.
                min_scaling_xy: Minimum scaling factors for x- and y-dimensions. If ``max_scaling_xy`` is not
                    set, these factors are always applied instead of selecting random factors from the range.
                max_scaling_xy: Maximum scaling factors for x- and y-dimensions.
            '''

            super().__init__(prob)
            self.min_scaling_xy = min_scaling_xy
            self.max_scaling_xy = max_scaling_xy

        def _apply(self, prior_trafo, image_hw):
            center = self._get_center_xy(image_hw)
            if self.max_scaling_xy is None:
                transformation = fn.transforms.scale(scale=self.min_scaling_xy, center=center)
            else:
                scale_x = self._get_random_in_range(self.min_scaling_xy[0], self.max_scaling_xy[0])
                scale_y = self._get_random_in_range(self.min_scaling_xy[1], self.max_scaling_xy[1])
                transformation = fn.transforms.scale(
                    prior_trafo, scale=fn.stack(scale_x, scale_y), center=center
                )
            return transformation

        def check_prev_types_compatible_and_add_current_type(self, prev_types: Set[type]) -> Set[type]:
            res = prev_types.copy()
            res.add(self.__class__)
            return res

    class Shearing(TransformationStep):
        '''Perform shearing.'''

        def __init__(
            self,
            prob: float,
            min_shearing_xy: Sequence[float],
            max_shearing_xy: Optional[Sequence[float]] = None,
        ):
            '''

            Args:
                prob: Probability to perform step.
                min_shearing_xy: Minimum shearing parameters for x- and y-dimensions. If ``max_shearing_xy``
                    is not set, these parameters are always applied instead of selecting random parameters
                    from the range.
                max_shearing_xy: Maximum shearing parameters.
            '''

            super().__init__(prob)
            self.min_shearing_xy = min_shearing_xy
            self.max_shearing_xy = max_shearing_xy

        def _apply(self, prior_trafo, image_hw):
            center = self._get_center_xy(image_hw)
            if self.max_shearing_xy is None:
                transformation = fn.transforms.shear(prior_trafo, angles=self.min_shearing_xy, center=center)
            else:
                shear_x = self._get_random_in_range(self.min_shearing_xy[0], self.max_shearing_xy[0])
                shear_y = self._get_random_in_range(self.min_shearing_xy[1], self.max_shearing_xy[1])
                transformation = fn.transforms.shear(
                    prior_trafo, angles=fn.stack(shear_x, shear_y), center=center
                )
            return transformation

        def check_prev_types_compatible_and_add_current_type(self, prev_types: Set[type]) -> Set[type]:
            res = prev_types.copy()
            res.add(self.__class__)
            return res

    class Selection(TransformationStep):
        '''Probabilistically choose one sequence of steps out of multiple alternatives and perform the steps in this sequence.'''

        __eps = 1e-6

        def __init__(
            self,
            prob: float,
            option_probs: Sequence[float],
            options: Sequence[
                Union[
                    List[AffineTransformer.TransformationStep],
                    Tuple[AffineTransformer.TransformationStep, ...],
                    AffineTransformer.TransformationStep,
                ]
            ],
        ):
            '''

            Args:
                prob: Probability to perform this step.
                option_probs: Probabilities for the individual options. Has to sum up to 1 as one option is
                    always taken.
                options: The individual options. Each option is a sequence of transformation steps or a single
                    step.

            '''

            super().__init__(prob)

            num_options = len(option_probs)
            assert (
                len(options) == num_options
            ), "Number of per-option probabilities and options does not match"

            self._options = [o if not isinstance(o, self.__class__.__bases__[0]) else [o] for o in options]

            self._options_accum_prob = [0] * num_options
            self._options_accum_prob[0] = option_probs[0]
            for i in range(1, num_options):
                self._options_accum_prob[i] = self._options_accum_prob[i - 1] + option_probs[i]
            assert (
                abs(self._options_accum_prob[-1] - 1.0) <= self.__eps
            ), "Probabilities for options do not sum up to 1"

        def _apply(self, prior_trafo, image_size):
            draw = self._get_random_in_range(0.0, 1.0)

            already_set = False
            res = prior_trafo
            for i in range(len(self._options_accum_prob)):
                if not already_set and draw <= self._options_accum_prob[i]:
                    res = self._apply_option(prior_trafo, image_size, self._options[i])
                    already_set = True

            return res

        def _apply_option(
            self, prior_trafo, image_size, option_steps: Sequence[AffineTransformer.TransformationStep]
        ):
            res = prior_trafo
            for s in option_steps:
                res = s(res, image_size)
            return res

        def check_prev_types_compatible_and_add_current_type(self, prev_types: Set[type]) -> Set[type]:
            per_option_types = []
            for option in self._options:
                option_types = prev_types
                for el in option:
                    option_types = el.check_prev_types_compatible_and_add_current_type(option_types)
                per_option_types.append(option_types)

            res = prev_types
            for ot in per_option_types:
                res = res.union(ot)

            return res

    class ResizingMode(Enum):
        '''Resizing mode types.

        The mode defines how the input viewport is adjusted to the output viewport when the output image shape has not the same aspect ratio as the input image shape.

        Note that as the image may be outside the input viewport due to affine transformations, it may e.g. happen that there is still image data in the padded region of the output viewport. In this case, the image will appear in the padded
        region and will not be replaced by the fill value.
        '''

        #: Viewport is extended to preserve aspect ratio (i.e. if there are no other transformations,
        #: the output image will be padded).
        STRETCH = 0
        #: Viewport is stretched (i.e. image is non-uniformly scaled).
        PAD = 1
        #: Viewport is cropped (i.e. if there are no other transformations, parts of the input image will be
        #: cropped away).
        CROP = 2

    class ResizingAnchor(Enum):
        '''Resizing mode anchor.

        The anchor defines which reference point in the output image is aligned to the corresponding point in
        the input image when adjusting the aspect ratio to match the output image using the PAD or CROP
        resizing mode.

        Important:
            Note that the anchor is only relevant when changing the aspect ratio of the image.
            The actual transformations such as scaling, rotation, etc. are not affected by the anchor,
            and always use the center of the image as reference point.
        '''

        #: The center of the output image corresponds to the center of the input image
        CENTER = 0
        #: The top left corner of the output image corresponds to the top left corner of the input image.
        #: Depending on which direction is padded / cropped, this corresponds to either keeping the top or
        #: the left border aligned.
        TOP_OR_LEFT = 1
        #: The bottom right corner of the output image corresponds to the bottom left corner of the input
        #: image. Depending on which direction is padded / cropped, this corresponds to either keeping the
        #: bottom or the right border aligned.
        BOTTOM_OR_RIGHT = 2

    def __init__(
        self,
        output_hw: Sequence[int],
        resizing_mode: AffineTransformer.ResizingMode,
        resizing_anchor: Optional[AffineTransformer.ResizingAnchor] = None,
        image_field_names: Optional[
            Union[str, int, List[Union[str, int]], Tuple[Union[str, int], ...]]
        ] = None,
        image_hw_field_names: Optional[
            Union[str, int, List[Union[str, int]], Tuple[Union[str, int], ...]]
        ] = None,
        projection_matrix_field_names: Optional[
            Union[str, int, List[Union[str, int]], Tuple[Union[str, int], ...]]
        ] = None,
        point_field_names: Optional[
            Union[str, int, List[Union[str, int]], Tuple[Union[str, int], ...]]
        ] = None,
        transformation_steps: Optional[Sequence[AffineTransformer.TransformationStep]] = None,
        transform_image_on_gpu: bool = True,
    ):
        '''

        Args:
            output_hw: Output resolution ``[height, width]``. The input image is resized to this size.
            resizing_mode: How to resolve aspect-ratio differences. See
                :class:`AffineTransformer.ResizingMode`.
            resizing_anchor: Anchor to use when ``resizing_mode`` is not ``STRETCH``. See
                :class:`AffineTransformer.ResizingAnchor`. Must be ``None`` when ``resizing_mode`` is
                ``STRETCH`` and set otherwise.
            image_field_names: Names of image fields to transform (see :class:`SampleDataGroup`). Set to
                ``None`` to not process images (e.g., only projection matrices or point sets). Cannot be
                set if ``image_hw_field_names`` is set.
            image_hw_field_names: Names of the fields containing image size ``[height, width]``. All listed
                fields must have identical values. If not, call this step separately per image (e.g., by
                name or by selecting a sub-tree, see :class:`GroupToApplyToSelectedStepBase`). Cannot be
                set if ``image_field_names`` is set. One of ``image_field_names`` or ``image_hw_field_names``
                must be provided (single source of truth for image size).
            projection_matrix_field_names: Names of fields with projection matrices that map to pixel
                coordinates. These matrices are updated to project correctly in the output image. Set to
                ``None`` to skip. If projection geometry is represented by extrinsics and intrinsics, only
                pass the intrinsics here; extrinsics are unaffected by an image-plane affine transform.
                Note that apart from true projection matrices, any matrices can be handled which transform
                points from a different coordinate system into the image coordinate system.
            point_field_names: Names of fields containing 2D point sets (e.g., landmarks). Points are
                transformed to remain consistent with the output images. Points are expected as rows; A row
                may contain multiple points, in which case consecutive pairs are treated as individual points
                and stored in the same format (e.g. ``[x1, y1, x2, y2]``).
            transformation_steps: Sequence of steps to perform. If ``None``,
                only resizing to the output resolution & handling of changed aspect ratio is performed
                (no augmentation).
            transform_image_on_gpu: Whether to transform images on the GPU. Must be ``True`` if images are
                already on GPU. Default: ``True``.
        '''

        # Ensure exactly one of image_field_names or image_hw_field_names is set (single source of truth)
        if (
            image_field_names is None or (isinstance(image_field_names, list) and len(image_field_names) == 0)
        ) and (
            image_hw_field_names is None
            or (isinstance(image_hw_field_names, list) and len(image_hw_field_names) == 0)
        ):
            raise ValueError(
                "Either 'image_field_names' or 'image_hw_field_names' must be provided (but not both) to determine image size."
            )
        if (
            image_field_names is not None
            and (not isinstance(image_field_names, list) or len(image_field_names) > 0)
        ) and (
            image_hw_field_names is not None
            and (not isinstance(image_hw_field_names, list) or len(image_hw_field_names) > 0)
        ):
            raise ValueError(
                "Only one of 'image_field_names' or 'image_hw_field_names' can be set (single source of truth for image size)."
            )

        if isinstance(image_field_names, str) or isinstance(image_field_names, int):
            image_field_names = [image_field_names]
        self._image_field_names = image_field_names

        # Flag to determine if we extract size from images or use size fields
        self._extract_size_from_images = image_field_names is not None and len(image_field_names) > 0

        if isinstance(image_hw_field_names, str) or isinstance(image_hw_field_names, int):
            image_hw_field_names = [image_hw_field_names]
        self._image_hw_field_names = image_hw_field_names

        if isinstance(projection_matrix_field_names, str) or isinstance(projection_matrix_field_names, int):
            projection_matrix_field_names = [projection_matrix_field_names]
        self._projection_matrix_field_names = projection_matrix_field_names

        if isinstance(point_field_names, str) or isinstance(point_field_names, int):
            point_field_names = [point_field_names]
        self._point_field_names = point_field_names

        if transformation_steps is not None:
            prev_steps = set()
            for tf in transformation_steps:
                prev_steps = tf.check_prev_types_compatible_and_add_current_type(prev_steps)

        self._transformation_steps = transformation_steps

        self._output_hw = output_hw
        self._transform_image_on_gpu = transform_image_on_gpu

        self._resizing_mode = resizing_mode
        self._resizing_anchor = resizing_anchor
        if resizing_mode == self.ResizingMode.STRETCH and resizing_anchor is not None:
            raise ValueError("When using STRETCH resizing mode, `resizing_anchor` has to be set to `None`.")
        if resizing_mode != self.ResizingMode.STRETCH and resizing_anchor is None:
            raise ValueError("When not using STRETCH resizing mode, a `resizing_anchor` has to be selected.")

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        @do_not_convert
        def raise_exception_size_different(size_0, size_1):
            raise ValueError(
                f"Defined sizes of images do not match. Example unmatched sizes: {size_0[0]}, {size_0[1]} | {size_1[0]}, {size_1[1]}"
            )

        image_hw = [0, 0]
        is_image_hw_set = False

        if self._extract_size_from_images:
            # Extract size from images using .shape
            for image_field_name in self._image_field_names:
                image_paths = data.find_all_occurrences(image_field_name)
                if len(image_paths) > 0:
                    start_index_paths = 0
                    if not is_image_hw_set:
                        image = data.get_item_in_path(image_paths[0])
                        image_shape = image.shape()
                        # Use fn.stack to create a proper tensor with [height, width]
                        # Cast to int32 for consistency with image_hw fields
                        image_hw = fn.cast(
                            fn.stack(image_shape[-3], image_shape[-2]), dtype=types.DALIDataType.INT32
                        )
                        is_image_hw_set = True
                        start_index_paths = 1
                    for ip in image_paths[start_index_paths:]:
                        image = data.get_item_in_path(ip)
                        image_shape = image.shape()
                        image_hw_i = fn.cast(
                            fn.stack(image_shape[-3], image_shape[-2]), dtype=types.DALIDataType.INT32
                        )
                        if image_hw_i[0] != image_hw[0] or image_hw_i[1] != image_hw[1]:
                            fn.python_function(image_hw, image_hw_i, function=raise_exception_size_different)
        else:
            # Use size fields
            for image_hw_field_name in self._image_hw_field_names:
                image_hw_paths = data.find_all_occurrences(image_hw_field_name)
                if len(image_hw_paths) > 0:
                    start_index_paths = 0
                    if not is_image_hw_set:
                        image_hw = data.get_item_in_path(image_hw_paths[0])
                        is_image_hw_set = True
                        start_index_paths = 1
                    for ip in image_hw_paths[start_index_paths:]:
                        image_hw_i = data.get_item_in_path(ip)
                        if image_hw_i[0] != image_hw[0] or image_hw_i[1] != image_hw[1]:
                            fn.python_function(image_hw, image_hw_i, function=raise_exception_size_different)

        transform = self._get_transformation(
            image_hw, self._transformation_steps, self._output_hw, self._resizing_mode, self._resizing_anchor
        )

        if self._image_field_names is not None:
            for image_field_name in self._image_field_names:
                image_paths = data.find_all_occurrences(image_field_name)
                for ip in image_paths:
                    parent = data.get_parent_of_path(ip)
                    image = parent[image_field_name]
                    image = self._affine_transform_image(transform, image)
                    parent[image_field_name] = image

        self._projection_matrix_field_names
        if self._projection_matrix_field_names is not None:
            for projection_matrix_field_name in self._projection_matrix_field_names:
                projection_matrix_paths = data.find_all_occurrences(projection_matrix_field_name)
                for pmp in projection_matrix_paths:
                    parent = data.get_parent_of_path(pmp)
                    matrix = parent[projection_matrix_field_name]
                    matrix = self._apply_affine_post_transform_to_matrix(transform, matrix)
                    parent[projection_matrix_field_name] = matrix

        if self._point_field_names is not None:
            for point_field_name in self._point_field_names:
                point_paths = data.find_all_occurrences(point_field_name)
                for pp in point_paths:
                    parent = data.get_parent_of_path(pp)
                    points = parent[point_field_name]
                    points = self._affine_transform_points(transform, points)
                    parent[point_field_name] = points

        if not self._extract_size_from_images:
            # When using size fields, update them as specified
            if self._image_hw_field_names is not None:
                for image_hw_field_name in self._image_hw_field_names:
                    image_hw_paths = data.find_all_occurrences(image_hw_field_name)
                    for sp in image_hw_paths:
                        parent = data.get_parent_of_path(sp)
                        parent[image_hw_field_name] = self._output_hw

        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        if self._extract_size_from_images:
            # When extracting sizes from images, check that images exist
            for image_name in self._image_field_names:
                image_paths = data_empty.find_all_occurrences(image_name)
                if len(image_paths) == 0:
                    raise KeyError(
                        f"No occurrences of images with name `{image_name}` found (the name / one of the names specified in the constructor)."
                    )
        else:
            # When using size fields, check that size fields exist
            for image_hw_name in self._image_hw_field_names:
                image_hw_paths = data_empty.find_all_occurrences(image_hw_name)
                if len(image_hw_paths) == 0:
                    raise KeyError(
                        f"No occurrences of image sizes with name '{image_hw_name}' found (the name / one of the names specified in the constructor)."
                    )

            # Also check for image fields if they are to be transformed
            if self._image_field_names is not None and len(self._image_field_names) > 0:
                for image_name in self._image_field_names:
                    image_paths = data_empty.find_all_occurrences(image_name)
                    if len(image_paths) == 0:
                        raise KeyError(
                            f"No occurrences of images with name `{image_name}` found (the name / one of the names specified in the constructor)."
                        )

        if self._projection_matrix_field_names is not None and len(self._projection_matrix_field_names) > 0:
            for projection_matrix_field_name in self._projection_matrix_field_names:
                projection_matrix_field_paths = data_empty.find_all_occurrences(projection_matrix_field_name)
                if len(projection_matrix_field_paths) == 0:
                    raise KeyError(
                        f"No occurrences of projection matrices with name `{projection_matrix_field_name}` found (the name / one of the names specified in the constructor)."
                    )

        if self._point_field_names is not None and len(self._point_field_names) > 0:
            for point_field_name in self._point_field_names:
                point_field_paths = data_empty.find_all_occurrences(point_field_name)
                if len(point_field_paths) == 0:
                    raise KeyError(
                        f"No occurrences of point sets with name `{point_field_name}` found (the name / one of the names specified in the constructor)."
                    )

        return data_empty

    def _get_transformation(self, image_hw, transformation_steps, output_hw, resizing_mode, resizing_anchor):
        image_resize = self._get_transformation_to_output_size(
            image_hw, output_hw, resizing_mode, resizing_anchor
        )
        if transformation_steps is not None:
            augmentation = self._get_augmentation_transformation(image_hw, transformation_steps)
            transformation = fn.transforms.combine(augmentation, image_resize)
        else:
            transformation = image_resize
        return transformation

    @staticmethod
    def _get_augmentation_transformation(image_hw, transformation_steps):
        transformation = None
        for t in transformation_steps:
            transformation = t(transformation, image_hw)

        if transformation is None:
            # Identity transform
            transformation = fn.transforms.translation(offset=[0.0, 0.0])

        return transformation

    @classmethod
    def _get_transformation_to_output_size(cls, input_hw, output_hw, resizing_mode, resizing_anchor):
        if resizing_mode == cls.ResizingMode.STRETCH:
            # If stretching is used, scale both dimensions of the image to fit the output size
            trafo_resolution = fn.transforms.scale(
                scale=fn.stack(output_hw[1] / input_hw[1], output_hw[0] / input_hw[0])
            )
        elif resizing_mode in [cls.ResizingMode.PAD, cls.ResizingMode.CROP]:
            # Otherwise, perform the following:
            # 1. Scaling to ensure input image to ensure that ...
            if resizing_mode == cls.ResizingMode.PAD:
                # ... scaled image completely fits inside ouput if padding is used
                scale_output_input = math.min(output_hw[0] / input_hw[0], output_hw[1] / input_hw[1])
            elif resizing_mode == cls.ResizingMode.CROP:
                # ... scaled image fills in the complete output image (and parts are cropped if needed) if cropping is used
                scale_output_input = math.max(output_hw[0] / input_hw[0], output_hw[1] / input_hw[1])
            else:
                assert False, "Unknown resizing mode"
            scale_mat_resolution = fn.transforms.scale(scale=fn.stack(scale_output_input, scale_output_input))

            # 2. Position the image according to the anchor
            if resizing_anchor == cls.ResizingAnchor.TOP_OR_LEFT:
                # No shift; represent as an affine translation transform with zero offset
                shift_output_input_mat = fn.transforms.translation(offset=fn.stack(0.0, 0.0))
            elif resizing_anchor in [cls.ResizingAnchor.CENTER, cls.ResizingAnchor.BOTTOM_OR_RIGHT]:
                scale = 0.5 if resizing_anchor == cls.ResizingAnchor.CENTER else 1.0
                point_orig_in_scaled_x = scale_output_input * input_hw[1] * scale
                point_orig_in_scaled_y = scale_output_input * input_hw[0] * scale
                point_scaled_x = output_hw[1] * scale
                point_scaled_y = output_hw[0] * scale
                shift_x = point_scaled_x - point_orig_in_scaled_x
                shift_y = point_scaled_y - point_orig_in_scaled_y
                shift_output_input_mat = fn.transforms.translation(offset=fn.stack(shift_x, shift_y))
            else:
                raise ValueError(f"Resizing anchor {resizing_anchor} not supported.")

            # 3. Get the final transformation as the scaling (1.) followed by centering (2.)
            trafo_resolution = fn.transforms.combine(scale_mat_resolution, shift_output_input_mat)
        else:
            raise ValueError(f"Resizing mode {resizing_mode} not supported.")
        return trafo_resolution

    def _affine_transform_image(self, transform: node.DataNode, image: node.DataNode):
        if self._transform_image_on_gpu:
            transformed_image = fn.warp_affine(
                image.gpu(),
                transform.gpu(),
                size=self._output_hw,
                interp_type=types.INTERP_LINEAR,
                fill_value=0.0,
                inverse_map=False,
            )
        else:
            transformed_image = fn.warp_affine(
                image,
                transform,
                size=self._output_hw,
                interp_type=types.INTERP_LINEAR,
                fill_value=0.0,
                inverse_map=False,
            )
        return transformed_image

    def _apply_affine_post_transform_to_matrix(self, transform: node.DataNode, proj_mat: node.DataNode):
        last_row = get_as_data_node([[0.0, 0.0, 1.0]])
        transform_to_use = fn.cat(transform, last_row, axis=0)
        proj_mat_out = apply_matrix(proj_mat, transform_to_use, False, False, False, False, False)
        return proj_mat_out

    def _affine_transform_points(self, transform: node.DataNode, points: node.DataNode):
        is_on_gpu = points.device == "gpu"
        if is_on_gpu:
            points = fn.python_function(points.gpu(), transform.gpu(), function=apply_transform_to_points)
        else:
            points = fn.python_function(points, transform, function=apply_transform_to_points)

        # Return resulting images, as well as the annotation (with modified bounding boxes) and the new image size
        return points
