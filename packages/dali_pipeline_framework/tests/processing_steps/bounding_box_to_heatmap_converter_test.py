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

import pytest

from typing import Tuple, Optional

import numpy as np
import torch

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import BoundingBoxToHeatmapConverter


def _get_bbox_min_max(bbox: Tuple[int, int, int, int]):
    min_x = min(bbox[0], bbox[2])
    min_y = min(bbox[1], bbox[3])
    max_x = max(bbox[0], bbox[2])
    max_y = max(bbox[1], bbox[3])
    return min_x, min_y, max_x, max_y


def _get_bboxes_canonical(bboxes: np.ndarray):
    for bbox in bboxes:
        min_x, min_y, max_x, max_y = _get_bbox_min_max(bbox)
        bbox[0] = min_x
        bbox[1] = min_y
        bbox[2] = max_x
        bbox[3] = max_y
    return bboxes


def _get_centers_from_bboxes(bboxes: np.ndarray):
    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0
    return centers


def _check_center_positions(
    centers: np.ndarray,
    center_offsets: np.ndarray,
    ref_centers: np.ndarray,
    orig_image_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
):

    scaling_x = heatmap_size[1] / orig_image_size[1]
    scaling_y = heatmap_size[0] / orig_image_size[0]

    ref_centers_scaled = ref_centers * np.array([scaling_x, scaling_y])
    ref_centers_scaled_int = np.floor(ref_centers_scaled).astype(np.int32)
    ref_center_offsets = ref_centers_scaled - ref_centers_scaled_int

    for i in range(len(centers)):
        ref = ref_centers_scaled_int[i]
        if ref[0] < 0:
            ref[0] = 0
            ref_centers_scaled_int[i, 0] = 0.0
            ref_center_offsets[i] = 0.0
        elif ref[0] >= heatmap_size[1]:
            ref[0] = heatmap_size[1] - 1
            ref_centers_scaled_int[i, 0] = heatmap_size[1] - 1
            ref_center_offsets[i] = 0.0
        if ref[1] < 0:
            ref[1] = 0
            ref_centers_scaled_int[i, 1] = 0.0
            ref_center_offsets[i] = 0.0
        elif ref[1] >= heatmap_size[0]:
            ref[1] = heatmap_size[0] - 1
            ref_centers_scaled_int[i, 1] = heatmap_size[0] - 1
            ref_center_offsets[i] = 0.0

    # Check that the centers are the same as the reference centers
    res = np.allclose(centers, ref_centers_scaled_int) and np.allclose(center_offsets, ref_center_offsets)
    return res


def _check_gaussian_centered_at_center(
    heatmap: np.ndarray, center: Tuple[int, int], bbox: Tuple[int, int, int, int]
):
    # Check that the gaussian is centered at the given center and has the given radius
    def is_maximum_at_center(heatmap: np.ndarray, center: Tuple[int, int], bbox: Tuple[int, int, int, int]):
        bbox = _get_bbox_min_max(bbox)
        heatmap_bbox = heatmap[bbox[1] : bbox[3], bbox[0] : bbox[2]].copy()
        center_in_bbox = (center[0] - bbox[0], center[1] - bbox[1])
        center_val = heatmap_bbox[center_in_bbox[1], center_in_bbox[0]]
        heatmap_center_zero = heatmap_bbox.copy()
        heatmap_center_zero[center_in_bbox[1], center_in_bbox[0]] = 0.0
        surrounding_max = np.max(heatmap_center_zero)
        return center_val > surrounding_max

    def is_center_maximum_bright(heatmap: np.ndarray, center: Tuple[int, int]):
        center_val = heatmap[center[1], center[0]]
        res = abs(1.0 - center_val) < 1e-6
        return res

    # Check that the gaussian is centered at the given center
    res = is_maximum_at_center(heatmap, center, bbox) and is_center_maximum_bright(heatmap, center)
    return res


def _check_zero_outside_radii(heatmap: np.ndarray, centers: np.ndarray, radii: float, border_width: int = 1):
    # Check that the heatmap is zero outside the given radius
    heatmap = heatmap.copy()
    for center, radius in zip(centers, radii):
        radius_border = radius + border_width
        min_x = max(0, center[0] - radius_border)
        min_y = max(0, center[1] - radius_border)
        max_x = min(heatmap.shape[1], center[0] + radius_border + 1)
        max_y = min(heatmap.shape[0], center[1] + radius_border + 1)
        heatmap[min_y:max_y, min_x:max_x] = 0.0
    res = heatmap.min() == 0.0 and heatmap.max() == 0.0
    return res


def _check_bbox_and_center_cropping(bboxes, centers, heatmap_hw):
    # Check that the bbox and center are inside the heatmap
    centers_in = (
        np.all(centers >= 0)
        and np.all(centers[:, 0] < heatmap_hw[1])
        and np.all(centers[:, 1] < heatmap_hw[0])
    )
    bboxes_in_lower = np.all(bboxes >= 0)
    bboxes_in_upper_1 = np.all(bboxes[:, 0] < heatmap_hw[1]) and np.all(bboxes[:, 1] < heatmap_hw[0])
    bboxes_in_upper_2 = np.all(bboxes[:, 2] < heatmap_hw[1]) and np.all(bboxes[:, 3] < heatmap_hw[0])
    res = centers_in and bboxes_in_lower and bboxes_in_upper_1 and bboxes_in_upper_2
    return res


def _get_radii(bboxes: np.ndarray, centers: Optional[np.ndarray] = None):
    bboxes = _get_bboxes_canonical(bboxes)
    if centers is None:
        centers = bboxes[:, :2] + bboxes[:, 2:] / 2.0
    # Compute the distance to all sides of the bbox
    dist_to_left = centers[:, 0] - bboxes[:, 0]
    dist_to_right = bboxes[:, 2] - centers[:, 0]
    dist_to_top = centers[:, 1] - bboxes[:, 1]
    dist_to_bottom = bboxes[:, 3] - centers[:, 1]
    # Get the minimum distance to any side of the bbox
    radii = np.minimum(dist_to_left, dist_to_right)
    radii = np.minimum(radii, dist_to_top)
    radii = np.minimum(radii, dist_to_bottom)

    # Is the center inside the bbox?
    is_inside_bbox = (
        (centers[:, 0] >= bboxes[:, 0])
        & (centers[:, 0] <= bboxes[:, 2])
        & (centers[:, 1] >= bboxes[:, 1])
        & (centers[:, 1] <= bboxes[:, 3])
    )
    # If the center is outside the bbox, the radius should be 0
    radii[~is_inside_bbox] = 0.0
    return radii


class TestProvider(DataProvider):

    def __init__(self, use_other_diagonal: bool = False, reverse_point_order: bool = False):
        self._use_other_diagonal = use_other_diagonal
        self._reverse_point_order = reverse_point_order

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()

        # Sibling image size
        res.add_data_field("image_hw", DALIDataType.INT32)

        # Annotation group expected by BoundingBoxToHeatmapConverter
        ann = SampleDataGroup()
        ann.add_data_field("bboxes", DALIDataType.FLOAT)
        ann.add_data_field("categories", DALIDataType.INT32)
        # Optional inputs
        ann.add_data_field("is_valid", DALIDataType.BOOL)
        ann.add_data_field("center_in_image", DALIDataType.FLOAT)

        res.add_data_group_field("annotation", ann)
        return res

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Image size (H, W)
        image_hw = np.array([100, 200], dtype=np.int32)

        # Bounding boxes [x1, y1, x2, y2] (float for deterministic rasterization)
        # 0: drawable, fully inside image
        # 1: too small per-category (size thresholds will be set in the test)
        # 2: after clipping < 25% area remains (mostly outside image)
        # 3: inactive via is_valid = False
        # 4: center provided outside its bbox (radius -> 0, will not be drawn)
        bboxes_canonical = np.array(
            [
                [20.0, 20.0, 60.0, 80.0],  # 0
                [5.0, 5.0, 7.0, 7.0],  # 1 (tiny)
                [180.0, 80.0, 380.0, 180.0],  # 2 (mostly outside image 200x100)
                [30.0, 10.0, 50.0, 30.0],  # 3 (inactive via is_valid)
                [70.0, 10.0, 90.0, 30.0],  # 4 (center outside)
                [180.0, 10.0, 210.0, 30.0],  # 5 (clipped, but still largely inside image)
                [
                    130.0,
                    50.0,
                    190.0,
                    90.0,
                ],  # 6 (large bbox inside image, but center is set to create small gaussian)
                [10.0, 10.0, 12.0, 12.0],  # 7 (tiny, but should be retained for this class)
            ],
            dtype=np.float32,
        )

        # Convert to requested diagonal/point order without redefining points separately
        xmin = bboxes_canonical[:, 0]
        ymin = bboxes_canonical[:, 1]
        xmax = bboxes_canonical[:, 2]
        ymax = bboxes_canonical[:, 3]

        if not self._use_other_diagonal:
            x1, y1, x2, y2 = xmin, ymin, xmax, ymax
        else:
            x1, y1, x2, y2 = xmin, ymax, xmax, ymin

        if self._reverse_point_order:
            x1, y1, x2, y2 = x2, y2, x1, y1

        bboxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # Categories for per-category min-size thresholds
        # Use small-box category = 1 to test per-category thresholds; others = 0
        categories = np.array([0, 1, 0, 0, 0, 1, 1, 0], dtype=np.int32)

        # Validity flags (input optional). Make index 3 inactive via validity = False
        is_valid = np.array([True, True, True, False, True, True, True, True], dtype=np.bool_)

        # Optional centers for all boxes (float, Nx2). Provide explicit centers.
        centers = np.array(
            [
                [40.0, 50.0],  # inside bbox 0
                [6.0, 6.0],  # inside tiny bbox 1
                [280.0, 130.0],  # center of bbox 2 (outside image but inside bbox)
                [40.0, 20.0],  # inside bbox 3 (but is_valid False)
                [95.0, 35.0],  # outside bbox 4 (will lead to radius 0)
                [190.0, 15.0],  # inside bbox 5
                [180.0, 80.0],  # inside bbox 6
                [11.0, 11.0],  # inside bbox 7
            ],
            dtype=np.float32,
        )

        res["image_hw"] = image_hw
        res["annotation"]["bboxes"] = bboxes
        res["annotation"]["categories"] = categories
        res["annotation"]["is_valid"] = is_valid
        res["annotation"]["center_in_image"] = centers
        return res

    @override
    def get_number_of_samples(self) -> int:
        return 10


@pytest.mark.parametrize(
    "use_other_diagonal, reverse_point_order, heatmap_hw, use_centers",
    [
        (False, False, [100, 200], True),
        (True, False, [100, 200], True),
        (False, True, [100, 200], True),
        (True, True, [100, 200], True),
        (False, False, [50, 100], True),
        (False, False, [50, 200], True),
        (False, False, [100, 100], True),
        (False, False, [100, 200], False),
    ],
    ids=[
        "canonical_bboxes",
        "other_diagonal_bboxes",
        "reverse_point_order_bboxes",
        "other_diagonal_reverse_point_order_bboxes",
        "uniform_scaling_heatmap_hw",
        "scaling_heatmap_y_only",
        "scaling_heatmap_x_only",
        "no_centers_provided",
    ],
)
def test_annotation_to_heatmap_converter(use_other_diagonal, reverse_point_order, heatmap_hw, use_centers):
    provider = TestProvider(use_other_diagonal=use_other_diagonal, reverse_point_order=reverse_point_order)
    provider_canonical_ref = TestProvider(use_other_diagonal=False, reverse_point_order=False)
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Heatmap config
    num_categories = 2
    image_hw = [100, 200]
    radius_scaling = 0.5
    # Per-category min object size thresholds: class 0: no min; class 1: require >= 3x3
    min_size_thresholds = [[1.0, 1.0], [3.0, 3.0]]

    step = BoundingBoxToHeatmapConverter(
        image_hw_field_name="image_hw",
        annotation_field_name="annotation",
        bboxes_in_name="bboxes",
        categories_in_name="categories",
        heatmap_out_name="heatmap",
        num_categories=num_categories,
        heatmap_hw=heatmap_hw,
        is_valid_opt_in_name="is_valid",
        center_opt_in_name="center_in_image" if use_centers else None,
        is_active_opt_out_name="is_active",
        center_opt_out_name="center",
        center_offset_opt_out_name="center_offset",
        height_width_bboxes_heatmap_opt_out_name="height_width_bboxes_heatmap",
        bboxes_heatmap_opt_out_name="bboxes_heatmap",
        radius_scaling_factor=radius_scaling,
        per_category_min_object_sizes=min_size_thresholds,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["image_hw"],
        passthrough_copy_branch_paths=["annotation"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(1, pipeline, pipeline_def.check_and_get_output_data_structure())
    res = next(iter(iterator))

    # Gather outputs
    heatmap = res["annotation"]["heatmap"][0]  # [C, H, W] torch tensor
    is_active = res["annotation"]["is_active"][0]
    center_offsets = res["annotation"]["center_offset"][0]
    centers_int = res["annotation"]["center"][0]
    bboxes_hm = res["annotation"]["bboxes_heatmap"][0]
    # Categories from input
    categories = res["annotation"]["categories"][0]

    # Get canonical reference
    canonical_ref_data = provider_canonical_ref.get_data(0)

    # Expectations for active flags:
    # 0: drawable → True
    # 1: too small per-category (class 1 with 2x2 < 3x3) → False
    # 2: <25% area after clipping → False
    # 3: is_valid=False → False
    # 4: center outside bbox (radius 0, but still active) → True
    # 5: clipped but sufficiently large → True
    # 6: large and valid → True
    # 7: tiny, but should be retained for this class → True
    expected_active = torch.tensor([True, False, False, False, True, True, True, True], dtype=torch.bool)
    assert torch.equal(
        is_active, expected_active
    ), f"is_active: {is_active}, expected_active: {expected_active}"

    # Convert to numpy for checks
    heatmap_np = heatmap.cpu().numpy()
    centers_np = centers_int.cpu().numpy().astype(np.int32)
    bboxes_np = bboxes_hm.cpu().numpy()
    categories_np = categories.cpu().numpy().astype(np.int32)
    is_active_np = is_active.cpu().numpy()

    # Compute radii from canonical boxes and provided centers, apply radius scaling and clamping as in the op
    # Use provider's canonical bboxes (provider default returns canonical ordering in this test case)
    canonical_bboxes = canonical_ref_data["annotation"]["bboxes"]

    if use_centers:
        centers_ref = canonical_ref_data["annotation"]["center_in_image"].astype(np.float32)
    else:
        centers_ref = _get_centers_from_bboxes(canonical_bboxes)
    radii_raw = _get_radii(canonical_bboxes, centers=centers_ref)
    radii_scaled = radii_raw * radius_scaling
    radii_scaled = np.clip(radii_scaled, 0.5, 10.0)
    radii_int = np.floor(radii_scaled).astype(np.int32)

    assert _check_bbox_and_center_cropping(
        bboxes_np, centers_np, heatmap_hw
    ), "Error in bbox and center cropping"

    assert _check_center_positions(
        centers_np, center_offsets, centers_ref, image_hw, heatmap_hw
    ), "Error in center positions"

    # For each class slice, ensure zero outside active radii regions
    for cls in range(num_categories):
        mask_idxs = np.where((categories_np == cls) & (is_active_np == True))[0]
        if mask_idxs.size == 0:
            continue
        centers_cls = centers_np[mask_idxs]
        radii_cls = radii_int[mask_idxs]
        assert _check_zero_outside_radii(
            heatmap_np[cls], centers_cls, radii_cls, border_width=1
        ), "Non-zero outside radii"

    # Verify gaussian maxima at centers for active indices with radius>0 using clipped bboxes from output
    for i in range(len(is_active_np)):
        if not is_active_np[i] or radii_int[i] <= 0:
            continue
        cls = int(categories_np[i])
        center_xy = (int(centers_np[i, 0]), int(centers_np[i, 1]))
        bbox_i = bboxes_np[i].astype(np.int32)
        assert _check_gaussian_centered_at_center(
            heatmap_np[cls], center_xy, bbox_i
        ), "Gaussian not centered at center"


@pytest.mark.parametrize(
    "use_per_class_thresholds",
    [False, True],
    ids=["global_min_size", "per_class_min_sizes"],
)
def test_annotation_to_heatmap_converter_single_heatmap(use_per_class_thresholds):
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Single heatmap config (not per-category)
    heatmap_hw = [100, 200]
    radius_scaling = 0.5

    ctor_kwargs = dict(
        image_hw_field_name="image_hw",
        annotation_field_name="annotation",
        bboxes_in_name="bboxes",
        heatmap_out_name="heatmap",
        heatmap_hw=heatmap_hw,
        is_valid_opt_in_name="is_valid",
        center_opt_in_name="center_in_image",
        is_active_opt_out_name="is_active",
        center_opt_out_name="center",
        center_offset_opt_out_name="center_offset",
        height_width_bboxes_heatmap_opt_out_name="height_width_bboxes_heatmap",
        bboxes_heatmap_opt_out_name="bboxes_heatmap",
        radius_scaling_factor=radius_scaling,
        use_per_category_heatmap=False,
    )
    if not use_per_class_thresholds:
        ctor_kwargs.update(min_object_size=[3.0, 3.0])
    else:
        ctor_kwargs.update(
            categories_in_name="categories",
            num_categories=2,
            per_category_min_object_sizes=[[1.0, 1.0], [3.0, 3.0]],
        )
    step = BoundingBoxToHeatmapConverter(**ctor_kwargs)

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["image_hw"],
        passthrough_copy_branch_paths=["annotation"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(1, pipeline, pipeline_def.check_and_get_output_data_structure())
    res = next(iter(iterator))

    # Single heatmap [H, W]
    heatmap = res["annotation"]["heatmap"][0]
    is_active = res["annotation"]["is_active"][0]
    centers_int = res["annotation"]["center"][0]
    bboxes_hm = res["annotation"]["bboxes_heatmap"][0]

    heatmap_np = heatmap.cpu().numpy()
    centers_np = centers_int.cpu().numpy().astype(np.int32)
    bboxes_np = bboxes_hm.cpu().numpy()
    is_active_np = is_active.cpu().numpy()

    # Validate cropping and centers
    assert _check_bbox_and_center_cropping(
        bboxes_np, centers_np, heatmap_hw
    ), "Error in bbox and center cropping"

    # Ensure bright maxima at centers for active detections with radius > 0
    provider_ref = TestProvider()
    canonical_bboxes = provider_ref.get_data(0)["annotation"]["bboxes"].astype(np.float32)
    centers_provided = provider_ref.get_data(0)["annotation"]["center_in_image"].astype(np.float32)
    radii_raw = _get_radii(canonical_bboxes, centers=centers_provided)
    radii_scaled = np.clip(radii_raw * radius_scaling, 0.5, 10.0)
    radii_int = np.floor(radii_scaled).astype(np.int32)

    for i in range(len(is_active_np)):
        if not is_active_np[i] or radii_int[i] <= 0:
            continue
        center_xy = (int(centers_np[i, 0]), int(centers_np[i, 1]))
        bbox_i = bboxes_np[i].astype(np.int32)
        assert _check_gaussian_centered_at_center(
            heatmap_np, center_xy, bbox_i
        ), "Gaussian not centered at center"

    # Validate selection of active detections according to thresholds
    if not use_per_class_thresholds:
        expected_active = torch.tensor([True, False, False, False, True, True, True, False], dtype=torch.bool)
    else:
        expected_active = torch.tensor([True, False, False, False, True, True, True, True], dtype=torch.bool)
    assert torch.equal(
        is_active, expected_active
    ), f"is_active: {is_active}, expected_active: {expected_active}"


def test_annotation_to_heatmap_converter_optional_outputs():
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    heatmap_hw = [100, 200]
    radius_scaling = 0.5
    num_categories = 2
    min_size_thresholds = [[1.0, 1.0], [3.0, 3.0]]

    step = BoundingBoxToHeatmapConverter(
        image_hw_field_name="image_hw",
        annotation_field_name="annotation",
        bboxes_in_name="bboxes",
        categories_in_name="categories",
        heatmap_out_name="heatmap",
        num_categories=num_categories,
        heatmap_hw=heatmap_hw,
        is_valid_opt_in_name="is_valid",
        center_opt_in_name=None,
        is_active_opt_out_name=None,
        center_opt_out_name=None,
        center_offset_opt_out_name=None,
        height_width_bboxes_heatmap_opt_out_name=None,
        bboxes_heatmap_opt_out_name=None,
        radius_scaling_factor=radius_scaling,
        per_category_min_object_sizes=min_size_thresholds,
        use_per_category_heatmap=True,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["image_hw"],
        passthrough_copy_branch_paths=["annotation"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(1, pipeline, pipeline_def.check_and_get_output_data_structure())
    res = next(iter(iterator))

    # Required output must be present
    assert "heatmap" in res["annotation"]
    heatmap = res["annotation"]["heatmap"][0].cpu().numpy()
    assert heatmap.max() > 0, "Heatmap must be non-empty"

    # Optional outputs must not be present when disabled
    for field in [
        "is_active",
        "center",
        "center_offset",
        "height_width_bboxes_heatmap",
        "bboxes_heatmap",
    ]:
        assert field not in res["annotation"], f"Optional field '{field}' should not be present"


if __name__ == "__main__":
    pytest.main([__file__])
