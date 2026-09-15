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

import torch

import numba

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np
from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import AffineTransformer

# Use a non-square base resolution to catch axis mix-ups
IN_H = 100
IN_W = 110

# -------------------------------------------------------------------------------------------------
# Test data provider & helper functions for the tests
# -------------------------------------------------------------------------------------------------


class PointImageProvider(DataProvider):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        points_xy: np.ndarray,
        projection_matrix: np.ndarray | None = None,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.points_xy = points_xy.astype(np.float64)
        self._image = _create_sparse_color_points_image(image_height, image_width, self.points_xy)
        if projection_matrix is None:
            projection_matrix = np.eye(3, dtype=np.float32)
        self._proj_mat = projection_matrix.astype(np.float32)

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        s = SampleDataGroup()
        s.add_data_field("image", DALIDataType.UINT8)
        s.add_data_field("image_hw", DALIDataType.INT32)
        s.add_data_field("points", DALIDataType.FLOAT)
        s.add_data_field("projection_matrix", DALIDataType.FLOAT)
        return s

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        s = self.sample_data_structure
        s["image"] = self._image
        s["image_hw"] = np.array([self.image_height, self.image_width], dtype=np.int32)
        s["points"] = self.points_xy.copy()
        s["projection_matrix"] = self._proj_mat
        return s

    @override
    def get_number_of_samples(self) -> int:
        return 10


class WhiteImageWithPointsProvider(DataProvider):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        points_xy: np.ndarray,
        projection_matrix: np.ndarray | None = None,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.points_xy = points_xy.astype(np.float32)
        # White image, points are metadata only
        self._image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
        if projection_matrix is None:
            projection_matrix = np.eye(3, dtype=np.float32)
        self._proj_mat = projection_matrix.astype(np.float32)

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        s = SampleDataGroup()
        s.add_data_field("image", DALIDataType.UINT8)
        s.add_data_field("image_hw", DALIDataType.INT32)
        s.add_data_field("points", DALIDataType.FLOAT)
        s.add_data_field("projection_matrix", DALIDataType.FLOAT)
        return s

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        s = self.sample_data_structure
        s["image"] = self._image
        s["image_hw"] = np.array([self.image_height, self.image_width], dtype=np.int32)
        s["points"] = self.points_xy.copy()
        s["projection_matrix"] = self._proj_mat
        return s

    @override
    def get_number_of_samples(self) -> int:
        return 1000


def _default_initial_projection_matrix() -> np.ndarray:
    # Non-identity: scaling + translation to ensure order-of-operations is tested
    scale_x, scale_y = 1.3, 1.2
    tx0, ty0 = 7.0, -5.0
    return np.array([[scale_x, 0.0, tx0], [0.0, scale_y, ty0], [0.0, 0.0, 1.0]], dtype=np.float32)


@numba.jit
def _create_sparse_color_points_image(h: int, w: int, points_xy: np.ndarray) -> np.ndarray:
    def draw_point(img: np.ndarray, x: int, y: int, color: tuple[int, int, int]):
        color = np.array(color, dtype=np.uint8)

        def clip_rect(x0, y0, x1, y1, w, h):
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 >= w:
                x1 = w - 1
            if y1 >= h:
                y1 = h - 1
            return x0, y0, x1, y1

        x0, y0, x1, y1 = clip_rect(x - 1, y - 1, x + 1, y + 1, w, h)
        img[y0 : y1 + 1, x0 : x1 + 1, :] = color * 0.8
        if 0 <= x < w and 0 <= y < h:
            img[y, x, :] = color

    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Expecting three points, one per primary color (R,G,B)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, (x, y) in enumerate(points_xy):
        cx, cy = int(x), int(y)
        draw_point(img, cx, cy, colors[i % 3])
    return img


@numba.jit
def _detect_color_point(img: np.ndarray, channel: int, tol: float = 1e-4) -> tuple[int, int]:
    # Returns (x, y) of the brightest pixel in the given channel
    chan = img[:, :, channel]
    sum_x = 0.0
    sum_y = 0.0
    sum_val = 0.0
    for y in range(chan.shape[0]):
        for x in range(chan.shape[1]):
            val = chan[y, x]
            sum_x += x * val
            sum_y += y * val
            sum_val += val
    if sum_val > tol:
        x = sum_x / sum_val
        y = sum_y / sum_val
    else:
        x = None
        y = None
    return (x, y)


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.array(arr)


def _center_of_image(h: int, w: int) -> tuple[float, float]:
    return (w * 0.5, h * 0.5)


def _affine_matrix_translate(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _affine_matrix_rotate_deg(angle_deg: float, cx: float, cy: float) -> np.ndarray:
    ang = np.deg2rad(angle_deg)
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Tc = _affine_matrix_translate(cx, cy)
    Tnc = _affine_matrix_translate(-cx, -cy)
    return Tc @ R @ Tnc


def _affine_matrix_scale(sx: float, sy: float, cx: float, cy: float) -> np.ndarray:
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Tc = _affine_matrix_translate(cx, cy)
    Tnc = _affine_matrix_translate(-cx, -cy)
    return Tc @ S @ Tnc


def _affine_matrix_shear_degrees(shx_deg: float, shy_deg: float, cx: float, cy: float) -> np.ndarray:
    shx = np.tan(np.deg2rad(shx_deg))
    shy = np.tan(np.deg2rad(shy_deg))
    Sh = np.array([[1.0, shx, 0.0], [shy, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Tc = _affine_matrix_translate(cx, cy)
    Tnc = _affine_matrix_translate(-cx, -cy)
    return Tc @ Sh @ Tnc


def _apply_affine_to_points(points_xy: np.ndarray, A: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xy.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points_xy.astype(np.float32), ones], axis=1)
    trans = (A @ homo.T).T
    res = trans[:, :2] / trans[:, 2:3]
    return res


def _compute_change_aspect_ratio_resize_expected_trafo(mode, anchor, in_h, in_w, out_h, out_w):

    def affine_matrix_scale_then_translate(sx: float, sy: float, tx: float, ty: float) -> np.ndarray:
        return np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]], dtype=np.float32)

    if mode == AffineTransformer.ResizingMode.STRETCH:
        # XY convention: scale x by out_w/in_w and y by out_h/in_h
        sx_x = out_w / in_w
        sy_y = out_h / in_h
        return affine_matrix_scale_then_translate(sx_x, sy_y, 0.0, 0.0)

    # PAD or CROP: uniform scale followed by anchor-based shift
    if mode == AffineTransformer.ResizingMode.PAD:
        s = min(out_h / in_h, out_w / in_w)
    elif mode == AffineTransformer.ResizingMode.CROP:
        s = max(out_h / in_h, out_w / in_w)
    else:
        raise AssertionError("Unsupported mode in test")

    if anchor == AffineTransformer.ResizingAnchor.TOP_OR_LEFT:
        tx, ty = 0.0, 0.0
    elif anchor == AffineTransformer.ResizingAnchor.CENTER:
        tx = (out_w - s * in_w) * 0.5
        ty = (out_h - s * in_h) * 0.5
    elif anchor == AffineTransformer.ResizingAnchor.BOTTOM_OR_RIGHT:
        tx = out_w - s * in_w
        ty = out_h - s * in_h
    else:
        raise AssertionError("Unsupported anchor in test")

    return affine_matrix_scale_then_translate(s, s, tx, ty)


# -------------------------------------------------------------------------------------------------
# General test functions used by multiple more specific test cases
# -------------------------------------------------------------------------------------------------


def _run_affine_plus_resize_test(
    transformation_steps: list[AffineTransformer.TransformationStep],
    expected_affine_matrix: np.ndarray,
    out_h: int,
    out_w: int,
    resizing_mode: AffineTransformer.ResizingMode,
    resizing_anchor: AffineTransformer.ResizingAnchor,
    atol_image: float | None = 2.0,
):
    in_h, in_w = IN_H, IN_W
    orig_points = np.array([[15, 50], [40, 30], [80, 70]], dtype=np.float32)
    proj_mat_init = _default_initial_projection_matrix()
    provider = PointImageProvider(in_h, in_w, orig_points, projection_matrix=proj_mat_init)

    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = AffineTransformer(
        output_hw=[out_h, out_w],
        resizing_mode=resizing_mode,
        resizing_anchor=resizing_anchor,
        image_field_names="image",
        projection_matrix_field_names="projection_matrix",
        point_field_names="points",
        transformation_steps=transformation_steps,
        transform_image_on_gpu=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        print_sample_data_group_format=False,
    )
    out_struct = pipeline_def.check_and_get_output_data_structure()
    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        num_threads=1,
        device_id=0,
        seed=123,
        prefetch_queue_depth=1,
        py_start_method="spawn",
    )
    pipeline.build()

    it = DALIStructuredOutputIterator(
        num_batches_in_epoch=1,
        pipeline=pipeline,
        sample_data_structure_blueprint=out_struct,
        convert_sample_data_group_to_dict=False,
    )
    batch = next(iter(it))

    img = _to_numpy(batch["image"][0])
    pts = _to_numpy(batch["points"][0])
    proj_mat = _to_numpy(batch["projection_matrix"][0])

    # Expected total matrix: resize @ affine
    expected_resize = _compute_change_aspect_ratio_resize_expected_trafo(
        resizing_mode, resizing_anchor, in_h, in_w, out_h, out_w
    )
    expected_trafo = expected_resize @ expected_affine_matrix

    exp_pts = _apply_affine_to_points(orig_points, expected_trafo)
    assert np.allclose(pts, exp_pts, atol=1e-5), f"Points mismatch. Got {pts}\nExpected {exp_pts}"
    assert proj_mat.shape == (3, 3)
    expected_proj_mat = expected_trafo @ proj_mat_init
    assert np.allclose(
        proj_mat, expected_proj_mat, atol=1e-5
    ), f"Projection matrix mismatch.\nGot\n{proj_mat}\nExpected\n{expected_proj_mat}"

    # Image shape check (validate actual image tensor size)
    assert np.all(
        img.shape == np.array([out_h, out_w, 3])
    ), f"Output image shape mismatch. Got {img.shape}, expected {(out_h, out_w, 3)}"

    if atol_image is not None:
        for ch in range(3):
            exp_xy = exp_pts[ch]
            if 0 <= exp_xy[0] < out_w and 0 <= exp_xy[1] < out_h:
                det_xy = np.array(_detect_color_point(img, ch))
                assert np.allclose(
                    det_xy, exp_xy, atol=atol_image
                ), f"Channel {ch} detected {det_xy} vs expected {exp_xy}"


# -------------------------------------------------------------------------------------------------
# Test cases
# -------------------------------------------------------------------------------------------------


def test_affine_transformer_shift_to_align_with_original_image_border_incompatible_transformations():
    """Test that ShiftToAlignWithOriginalImageBorder raises an error when used with incompatible transformations."""
    # Test with rotation (should raise an error)
    rotation_transformation_steps = [
        AffineTransformer.Rotation(1.0, 45.0),  # 45 degree rotation
        AffineTransformer.ShiftToAlignWithOriginalImageBorder(
            1.0, AffineTransformer.ShiftToAlignWithOriginalImageBorder.Border.TOP
        ),
    ]

    with pytest.raises(
        ValueError, match="Cannot perform `ShiftToAlignWithOriginalImageBorder` if rotation or shearing are"
    ):
        AffineTransformer(
            output_hw=[60, 60],
            resizing_mode=AffineTransformer.ResizingMode.PAD,
            resizing_anchor=AffineTransformer.ResizingAnchor.CENTER,
            image_field_names="image",
            transformation_steps=rotation_transformation_steps,
            transform_image_on_gpu=False,
        )

    # Test with shearing (should raise an error)
    shearing_transformation_steps = [
        AffineTransformer.Shearing(1.0, [0.1, 0.0], [0.1, 0.0]),  # X-axis shearing
        AffineTransformer.ShiftToAlignWithOriginalImageBorder(
            1.0, AffineTransformer.ShiftToAlignWithOriginalImageBorder.Border.BOTTOM
        ),
    ]

    with pytest.raises(
        ValueError, match="Cannot perform `ShiftToAlignWithOriginalImageBorder` if rotation or shearing are"
    ):
        AffineTransformer(
            output_hw=[60, 60],
            resizing_mode=AffineTransformer.ResizingMode.PAD,
            resizing_anchor=AffineTransformer.ResizingAnchor.CENTER,
            image_field_names="image",
            transformation_steps=shearing_transformation_steps,
            transform_image_on_gpu=False,
        )


@pytest.mark.parametrize(
    "border",
    [
        AffineTransformer.ShiftToAlignWithOriginalImageBorder.Border.TOP,
        AffineTransformer.ShiftToAlignWithOriginalImageBorder.Border.BOTTOM,
        AffineTransformer.ShiftToAlignWithOriginalImageBorder.Border.LEFT,
        AffineTransformer.ShiftToAlignWithOriginalImageBorder.Border.RIGHT,
    ],
    ids=["align_top", "align_bottom", "align_left", "align_right"],
)
def test_affine_transformer_shift_align_to_border_using_points_output(border):
    image_height, image_width = IN_H, IN_W
    s = 0.6
    orig_points = np.array([[15, 10], [40, 30], [80, 70]], dtype=np.float32)
    provider = PointImageProvider(
        image_height, image_width, orig_points, projection_matrix=_default_initial_projection_matrix()
    )

    input_callable = ShuffledShardedInputCallable(
        provider, batch_size=1, num_shards=1, shard_id=0, shuffle=False
    )

    transformation_steps = [
        AffineTransformer.UniformScaling(1.0, s, s),
        AffineTransformer.ShiftToAlignWithOriginalImageBorder(1.0, border),
    ]

    step = AffineTransformer(
        output_hw=[image_height, image_width],
        resizing_mode=AffineTransformer.ResizingMode.PAD,
        resizing_anchor=AffineTransformer.ResizingAnchor.CENTER,
        image_field_names="image",
        point_field_names="points",
        projection_matrix_field_names="projection_matrix",
        transformation_steps=transformation_steps,
        transform_image_on_gpu=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        print_sample_data_group_format=False,
    )
    out_struct = pipeline_def.check_and_get_output_data_structure()
    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        num_threads=1,
        device_id=0,
        seed=42,
        prefetch_queue_depth=1,
        py_start_method="spawn",
        exec_dynamic=True,
    )
    pipeline.build()

    iterator = DALIStructuredOutputIterator(
        num_batches_in_epoch=1,
        pipeline=pipeline,
        sample_data_structure_blueprint=out_struct,
        convert_sample_data_group_to_dict=False,
    )
    batch = next(iter(iterator))
    pts = _to_numpy(batch["points"][0])

    # Verify distance to aligned border equals scaled original distance
    tol = 1e-4
    if border == AffineTransformer.ShiftToAlignWithOriginalImageBorder.Border.TOP:
        # distance to top = y'
        d_actual = pts[:, 1]
        d_expected = s * orig_points[:, 1]
    elif border == AffineTransformer.ShiftToAlignWithOriginalImageBorder.Border.BOTTOM:
        d_actual = image_height - pts[:, 1]
        d_expected = s * (image_height - orig_points[:, 1])
    elif border == AffineTransformer.ShiftToAlignWithOriginalImageBorder.Border.LEFT:
        d_actual = pts[:, 0]
        d_expected = s * orig_points[:, 0]
    elif border == AffineTransformer.ShiftToAlignWithOriginalImageBorder.Border.RIGHT:
        d_actual = image_width - pts[:, 0]
        d_expected = s * (image_width - orig_points[:, 0])
    else:
        raise AssertionError("Unsupported border")

    assert np.allclose(
        d_actual, d_expected, atol=tol
    ), f"Border {border}: distances mismatch.\nActual: {d_actual}\nExpected: {d_expected}"


@pytest.mark.parametrize(
    "angle_deg,image_tolerance",
    [
        (180.0, 1),
        (45.0, 1),
    ],
    ids=["rot_180_int", "rot_45_subpixel"],
)
def test_affine_transformer_rotation_points_and_matrix(angle_deg, image_tolerance):
    image_height, image_width = IN_H, IN_W
    cx, cy = _center_of_image(image_height, image_width)
    # Use negative angle to match implementation (CCW for positive angles)
    expected_trafo = _affine_matrix_rotate_deg(-angle_deg, cx, cy)
    steps = [AffineTransformer.Rotation(1.0, angle_deg)]
    _run_affine_plus_resize_test(
        steps,
        expected_trafo,
        image_height,
        image_width,
        AffineTransformer.ResizingMode.PAD,
        AffineTransformer.ResizingAnchor.CENTER,
        atol_image=image_tolerance,
    )


@pytest.mark.parametrize(
    "sx,sy,image_tolerance",
    [
        (2.0, 2.0, 1),
        (0.5, 0.5, 1),
        (2.0, 1.5, 1),
    ],
    ids=["scale_2", "scale_0_5", "scale_nonuniform"],
)
def test_affine_transformer_scaling_points_and_matrix(sx, sy, image_tolerance):
    image_height, image_width = IN_H, IN_W
    cx, cy = _center_of_image(image_height, image_width)
    # Expected matrix is scaling around image center
    expected_trafo = _affine_matrix_scale(sx, sy, cx, cy)
    if abs(sx - sy) < 1e-6:
        steps = [AffineTransformer.UniformScaling(1.0, sx)]
    else:
        steps = [AffineTransformer.NonUniformScaling(1.0, [sx, sy])]
    _run_affine_plus_resize_test(
        steps,
        expected_trafo,
        image_height,
        image_width,
        AffineTransformer.ResizingMode.PAD,
        AffineTransformer.ResizingAnchor.CENTER,
        atol_image=image_tolerance,
    )


def test_probabilistic_first_affine_transformation_has_consistent_branch_layouts():
    """A skipped first affine step must be mergeable with its applied branch."""
    orig_points = np.array([[15, 50], [40, 30], [80, 70]], dtype=np.float32)
    provider = PointImageProvider(IN_H, IN_W, orig_points)
    batch_size = provider.get_number_of_samples()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=batch_size,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = AffineTransformer(
        output_hw=[IN_H, IN_W],
        resizing_mode=AffineTransformer.ResizingMode.PAD,
        resizing_anchor=AffineTransformer.ResizingAnchor.CENTER,
        image_field_names="image",
        projection_matrix_field_names="projection_matrix",
        point_field_names="points",
        transformation_steps=[AffineTransformer.NonUniformScaling(0.5, [0.8, 1.2])],
        transform_image_on_gpu=False,
    )
    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        print_sample_data_group_format=False,
    )
    out_struct = pipeline_def.check_and_get_output_data_structure()
    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=batch_size,
        num_threads=1,
        device_id=0,
        seed=123,
        prefetch_queue_depth=1,
        py_start_method="spawn",
    )
    pipeline.build()

    iterator = DALIStructuredOutputIterator(
        num_batches_in_epoch=1,
        pipeline=pipeline,
        sample_data_structure_blueprint=out_struct,
        convert_sample_data_group_to_dict=False,
    )
    batch = next(iter(iterator))

    assert len(batch["image"]) == batch_size
    assert np.all(np.isfinite(_to_numpy(batch["points"])))
    assert np.all(np.isfinite(_to_numpy(batch["projection_matrix"])))


@pytest.mark.parametrize(
    "shx,shy,image_tolerance",
    [
        (10.0, 0.0, 1),
        (0.0, -15.0, 1),
        (-5.0, 7.0, 1),
    ],
    ids=["shear_x", "shear_y", "shear_xy_subpixel"],
)
def test_affine_transformer_shearing_points_and_matrix(shx, shy, image_tolerance):
    image_height, image_width = IN_H, IN_W
    cx, cy = _center_of_image(image_height, image_width)
    expected_trafo = _affine_matrix_shear_degrees(shx, shy, cx, cy)
    steps = [AffineTransformer.Shearing(1.0, [shx, shy])]
    _run_affine_plus_resize_test(
        steps,
        expected_trafo,
        image_height,
        image_width,
        AffineTransformer.ResizingMode.PAD,
        AffineTransformer.ResizingAnchor.CENTER,
        atol_image=image_tolerance,
    )


@pytest.mark.parametrize(
    "dx,dy,image_tolerance",
    [
        (0, 0, 1),
        (5, 0, 1),
        (0, 5, 1),
        (12.5, 7.25, 1),
        (-10.5, 3.5, 1),
    ],
    ids=[
        "int_0_0",
        "int_5_0",
        "int_0_5",
        "subpixel_12_5_7_25",
        "subpixel_-10_5_3_5",
    ],
)
def test_affine_transformer_translation_image_points_and_matrix(dx, dy, image_tolerance):
    image_height, image_width = IN_H, IN_W
    dx = float(dx)
    dy = float(dy)
    steps = [AffineTransformer.Translation(1.0, [dx, dy])]
    expected_trafo = _affine_matrix_translate(dx, dy)
    _run_affine_plus_resize_test(
        steps,
        expected_trafo,
        image_height,
        image_width,
        AffineTransformer.ResizingMode.PAD,
        AffineTransformer.ResizingAnchor.CENTER,
        atol_image=image_tolerance,
    )


def test_affine_transformer_combined_affine_then_resize():
    # Rotation followed by translation, then downsample with aspect ratio change using PAD + CENTER
    angle = 30.0
    dx, dy = 12.0, -7.0
    out_h = IN_H // 2
    out_w = IN_W // 2 + 7

    cx, cy = _center_of_image(IN_H, IN_W)
    A_rot = _affine_matrix_rotate_deg(-angle, cx, cy)
    A_trans = _affine_matrix_translate(dx, dy)
    A_affine = A_trans @ A_rot

    _run_affine_plus_resize_test(
        [
            AffineTransformer.Rotation(1.0, angle),
            AffineTransformer.Translation(1.0, [dx, dy]),
        ],
        A_affine,
        out_h,
        out_w,
        AffineTransformer.ResizingMode.PAD,
        AffineTransformer.ResizingAnchor.CENTER,
        atol_image=2.0,
    )


@pytest.mark.parametrize(
    "mode,anchor,out_hw,atol",
    [
        # PAD cases
        (AffineTransformer.ResizingMode.PAD, AffineTransformer.ResizingAnchor.CENTER, (IN_H, IN_W + 60), 1),
        (
            AffineTransformer.ResizingMode.PAD,
            AffineTransformer.ResizingAnchor.TOP_OR_LEFT,
            (IN_H, IN_W + 60),
            1,
        ),
        (
            AffineTransformer.ResizingMode.PAD,
            AffineTransformer.ResizingAnchor.BOTTOM_OR_RIGHT,
            (IN_H, IN_W + 60),
            1,
        ),
        (AffineTransformer.ResizingMode.PAD, AffineTransformer.ResizingAnchor.CENTER, (IN_H + 60, IN_W), 1),
        (
            AffineTransformer.ResizingMode.PAD,
            AffineTransformer.ResizingAnchor.TOP_OR_LEFT,
            (IN_H + 60, IN_W),
            1,
        ),
        (
            AffineTransformer.ResizingMode.PAD,
            AffineTransformer.ResizingAnchor.BOTTOM_OR_RIGHT,
            (IN_H + 60, IN_W),
            1,
        ),
        # CROP cases
        (AffineTransformer.ResizingMode.CROP, AffineTransformer.ResizingAnchor.CENTER, (IN_H, IN_W - 40), 1),
        (
            AffineTransformer.ResizingMode.CROP,
            AffineTransformer.ResizingAnchor.TOP_OR_LEFT,
            (IN_H, IN_W - 40),
            1,
        ),
        (
            AffineTransformer.ResizingMode.CROP,
            AffineTransformer.ResizingAnchor.BOTTOM_OR_RIGHT,
            (IN_H, IN_W - 40),
            1,
        ),
        (AffineTransformer.ResizingMode.CROP, AffineTransformer.ResizingAnchor.CENTER, (IN_H - 40, IN_W), 1),
        (
            AffineTransformer.ResizingMode.CROP,
            AffineTransformer.ResizingAnchor.TOP_OR_LEFT,
            (IN_H - 40, IN_W),
            1,
        ),
        (
            AffineTransformer.ResizingMode.CROP,
            AffineTransformer.ResizingAnchor.BOTTOM_OR_RIGHT,
            (IN_H - 40, IN_W),
            1,
        ),
    ],
    ids=[
        # PAD ids
        "pad_center_wide",
        "pad_tl_wide",
        "pad_br_wide",
        "pad_center_tall",
        "pad_tl_tall",
        "pad_br_tall",
        # CROP ids
        "crop_center_narrow",
        "crop_tl_narrow",
        "crop_br_narrow",
        "crop_center_short",
        "crop_tl_short",
        "crop_br_short",
    ],
)
def test_resizing_mode_pad_and_crop(mode, anchor, out_hw, atol):
    out_h, out_w = out_hw
    identity = np.eye(3, dtype=np.float32)
    _run_affine_plus_resize_test(
        [],
        identity,
        out_h,
        out_w,
        mode,
        anchor,
        atol_image=atol,
    )


@pytest.mark.parametrize(
    "out_hw",
    [
        (IN_H, IN_W + 60),  # horizontal stretch
        (IN_H + 60, IN_W),  # vertical stretch
        (IN_H - 40, IN_W + 60),  # both directions
    ],
    ids=["stretch_wide", "stretch_tall", "stretch_both"],
)
def test_resizing_mode_stretch(out_hw):
    out_h, out_w = out_hw
    identity = np.eye(3, dtype=np.float32)
    _run_affine_plus_resize_test(
        [],
        identity,
        out_h,
        out_w,
        AffineTransformer.ResizingMode.STRETCH,
        None,
        atol_image=2.0,
    )


def test_shift_inside_original_image_random_shifts_white_image():
    # Make the image larger than the viewport by scaling up first, then random ShiftInsideOriginalImage should keep viewport fully inside image
    image_height, image_width = IN_H, IN_W
    # Define points in various locations
    orig_points = np.array([[10.0, 10.0], [IN_W - 15.0, 12.0], [IN_W / 2.0, IN_H / 2.0]], dtype=np.float32)
    proj_mat_init = _default_initial_projection_matrix()

    provider = WhiteImageWithPointsProvider(
        image_height, image_width, orig_points, projection_matrix=proj_mat_init
    )

    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Scale up uniformly (e.g., 1.6x) to ensure image is larger than viewport in both dims
    scale_factor = 1.6
    steps = [
        # Enlarge the image to ensure that is is larger than the viewport
        AffineTransformer.UniformScaling(1.0, scale_factor),
        # Add translation to shift the image outside the viewport
        AffineTransformer.Translation(1.0, [1000.0, 0.0]),
        # This should shift the image so as to cover the entire viewport (and randomly shift it
        # while keeping the viewport inside the image)
        AffineTransformer.ShiftInsideOriginalImage(1.0, True, True),
    ]

    step = AffineTransformer(
        output_hw=[image_height, image_width],
        resizing_mode=AffineTransformer.ResizingMode.PAD,
        resizing_anchor=AffineTransformer.ResizingAnchor.CENTER,
        image_field_names="image",
        projection_matrix_field_names="projection_matrix",
        point_field_names="points",
        transformation_steps=steps,
        transform_image_on_gpu=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        print_sample_data_group_format=False,
    )
    out_struct = pipeline_def.check_and_get_output_data_structure()
    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=128,
        num_threads=1,
        device_id=0,
        seed=234,
        prefetch_queue_depth=1,
        py_start_method="spawn",
    )
    pipeline.build()

    # Iterate multiple times on same pipeline to exercise randomness
    it = DALIStructuredOutputIterator(
        num_batches_in_epoch=20,
        pipeline=pipeline,
        sample_data_structure_blueprint=out_struct,
        convert_sample_data_group_to_dict=False,
    )

    batch = next(iter(it))

    # Check image content
    for i in range(128):
        img = _to_numpy(batch["image"][i])
        # Image is completely white
        assert img.dtype == np.uint8
        # Allow for some border effects in the image, so that:
        # 1. The image is completely white apart from a border of 1 pixel
        assert (
            img[1:-1, 1:-1, :].min() == 255
        ), "Resulting image should be completely white (possibly apart from a border of 1 pixel)"
        # 2. When including the border, the image should be at least "almost white", but some interpolation
        #    effects are allowed.
        assert img.min() >= 200, "Resulting image should be bright in the border region"

    # Check points
    all_pts = _to_numpy(batch["points"])
    # Check that points vary between iterations using the standard deviation
    std_dev = np.std(all_pts, axis=0)
    for i in range(3):
        assert std_dev[i, 0] > IN_W * 0.6 * 0.2, "Points should vary between iterations"
        assert std_dev[i, 1] > IN_H * 0.6 * 0.2, "Points should vary between iterations"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])
