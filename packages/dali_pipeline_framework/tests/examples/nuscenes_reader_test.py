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

"""Test example-reader preparation without a dataset or DALI graph construction."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="module")
def example_modules():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    try:
        yield SimpleNamespace(
            reader=importlib.import_module("pipeline_setup.additional_impl.data_loading.nuscenes_reader"),
            pipeline=importlib.import_module("pipeline_setup.stream_petr_pipeline"),
        )
    finally:
        sys.path.pop(0)


@pytest.fixture
def synthetic_scene(example_modules, monkeypatch):
    reader = example_modules.reader.NuScenesReader
    # One frame with six cameras; no exported image-annotation table is provided.
    sdk = SimpleNamespace(
        get=lambda table, token: {"name": "scene-test"},
        get_sample_data=lambda token: (None, [], None),
    )
    state = SimpleNamespace(
        sdk_loader=Mock(return_value=sdk),
        annotation_map=Mock(return_value={}),
        exported=Mock(return_value=[[[{"source": "exported"}] for _ in range(6)]]),
        projected=Mock(return_value=[[{"source": "projected"} for _ in range(6)]]),
    )
    monkeypatch.setattr(reader, "_get_dataset", state.sdk_loader)
    monkeypatch.setattr(
        reader,
        "_get_sequences_of_samples",
        lambda self, data: [[{"scene_token": "scene-test", "data": {"LIDAR_TOP": "lidar"}}]],
    )
    monkeypatch.setattr(reader, "_get_sample_data_cams", lambda self, data, sample: [{}] * 6)
    monkeypatch.setattr(reader, "_get_sample_data_lidar", lambda self, data, sample: {})
    identity_pose = {"rotation": [1, 0, 0, 0]}
    for name in ("_get_calibrated_sensors", "_get_ego_poses"):
        monkeypatch.setattr(
            reader,
            name,
            lambda self, data, records, is_single_sensor: (
                [identity_pose] if is_single_sensor else [[identity_pose] * 6]
            ),
        )
    for name in ("_get_annotations", "_get_annotation_velocities", "_get_visibilities"):
        monkeypatch.setattr(reader, name, lambda self, *args: [[]])
    monkeypatch.setattr(reader, "_build_image_annotations_map", state.annotation_map)
    monkeypatch.setattr(reader, "_get_image_data_annotations", state.exported)
    monkeypatch.setattr(reader, "_generate_projected_bboxes_annotations", state.projected)
    return state


def test_projected_annotations_do_not_require_exported_annotations(example_modules, synthetic_scene):
    synthetic_scene.annotation_map.side_effect = AssertionError("Exported annotations must not be accessed")
    data = example_modules.reader.NuScenesReader(
        "/unused",
        "v1.0-mini",
        add_image_annotations=False,
        add_projected_bboxes_annotations=True,
        image_size=(1600, 900),
    ).data
    sample = data.get_sample_in_sequence(0, 0)
    assert sample.image_annotations is None
    assert sample.projected_bboxes_annotations == synthetic_scene.projected.return_value[0]
    synthetic_scene.annotation_map.assert_not_called()
    synthetic_scene.exported.assert_not_called()
    synthetic_scene.projected.assert_called_once()


def test_exported_annotations_remain_enabled_by_default(example_modules, synthetic_scene):
    data = example_modules.reader.NuScenesReader("/unused", "v1.0-mini").data
    sample = data.get_sample_in_sequence(0, 0)
    assert sample.image_annotations == synthetic_scene.exported.return_value[0]
    assert sample.projected_bboxes_annotations is None
    synthetic_scene.annotation_map.assert_called_once()
    synthetic_scene.exported.assert_called_once()
    synthetic_scene.projected.assert_not_called()


@pytest.mark.parametrize("add_image_annotations", [False, True], ids=["without-exported", "with-exported"])
def test_cache_creation_and_reuse(example_modules, synthetic_scene, tmp_path, add_image_annotations):
    load = example_modules.reader.NuScenesReader.load_data_if_available_else_create_and_store
    args = (str(tmp_path), "v1.0-mini_preproc_streampetr.pkl", "v1.0-mini")
    kwargs = dict(
        add_image_annotations=add_image_annotations,
        add_projected_bboxes_annotations=True,
        image_size=(1600, 900),
    )
    if add_image_annotations:
        version_dir = tmp_path / args[2]
        version_dir.mkdir()
        (version_dir / "image_annotations.json").write_text("[]")

    created = load(*args, **kwargs)
    assert (tmp_path / args[1]).is_file()
    expected_annotations = synthetic_scene.exported.return_value[0] if add_image_annotations else None
    assert created.get_sample_in_sequence(0, 0).image_annotations == expected_annotations
    if add_image_annotations:
        synthetic_scene.annotation_map.assert_called_once()
        synthetic_scene.exported.assert_called_once()
    else:
        synthetic_scene.annotation_map.assert_not_called()
        synthetic_scene.exported.assert_not_called()
    synthetic_scene.projected.assert_called_once()

    synthetic_scene.sdk_loader.reset_mock()
    synthetic_scene.annotation_map.reset_mock()
    synthetic_scene.exported.reset_mock()
    synthetic_scene.projected.reset_mock()
    synthetic_scene.sdk_loader.side_effect = AssertionError("A cache hit must not read the dataset")
    restored = load(*args, **kwargs)
    assert restored.get_sample_in_sequence(0, 0).image_annotations == expected_annotations
    assert restored.get_sample_in_sequence(0, 0).projected_bboxes_annotations == (
        created.get_sample_in_sequence(0, 0).projected_bboxes_annotations
    )
    synthetic_scene.sdk_loader.assert_not_called()
    synthetic_scene.annotation_map.assert_not_called()
    synthetic_scene.exported.assert_not_called()
    synthetic_scene.projected.assert_not_called()


@pytest.mark.parametrize("version", ["v1.0-mini", "v1.0-trainval"])
def test_exported_annotation_guard_uses_configured_version_subdirectory(
    example_modules, synthetic_scene, tmp_path, version
):
    version_dir = tmp_path / version
    version_dir.mkdir()
    (version_dir / "image_annotations.json").write_text("[]")
    assert not (tmp_path / "image_annotations.json").exists()
    data = example_modules.reader.NuScenesReader.load_data_if_available_else_create_and_store(
        str(tmp_path),
        "cache.pkl",
        version,
    )
    assert data.get_sample_in_sequence(0, 0).image_annotations is not None
    synthetic_scene.sdk_loader.assert_called_once_with(str(tmp_path), version)
    assert (tmp_path / "cache.pkl").is_file()


@pytest.mark.parametrize("location", [None, "root", "other-version"])
def test_missing_versioned_annotations_reports_correct_path(
    example_modules, synthetic_scene, tmp_path, location
):
    if location is not None:
        annotation_dir = tmp_path if location == "root" else tmp_path / "v1.0-trainval"
        annotation_dir.mkdir(exist_ok=True)
        (annotation_dir / "image_annotations.json").write_text("[]")
    expected_path = tmp_path / "v1.0-mini" / "image_annotations.json"
    with pytest.raises(RuntimeError) as caught:
        example_modules.reader.NuScenesReader.load_data_if_available_else_create_and_store(
            str(tmp_path),
            "cache.pkl",
            "v1.0-mini",
        )
    assert str(expected_path) in str(caught.value)
    synthetic_scene.sdk_loader.assert_not_called()
    assert not (tmp_path / "cache.pkl").exists()


def test_streampetr_setup_requests_only_projected_annotations(example_modules, monkeypatch):
    class ReaderCalled(Exception):
        pass

    load = Mock(side_effect=ReaderCalled)
    monkeypatch.setattr(
        example_modules.reader.NuScenesReader, "load_data_if_available_else_create_and_store", load
    )
    with pytest.raises(ReaderCalled):
        example_modules.pipeline.setup_dali_pipeline_stream_petr_train(
            "/unused",
            "v1.0-mini",
            10,
            SimpleNamespace(image_size=[1600, 900]),
        )
    load.assert_called_once_with(
        "/unused",
        "v1.0-mini_preproc_streampetr.pkl",
        "v1.0-mini",
        add_image_annotations=False,
        add_projected_bboxes_annotations=True,
        image_size=[1600, 900],
    )
