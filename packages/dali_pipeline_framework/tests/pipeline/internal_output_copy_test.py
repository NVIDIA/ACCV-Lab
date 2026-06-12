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

import numpy as np
import pytest
import torch
from typing import Set, Tuple, Union

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import DataProvider, ShuffledShardedInputCallable
from accvlab.dali_pipeline_framework.pipeline import (
    DALIStructuredOutputIterator,
    PipelineDefinition,
    SampleDataGroup,
)
from accvlab.dali_pipeline_framework.pipeline import _insert_copy_for_passthrough
from accvlab.dali_pipeline_framework.pipeline._insert_copy_for_passthrough import (
    _InsertCopyForPassthrough,
)


def _build_output_structure() -> SampleDataGroup:
    # SampleDataGroup structure used for testing the copy helper.
    res = SampleDataGroup()
    res.add_data_field("image", DALIDataType.UINT8)

    camera = SampleDataGroup()
    camera.add_data_field("image", DALIDataType.UINT8)
    camera.add_data_field("label", DALIDataType.INT32)

    metadata = SampleDataGroup()
    metadata.add_data_field("label", DALIDataType.INT32)
    metadata.add_data_field("score", DALIDataType.FLOAT)
    camera.add_data_group_field("metadata", metadata)

    lidar = SampleDataGroup()
    lidar.add_data_field("points", DALIDataType.FLOAT)

    res.add_data_group_field("camera", camera)
    res.add_data_group_field("lidar", lidar)
    return res


def _build_copy_helper(**kwargs) -> _InsertCopyForPassthrough:
    structure = _build_output_structure()
    return _InsertCopyForPassthrough(structure, **kwargs)


def _paths_as_set(helper: _InsertCopyForPassthrough) -> Set[Tuple[Union[str, int], ...]]:
    # Keep resolution assertions readable without exposing the helper's tuple ordering.
    return set(helper._paths_to_copy)


# Values assigned to each leaf in the test output tree when using string marker values (see tests). The path keys are also the
# complete set of leaf paths expected when the copy helper is configured without selectors.
_MARKER_VALUES = {
    ("image",): "root-image",
    ("camera", "image"): "camera-image",
    ("camera", "label"): "camera-label",
    ("camera", "metadata", "label"): "metadata-label",
    ("camera", "metadata", "score"): "metadata-score",
    ("lidar", "points"): "lidar-points",
}


def _build_data_with_marker_values() -> SampleDataGroup:
    '''Build test data populated with string marker values at every leaf.'''

    data = _build_output_structure()
    # Use plain string marker values instead of DALI nodes so tests can assert exact object flow without
    # building a graph. Conversion and type checks are disabled because the blueprint declares numeric fields.
    data.set_do_convert(False)
    data.set_do_check_type(False)
    for path, value in _MARKER_VALUES.items():
        data.set_item_in_path(path, value)
    return data


def _apply_copy_with_fake_fn(data: SampleDataGroup, monkeypatch, **kwargs) -> SampleDataGroup:
    '''Apply the copy helper with a fake ``fn.copy`` that marks copied values.'''

    def fake_copy(value):
        return f"copied-{value}"

    monkeypatch.setattr(_insert_copy_for_passthrough.fn, "copy", fake_copy)

    helper = _InsertCopyForPassthrough(data.get_empty_like_self(), **kwargs)
    return helper(data)


def _assert_copied_paths(data: SampleDataGroup, copied_paths: Set[Tuple[Union[str, int], ...]]) -> None:
    '''Assert selected paths were copied and unselected paths stayed unchanged.'''

    for path, original_value in _MARKER_VALUES.items():
        expected_value = f"copied-{original_value}" if path in copied_paths else original_value
        assert data.get_item_in_path(path) == expected_value


def test_internal_output_copy_without_selectors_resolves_all_output_leaves():
    helper = _build_copy_helper()

    assert _paths_as_set(helper) == {
        ("image",),
        ("camera", "image"),
        ("camera", "label"),
        ("camera", "metadata", "label"),
        ("camera", "metadata", "score"),
        ("lidar", "points"),
    }


@pytest.mark.parametrize("kwargs", [{"field_names": []}, {"branch_paths": []}])
def test_internal_output_copy_empty_selectors_resolve_no_output_leaves(kwargs):
    helper = _build_copy_helper(**kwargs)

    assert _paths_as_set(helper) == set()


@pytest.mark.parametrize("kwargs", [{"field_names": []}, {"branch_paths": []}])
def test_internal_output_copy_empty_selectors_do_not_copy_outputs(kwargs, monkeypatch):
    data = _build_data_with_marker_values()

    _apply_copy_with_fake_fn(data, monkeypatch, **kwargs)

    _assert_copied_paths(data, set())


def test_internal_output_copy_resolves_field_names_globally():
    helper = _build_copy_helper(field_names=["label"])

    assert _paths_as_set(helper) == {
        ("camera", "label"),
        ("camera", "metadata", "label"),
    }


def test_internal_output_copy_resolves_field_names_under_scope_paths():
    helper = _build_copy_helper(
        field_names=["label"],
        field_names_scope_paths=[("camera", "metadata")],
    )

    assert _paths_as_set(helper) == {("camera", "metadata", "label")}


def test_internal_output_copy_resolves_branch_paths():
    helper = _build_copy_helper(branch_paths=[("camera", "metadata"), ("lidar", "points")])

    assert _paths_as_set(helper) == {
        ("camera", "metadata", "label"),
        ("camera", "metadata", "score"),
        ("lidar", "points"),
    }


def test_internal_output_copy_rejects_invalid_paths():
    with pytest.raises(ValueError, match="does not exist"):
        _build_copy_helper(branch_paths=["missing"])

    with pytest.raises(ValueError, match="data group field"):
        _build_copy_helper(
            field_names=["label"],
            field_names_scope_paths=["image"],
        )


def test_internal_output_copy_applies_copy_to_field_names_globally(monkeypatch):
    data = _build_data_with_marker_values()

    _apply_copy_with_fake_fn(data, monkeypatch, field_names=["label"])

    _assert_copied_paths(
        data,
        {
            ("camera", "label"),
            ("camera", "metadata", "label"),
        },
    )


def test_internal_output_copy_applies_copy_to_field_names_under_scope_paths(monkeypatch):
    data = _build_data_with_marker_values()

    _apply_copy_with_fake_fn(
        data,
        monkeypatch,
        field_names=["label"],
        field_names_scope_paths=[("camera", "metadata")],
    )

    _assert_copied_paths(data, {("camera", "metadata", "label")})


def test_internal_output_copy_applies_copy_to_branch_paths(monkeypatch):
    data = _build_data_with_marker_values()

    _apply_copy_with_fake_fn(
        data,
        monkeypatch,
        branch_paths=[("camera", "metadata"), ("lidar", "points")],
    )

    _assert_copied_paths(
        data,
        {
            ("camera", "metadata", "label"),
            ("camera", "metadata", "score"),
            ("lidar", "points"),
        },
    )


def test_internal_output_copy_applies_copy_to_all_outputs_when_no_selectors(monkeypatch):
    data = _build_data_with_marker_values()

    _apply_copy_with_fake_fn(data, monkeypatch)

    _assert_copied_paths(data, set(_MARKER_VALUES))


class _OutputCopyProvider(DataProvider):
    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure
        res["image"] = np.full((2,), sample_id, dtype=np.uint8)
        res["camera"]["label"] = np.array([sample_id + 10], dtype=np.int32)
        return res

    @override
    def get_number_of_samples(self) -> int:
        return 4

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()
        res.add_data_field("image", DALIDataType.UINT8)

        camera = SampleDataGroup()
        camera.add_data_field("label", DALIDataType.INT32)
        res.add_data_group_field("camera", camera)
        return res


def test_pipeline_definition_copies_all_passthrough_outputs_by_default(monkeypatch):
    batch_size = 1
    input_callable = ShuffledShardedInputCallable(
        data_provider=_OutputCopyProvider(),
        batch_size=batch_size,
        shard_id=0,
        num_shards=1,
        shuffle=False,
    )
    with pytest.warns(UserWarning, match="Copying all final pipeline outputs by default"):
        pipeline_def = PipelineDefinition(
            data_loading_callable_iterable=input_callable,
            preprocess_functors=[],
            # The fake copy below adds an integer constant, which can promote DALI output dtypes.
            # Keep this test focused on whether the copy hook is applied.
            check_data_format=False,
        )

    def fake_copy(value):
        # Deliberately modify the graph output so the test can observe that the copy hook was applied.
        return value + 1

    monkeypatch.setattr(_insert_copy_for_passthrough.fn, "copy", fake_copy)

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=batch_size,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )
    iterator = DALIStructuredOutputIterator(
        batch_size, pipeline, pipeline_def.check_and_get_output_data_structure()
    )

    batch = next(iter(iterator))

    assert torch.equal(batch["image"][0].to(torch.int64), torch.tensor([1, 1], dtype=torch.int64))
    assert torch.equal(batch["camera"]["label"][0].to(torch.int64), torch.tensor([11], dtype=torch.int64))


def test_pipeline_definition_rejects_copy_selectors_when_copying_is_default():
    with pytest.raises(ValueError, match="copy_external_source_passthrough_outputs=True"):
        PipelineDefinition(
            data_loading_callable_iterable=ShuffledShardedInputCallable(
                data_provider=_OutputCopyProvider(),
                batch_size=1,
                shard_id=0,
                num_shards=1,
                shuffle=False,
            ),
            preprocess_functors=[],
            passthrough_copy_branch_paths=["image"],
        )


def test_pipeline_definition_can_copy_selected_passthrough_outputs(monkeypatch):
    batch_size = 2
    input_callable = ShuffledShardedInputCallable(
        data_provider=_OutputCopyProvider(),
        batch_size=batch_size,
        shard_id=0,
        num_shards=1,
        shuffle=False,
    )
    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[],
        # The fake copy below adds an integer constant, which can promote DALI output dtypes.
        # Batch size > 1 in combination with callable (i.e. per-sample) inputs avoids the potential pass-through issue,
        # allowing to reliably test for unchanged values.
        check_data_format=False,
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["label"],
    )

    def fake_copy(value):
        return value + 1

    monkeypatch.setattr(_insert_copy_for_passthrough.fn, "copy", fake_copy)

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=batch_size,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )
    iterator = DALIStructuredOutputIterator(
        batch_size, pipeline, pipeline_def.check_and_get_output_data_structure()
    )

    batch = next(iter(iterator))

    # With ``shuffle=False`` the first batch contains sample ids 0 and 1. ``image`` is not selected for
    # copying, so it should remain filled with the original sample id. ``label`` starts at
    # ``sample_id + 10`` and is selected, so the fake copy adds one more. ``batch_size=2`` keeps the
    # unselected callable-input output away from the single-sample pass-through case documented above.
    assert torch.equal(batch["image"].to(torch.int64), torch.tensor([[0, 0], [1, 1]], dtype=torch.int64))
    assert torch.equal(
        batch["camera"]["label"].to(torch.int64), torch.tensor([[11], [12]], dtype=torch.int64)
    )


def test_pipeline_definition_rejects_copy_selectors_when_copying_is_disabled():
    with pytest.raises(ValueError, match="copy_external_source_passthrough_outputs=True"):
        PipelineDefinition(
            data_loading_callable_iterable=ShuffledShardedInputCallable(
                data_provider=_OutputCopyProvider(),
                batch_size=1,
                shard_id=0,
                num_shards=1,
                shuffle=False,
            ),
            preprocess_functors=[],
            copy_external_source_passthrough_outputs=False,
            passthrough_copy_branch_paths=["image"],
        )


def test_pipeline_definition_reports_invalid_copy_setup_with_context():
    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=ShuffledShardedInputCallable(
            data_provider=_OutputCopyProvider(),
            batch_size=1,
            shard_id=0,
            num_shards=1,
            shuffle=False,
        ),
        preprocess_functors=[],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["missing"],
    )

    with pytest.raises(
        ValueError,
        match="Invalid pass-through output copy configuration for final output format: .*does not exist",
    ):
        pipeline_def.check_and_get_output_data_structure()


if __name__ == "__main__":
    pytest.main([__file__])
