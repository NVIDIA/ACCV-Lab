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

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np
import torch

from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import AnnotationElementConditionEval


class TestProvider(DataProvider):

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure
        # Create test data with 6 objects
        res["annotation"]["num_lidar_points"] = np.array([0, 1, 0, 1, 5, 10])
        res["annotation"]["num_radar_points"] = np.array([9, 10, 0, 1, 0, 0])
        res["annotation"]["visibility_levels"] = np.array([1, 1, 1, 0, 2, 3])
        res["annotation"]["visibility_2"] = np.array([1, 1, 1, 0, 2, 3])
        res["annotation"]["categories"] = np.array([0, 1, 2, 3, 4, 5])
        res["annotation"]["is_bbox_in_range"] = np.array([True, True, False, True, True, False])
        res["annotation"]["other_field"] = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        # Add nested annotation inside the main annotation for testing nested structure
        res["annotation"]["annotation"]["num_lidar_points"] = np.array([0, 0, 0, 1])
        res["annotation"]["annotation"]["num_radar_points"] = np.array([0, 100, 0, 1])
        res["annotation"]["annotation"]["visibility_levels"] = np.array([1, 2, 1, -1])
        res["annotation"]["annotation"]["visibility_2"] = np.array([1, 2, 1, -1])
        res["annotation"]["annotation"]["categories"] = np.array([0, 1, 2, 3])
        res["annotation"]["annotation"]["is_bbox_in_range"] = np.array([True, True, False, True])
        res["annotation"]["annotation"]["other_field"] = np.array([100.0, 200.0, 300.0, 400.0])

        res["other_data"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        return res

    @override
    def get_number_of_samples(self) -> int:
        return 10

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()

        # Main annotation group
        annotation = SampleDataGroup()
        annotation.add_data_field("num_lidar_points", DALIDataType.INT32)
        annotation.add_data_field("num_radar_points", DALIDataType.INT32)
        annotation.add_data_field("visibility_levels", DALIDataType.INT32)
        annotation.add_data_field("visibility_2", DALIDataType.INT32)
        annotation.add_data_field("categories", DALIDataType.INT32)
        annotation.add_data_field("is_bbox_in_range", DALIDataType.BOOL)
        annotation.add_data_field("other_field", DALIDataType.FLOAT)

        # Nested annotation group inside the main annotation (same name for testing find_all_occurrences)
        nested_annotation = SampleDataGroup()
        nested_annotation.add_data_field("num_lidar_points", DALIDataType.INT32)
        nested_annotation.add_data_field("num_radar_points", DALIDataType.INT32)
        nested_annotation.add_data_field("visibility_levels", DALIDataType.INT32)
        nested_annotation.add_data_field("visibility_2", DALIDataType.INT32)
        nested_annotation.add_data_field("categories", DALIDataType.INT32)
        nested_annotation.add_data_field("is_bbox_in_range", DALIDataType.BOOL)
        nested_annotation.add_data_field("other_field", DALIDataType.FLOAT)
        annotation.add_data_group_field("annotation", nested_annotation)  # Nested with same name

        res.add_data_group_field("annotation", annotation)
        res.add_data_field("other_data", DALIDataType.FLOAT)
        return res


def test_simple_comparison_condition():
    """Test a simple comparison condition."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="has_lidar_points = num_lidar_points >= 1",
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    # Check that the condition result is correct
    # Expected: [False, True, False, True, True, True] for num_lidar_points >= 1
    expected_result = torch.tensor([False, True, False, True, True, True], dtype=torch.bool)
    assert torch.equal(res["annotation"]["has_lidar_points"][0], expected_result)

    # Check that original fields are preserved
    assert "num_lidar_points" in res["annotation"]
    assert torch.equal(
        res["annotation"]["num_lidar_points"][0], torch.tensor([0, 1, 0, 1, 5, 10], dtype=torch.int32)
    )


def test_complex_condition_with_and_or():
    """Test a complex condition using And and Or operators."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="is_valid = (num_lidar_points >= 1 or num_radar_points >= 1) and visibility_levels > 0",
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    # Check that the condition result is correct
    # Expected: [True, True, False, False, True, True]
    # - Object 0: has radar points (9) AND visibility > 0 (1) = True
    # - Object 1: has lidar points (1) AND visibility > 0 (1) = True
    # - Object 2: no points AND visibility > 0 (1) = False
    # - Object 3: has radar points (1) AND visibility > 0 (0) = False
    # - Object 4: has lidar points (5) AND visibility > 0 (2) = True
    # - Object 5: has lidar points (10) AND visibility > 0 (3) = True
    expected_result = torch.tensor([True, True, False, False, True, True], dtype=torch.bool)
    assert torch.equal(res["annotation"]["is_valid"][0], expected_result)


def test_not_condition():
    """Test the Not operator."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="not_highly_visible = not (visibility_levels > 1)",
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    # Check that the condition result is correct
    # visibility_levels: [1, 1, 1, 0, 2, 3]
    # visibility_levels > 1: [False, False, False, False, True, True]
    # not (visibility_levels > 1): [True, True, True, True, False, False]
    expected_result = torch.tensor([True, True, True, True, False, False], dtype=torch.bool)
    assert torch.equal(res["annotation"]["not_highly_visible"][0], expected_result)


def test_remove_condition_fields():
    """Test that fields used in the condition can be removed."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="is_valid = num_lidar_points >= 1 and visibility_levels > 0",
        remove_data_fields_used_in_condition=True,  # Remove fields used in condition
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    # Check that the condition result is correct
    expected_result = torch.tensor([False, True, False, False, True, True], dtype=torch.bool)
    assert torch.equal(res["annotation"]["is_valid"][0], expected_result)

    # Check that fields used in the condition were removed
    assert "num_lidar_points" not in res["annotation"]
    assert "visibility_levels" not in res["annotation"]

    # Check that other fields are still present
    assert "num_radar_points" in res["annotation"]
    assert "categories" in res["annotation"]
    assert "is_bbox_in_range" in res["annotation"]
    assert "other_field" in res["annotation"]


def test_multiple_annotation_fields():
    """Test that the step works with multiple annotation fields."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="is_visible = visibility_levels > 0",
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    # Check that both annotation fields were processed (independently of each other)
    # Main annotation: [True, True, True, False, True, True]
    expected_main = torch.tensor([True, True, True, False, True, True], dtype=torch.bool)
    assert torch.equal(res["annotation"]["is_visible"][0], expected_main)

    # Nested annotation: [True, True, True, False]
    expected_nested = torch.tensor([True, True, True, False], dtype=torch.bool)
    assert torch.equal(res["annotation"]["annotation"]["is_visible"][0], expected_nested)


@pytest.mark.parametrize(
    "operator,operator_name,value,expected",
    [
        ("==", "eq", 1, [False, True, False, True, False, False]),
        ("!=", "neq", 1, [True, False, True, False, True, True]),
        ("<", "lt", 2, [True, True, True, True, False, False]),
        (">", "gt", 2, [False, False, False, False, True, True]),
        ("<=", "le", 1, [True, True, True, True, False, False]),
        (">=", "ge", 1, [False, True, False, True, True, True]),
    ],
)
def test_comparison_operators(operator, operator_name, value, expected):
    """Test all comparison operators."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Test different comparison operators
    condition = f"test_{operator_name}_{value} = num_lidar_points {operator} {value}"
    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition=condition,
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    expected_result = torch.tensor(expected, dtype=torch.bool)
    assert torch.equal(res["annotation"][f"test_{operator_name}_{value}"][0], expected_result)


def test_boolean_field_condition():
    """Test comparison with boolean fields."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="is_in_range = is_bbox_in_range",
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    # Expected: [True, True, False, True, True, False]
    expected_result = torch.tensor([True, True, False, True, True, False], dtype=torch.bool)
    assert torch.equal(res["annotation"]["is_in_range"][0], expected_result)


def test_other_data_preserved():
    """Test that data outside annotation fields is preserved."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="has_lidar_points = num_lidar_points >= 1",
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    # Check that non-annotation data is preserved
    assert "other_data" in res
    assert torch.equal(res["other_data"][0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    # Check that annotation data is also preserved
    assert "annotation" in res
    assert "num_lidar_points" in res["annotation"]
    assert "num_radar_points" in res["annotation"]
    assert "visibility_levels" in res["annotation"]


def test_number_in_identifier():
    """Test conditions with field names that contain numbers in the identifier."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Simple condition to test if the `visibility_2` field is accessible
    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="highly_visible_2 = visibility_2 > 1",
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    # visibility_2: [1, 1, 1, 0, 2, 3]
    # visibility_2 > 1: [False, False, False, False, True, True]
    expected_result = torch.tensor([False, False, False, False, True, True], dtype=torch.bool)
    assert torch.equal(res["annotation"]["highly_visible_2"][0], expected_result)

    # Verify that the original visibility_2 field is preserved
    assert "visibility_2" in res["annotation"]
    assert torch.equal(
        res["annotation"]["visibility_2"][0], torch.tensor([1, 1, 1, 0, 2, 3], dtype=torch.int32)
    )


def test_logical_vs_bitwise_and():
    """Test that 'and' operator works logically, not bitwise, with non-boolean fields."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Test logical AND with non-boolean fields
    # num_lidar_points: [0, 1, 0, 1, 5, 10]
    # num_radar_points: [9, 10, 0, 1, 0, 0]
    #
    # Logical AND should be:
    # - Object 0: 0 and 9 -> False (first is zero/falsy)
    # - Object 1: 1 and 10 -> True (both non-zero/truthy)
    # - Object 2: 0 and 0 -> False (both zero/falsy)
    # - Object 3: 1 and 1 -> True (both non-zero/truthy)
    # - Object 4: 5 and 0 -> False (second is zero/falsy)
    # - Object 5: 10 and 0 -> False (second is zero/falsy)
    #
    # If it were bitwise AND:
    # - Object 0: 0 & 9 -> 0 (False)
    # - Object 1: 1 & 10 -> 0 (False)
    # - Object 2: 0 & 0 -> 0 (False)
    # - Object 3: 1 & 1 -> 1 (True)
    # - Object 4: 5 & 0 -> 0 (False)
    # - Object 5: 10 & 0 -> 0 (False)
    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="both_points = num_lidar_points and num_radar_points",
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    # Expected logical AND result: [False, True, False, True, False, False]
    expected_logical_result = torch.tensor([False, True, False, True, False, False], dtype=torch.bool)
    assert torch.equal(res["annotation"]["both_points"][0], expected_logical_result)

    # Verify that the original fields are preserved
    assert "num_lidar_points" in res["annotation"]
    assert "num_radar_points" in res["annotation"]
    assert torch.equal(
        res["annotation"]["num_lidar_points"][0], torch.tensor([0, 1, 0, 1, 5, 10], dtype=torch.int32)
    )
    assert torch.equal(
        res["annotation"]["num_radar_points"][0], torch.tensor([9, 10, 0, 1, 0, 0], dtype=torch.int32)
    )


@pytest.mark.parametrize(
    "operator,operator_name,expected",
    [
        ("==", "eq", [False, False, True, True, False, False]),
        ("!=", "neq", [True, True, False, False, True, True]),
        ("<", "lt", [True, True, False, False, False, False]),
        (">", "gt", [False, False, False, False, True, True]),
        ("<=", "le", [True, True, True, True, False, False]),
        (">=", "ge", [False, False, True, True, True, True]),
    ],
)
def test_direct_field_comparison_operators(operator, operator_name, expected):
    """Test all comparison operators with direct field comparison."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Test different comparison operators between fields
    # num_lidar_points: [0, 1, 0, 1, 5, 10]
    # num_radar_points: [9, 10, 0, 1, 0, 0]
    condition = f"lidar_{operator_name}_radar = num_lidar_points {operator} num_radar_points"
    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition=condition,
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    expected_result = torch.tensor(expected, dtype=torch.bool)
    assert torch.equal(res["annotation"][f"lidar_{operator_name}_radar"][0], expected_result)


def test_decimal_values_in_condition():
    """Test using decimal values in conditions."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Test various decimal/float operations
    # other_field: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="ranged_value = other_field > 25.5 and other_field <= 55.0",
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    # Expected: [False, False, True, True, True, False]
    expected_result = torch.tensor([False, False, True, True, True, False], dtype=torch.bool)
    assert torch.equal(res["annotation"]["ranged_value"][0], expected_result)


def test_negative_values_and_unary_minus():
    """Test using negative values and unary minus operations on variables."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Test with negative literal values and unary minus operator
    # visibility_levels: [1, 1, 1, 0, 2, 3]
    step = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="high_visibility = -visibility_levels <= -1",
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    expected_result = torch.tensor([True, True, True, False, True, True], dtype=torch.bool)
    assert torch.equal(res["annotation"]["high_visibility"][0], expected_result)

    # Test comparison with negative literal values
    step2 = AnnotationElementConditionEval(
        annotation_field_name="annotation",
        condition="negative_lidar = num_lidar_points < -2",
        remove_data_fields_used_in_condition=False,
    )

    pipeline_def2 = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step2],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
    )

    pipeline2 = pipeline_def2.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator2 = DALIStructuredOutputIterator(
        10, pipeline2, pipeline_def2.check_and_get_output_data_structure()
    )
    iterator_iter2 = iter(iterator2)
    res2 = next(iterator_iter2)

    expected_result2 = torch.tensor([False, False, False, False, False, False], dtype=torch.bool)
    assert torch.equal(res2["annotation"]["negative_lidar"][0], expected_result2)


if __name__ == "__main__":
    test_negative_values_and_unary_minus()
    pytest.main([__file__])
