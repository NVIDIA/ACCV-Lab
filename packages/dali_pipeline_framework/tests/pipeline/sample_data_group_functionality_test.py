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
import numpy as np

from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.pipeline import SampleDataGroup


def test_type_matches_identical():
    a = _make_blueprint_for_get_set_tests()
    b = _make_blueprint_for_get_set_tests()
    assert a.type_matches(b)
    assert b.type_matches(a)


def test_type_matches_detects_type_change():
    base = _make_blueprint_for_get_set_tests()
    modified = _make_blueprint_for_get_set_tests()
    # Change type of a specific leaf
    modified.change_type_of_data_and_remove_data(("annotation", "token"), DALIDataType.INT32)
    assert not base.type_matches(modified)
    assert not modified.type_matches(base)


def test_type_matches_detects_structural_addition():
    base = _make_blueprint_for_get_set_tests()
    modified = _make_blueprint_for_get_set_tests()
    # Add a new field inside 'meta'
    modified["meta"].add_data_field("extra", DALIDataType.FLOAT)
    assert not base.type_matches(modified)
    assert not modified.type_matches(base)


def test_type_matches_ignores_mapping_changes():
    base = _make_blueprint_for_get_set_tests()
    modified = _make_blueprint_for_get_set_tests()
    # Change mapping of an existing field without changing its type
    new_mapping = {"car": 10, "ped": 20, None: -5}
    modified.change_type_of_data_and_remove_data(
        ("annotation", "categories"), DALIDataType.INT32, new_mapping
    )
    # Types and structure are the same → still matches
    assert base.type_matches(modified)
    assert modified.type_matches(base)


def test_apply_mapping_on_lists_and_nested_sequences():
    # Mapping similar to utilities providers (with default mapped via None)
    categories_mapping = {"pedestrian": 0, "bicycle": 1, "vehicle": 2, None: 3}

    root = SampleDataGroup()
    ann = SampleDataGroup()
    ann.add_data_field("categories", DALIDataType.INT32, categories_mapping)
    root.add_data_group_field("annotation", ann)

    # 1D list mapping, including an unknown label → default via None
    root["annotation"]["categories"] = ["vehicle", "pedestrian", "unknown"]
    out_1d = root["annotation"]["categories"]
    assert isinstance(out_1d, np.ndarray)
    assert out_1d.dtype == np.int32
    np.testing.assert_array_equal(out_1d, np.array([2, 0, 3], dtype=np.int32))

    # Nested sequences mapping (list of lists)
    root["annotation"]["categories"] = [["vehicle", "bicycle"], ["unknown", "pedestrian"]]
    out_2d = root["annotation"]["categories"]
    assert out_2d.dtype == np.int32
    np.testing.assert_array_equal(out_2d, np.array([[2, 1], [3, 0]], dtype=np.int32))


def test_apply_mapping_with_tuple_and_default_case():
    mapping = {"a": 5, "b": 7, None: 9}

    s = SampleDataGroup()
    s.add_data_field("labels", DALIDataType.INT32, mapping)

    # Tuple input with an unknown label → default value 9
    s["labels"] = ("a", "b", "nope")
    out = s["labels"]
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, np.array([5, 7, 9], dtype=np.int32))


def test_get_empty_like_self_structure_types_and_no_data():
    # Build a moderately nested structure with mappings
    categories_mapping = {"pedestrian": 0, "bicycle": 1, "vehicle": 2, None: 3}
    visibility_mapping = {"v0-40": 0, "v40-60": 1, None: -1}

    root = SampleDataGroup()
    root.add_data_field("id", DALIDataType.INT32)

    meta = SampleDataGroup()
    meta.add_data_field("score", DALIDataType.FLOAT)
    root.add_data_group_field("meta", meta)

    ann = SampleDataGroup()
    ann.add_data_field("categories", DALIDataType.INT32, categories_mapping)
    ann.add_data_field("visibility", DALIDataType.INT32, visibility_mapping)
    root.add_data_group_field("annotation", ann)

    # Fill with some actual data so we can verify the new object is empty
    root["id"] = 42
    root["meta"]["score"] = 0.5
    root["annotation"]["categories"] = ["vehicle", "pedestrian", "unknown"]
    root["annotation"]["visibility"] = ["v0-40", "v40-60", "nope"]

    empty = root.get_empty_like_self()

    # Same structure
    assert empty.type_matches(root)
    assert empty.field_names_flat == root.field_names_flat

    # Same data types
    assert empty.field_types_flat == root.field_types_flat

    # New object does not contain actual data
    original_data = root.get_data()
    empty_data = empty.get_data()
    # Ensure original actually had non-None values for meaningful check
    assert any(v is not None for v in original_data)
    # Empty blueprint should have None for all leaf nodes
    assert all(v is None for v in empty_data)


def test_get_copy_structure_types_and_data_copied_and_independent_structure():
    # Build a moderately nested structure with mappings and actual data
    categories_mapping = {"pedestrian": 0, "bicycle": 1, "vehicle": 2, None: 3}
    visibility_mapping = {"v0-40": 0, "v40-60": 1, None: -1}

    root = SampleDataGroup()
    root.add_data_field("id", DALIDataType.INT32)

    meta = SampleDataGroup()
    meta.add_data_field("score", DALIDataType.FLOAT)
    root.add_data_group_field("meta", meta)

    ann = SampleDataGroup()
    ann.add_data_field("categories", DALIDataType.INT32, categories_mapping)
    ann.add_data_field("visibility", DALIDataType.INT32, visibility_mapping)
    root.add_data_group_field("annotation", ann)

    # Fill with actual data
    root["id"] = 7
    root["meta"]["score"] = 0.75
    root["annotation"]["categories"] = ["vehicle", "pedestrian", "unknown"]
    root["annotation"]["visibility"] = ["v40-60", "v0-40", "nope"]

    # Obtain copy
    copy_obj = root.get_copy()

    # Same structure and types
    assert copy_obj.type_matches(root)
    assert copy_obj.field_names_flat == root.field_names_flat
    assert copy_obj.field_types_flat == root.field_types_flat

    # Data should be present and equal to original
    orig_data = root.get_data()
    copy_data = copy_obj.get_data()
    assert len(orig_data) == len(copy_data)
    for a, b in zip(orig_data, copy_data):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b

    # Modify structure of the copy: add a new field and remove an existing one
    copy_obj.add_data_field("new_field", DALIDataType.FLOAT)
    copy_obj.remove_field("meta")

    # The original should be unaffected structurally
    assert root.has_child("meta")
    assert not root.has_child("new_field")

    # Structures now differ
    assert not copy_obj.type_matches(root)


def _make_blueprint_for_get_set_tests() -> SampleDataGroup:
    s = SampleDataGroup()
    s.add_data_field("id", DALIDataType.INT32)

    meta = SampleDataGroup()
    meta.add_data_field("score", DALIDataType.FLOAT)
    meta.add_data_field("flag", DALIDataType.BOOL)
    s.add_data_group_field("meta", meta)

    ann = SampleDataGroup()
    ann.add_data_field("bboxes", DALIDataType.FLOAT)
    ann.add_data_field("categories", DALIDataType.INT32, {"car": 0, "ped": 1, None: 2})
    ann.add_data_field("token", DALIDataType.STRING)
    s.add_data_group_field("annotation", ann)

    seq = SampleDataGroup()
    seq.add_data_field("value", DALIDataType.FLOAT)
    s.add_data_group_field_array("sequence", seq, 2)

    return s


def test_get_data_and_set_data_roundtrip_between_blueprints():
    # Create the blueprint twice via the same function
    a = _make_blueprint_for_get_set_tests()
    b = _make_blueprint_for_get_set_tests()

    # Fill A with some data
    a["id"] = 123
    a["meta"]["score"] = 0.25
    a["meta"]["flag"] = True
    a["annotation"]["bboxes"] = [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]]
    a["annotation"]["categories"] = ["car", "ped", "unknown"]
    a["annotation"]["token"] = "sample token"
    a["sequence"][0]["value"] = 3.14
    a["sequence"][1]["value"] = -2.5

    # Transfer data via get_data and set_data
    flat_data = a.get_data()
    b.set_data(flat_data)

    # Compare flattened data equality
    a_flat = a.get_data()
    b_flat = b.get_data()
    assert len(a_flat) == len(b_flat)
    for va, vb in zip(a_flat, b_flat):
        if isinstance(va, np.ndarray):
            np.testing.assert_array_equal(va, vb)
        else:
            assert va == vb

    # Also ensure names/types line up
    assert a.field_names_flat == b.field_names_flat
    assert a.field_types_flat == b.field_types_flat


def test_set_data_from_dali_generic_iterator_output_with_flattened_names():
    # Build blueprint and fill a source instance
    src = _make_blueprint_for_get_set_tests()
    src["id"] = 5
    src["meta"]["score"] = 9.5
    src["meta"]["flag"] = False
    src["annotation"]["bboxes"] = [[1.0, 2.0, 3.0, 4.0]]
    src["annotation"]["categories"] = ["ped", "unknown", "car"]
    src["annotation"]["token"] = "example token"
    src["sequence"][0]["value"] = 0.0
    src["sequence"][1]["value"] = 1.0

    # Simulate DALIGenericIterator output: a list of dictionaries keyed by flattened names
    flattened_names = src.field_names_flat
    flat_values = src.get_data()
    iterator_like_output = [{name: value for name, value in zip(flattened_names, flat_values)}]

    # Fill destination from the simulated iterator output
    dst = _make_blueprint_for_get_set_tests()
    dst.set_data_from_dali_generic_iterator_output(iterator_like_output, index=0)

    # Verify equality of flattened data
    src_flat = src.get_data()
    dst_flat = dst.get_data()
    assert len(src_flat) == len(dst_flat)
    for vs, vd in zip(src_flat, dst_flat):
        if isinstance(vs, np.ndarray):
            np.testing.assert_array_equal(vs, vd)
        else:
            assert vs == vd

    # Names and types should also match
    assert src.field_names_flat == dst.field_names_flat
    assert src.field_types_flat == dst.field_types_flat


@pytest.mark.parametrize("ingest_mode", ["set_data", "get_like_self"])
def test_iterator_output_ingestion_with_flattened_names_parametrized(ingest_mode: str):
    # Source: build and fill
    src = _make_blueprint_for_get_set_tests()
    src["id"] = 11
    src["meta"]["score"] = -1.25
    src["meta"]["flag"] = True
    src["annotation"]["bboxes"] = [[9.0, 8.0, 7.0, 6.0], [0.5, 1.5, 2.5, 3.5]]
    src["annotation"]["categories"] = ["unknown", "car", "ped"]
    src["annotation"]["token"] = "param token"
    src["sequence"][0]["value"] = 42.0
    src["sequence"][1]["value"] = -42.0

    # Simulated iterator output
    names = src.field_names_flat
    values = src.get_data()
    iterator_like_output = [{name: val for name, val in zip(names, values)}]

    if ingest_mode == "set_data":
        # Destination blueprint to be filled in-place
        dst = _make_blueprint_for_get_set_tests()
        dst.set_data_from_dali_generic_iterator_output(iterator_like_output, 0)
        target = dst
    else:
        # Destination blueprint should remain empty; data returned in a new instance
        blueprint = _make_blueprint_for_get_set_tests()
        target = blueprint.get_empty_like_self()
        target.set_data_from_dali_generic_iterator_output(iterator_like_output, 0)
        # Ensure original blueprint still empty
        assert all(v is None for v in blueprint.get_data())

    # Verify data equality and schema match
    src_flat = src.get_data()
    tgt_flat = target.get_data()
    assert len(src_flat) == len(tgt_flat)
    for vs, vt in zip(src_flat, tgt_flat):
        if isinstance(vs, np.ndarray):
            np.testing.assert_array_equal(vs, vt)
        else:
            assert vs == vt

    assert src.field_names_flat == target.field_names_flat
    assert src.field_types_flat == target.field_types_flat


def test_set_data_without_flattening_verifies_nested_fields():
    # Build and fill source with distinct values to detect any misplacement
    src = _make_blueprint_for_get_set_tests()
    src["id"] = 99
    src["meta"]["score"] = 1.25
    src["meta"]["flag"] = True
    src["annotation"]["bboxes"] = [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
    src["annotation"]["categories"] = ["ped", "car", "unknown"]  # maps to [1, 0, 2]
    src["annotation"]["token"] = "noflat token"
    src["sequence"][0]["value"] = 100.5
    src["sequence"][1]["value"] = -100.5

    # Destination blueprint; ingest via set_data using flat sequence only (no flattened names)
    dst = _make_blueprint_for_get_set_tests()
    dst.set_data(src.get_data())

    # Verify nested fields individually to ensure correct placement, not just flat equality
    assert dst["id"] == 99
    assert np.isclose(dst["meta"]["score"], 1.25)
    assert bool(dst["meta"]["flag"]) is True
    np.testing.assert_array_equal(
        dst["annotation"]["bboxes"], np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(dst["annotation"]["categories"], np.array([1, 0, 2], dtype=np.int32))
    assert dst["annotation"]["token"] == "noflat token"
    assert np.isclose(dst["sequence"][0]["value"], 100.5)
    assert np.isclose(dst["sequence"][1]["value"], -100.5)

    # Flat equality as a sanity check
    np.testing.assert_equal(src.field_names_flat, dst.field_names_flat)
    assert src.field_types_flat == dst.field_types_flat
    a_flat, b_flat = src.get_data(), dst.get_data()
    assert len(a_flat) == len(b_flat)
    for va, vb in zip(a_flat, b_flat):
        if isinstance(va, np.ndarray):
            np.testing.assert_array_equal(va, vb)
        else:
            assert va == vb


def test_change_type_of_data_and_remove_data_on_data_field_and_mapping():
    # Build blueprint and fill data
    s = _make_blueprint_for_get_set_tests()
    s["id"] = 1
    s["meta"]["score"] = 0.0
    s["meta"]["flag"] = False
    s["annotation"]["bboxes"] = [[1.0, 1.0, 2.0, 2.0]]
    s["annotation"]["categories"] = ["car", "ped", "unknown"]  # maps to [0,1,2]

    # Change nested data field type via path to FLOAT and remove mapping
    before_types = s.field_types_flat
    s.change_type_of_data_and_remove_data(("annotation", "categories"), DALIDataType.FLOAT)

    # Value removed (reset to None)
    assert s["annotation"]["categories"] is None

    # Type updated in flattened types at the correct position
    after_types = s.field_types_flat
    assert len(before_types) == len(after_types)
    diff_idx = -1
    for i, (t0, t1) in enumerate(zip(before_types, after_types)):
        if t0 != t1:
            diff_idx = i
            break
    assert diff_idx != -1
    assert after_types[diff_idx] == DALIDataType.FLOAT

    # Setting numeric values should work without mapping
    s["annotation"]["categories"] = [0.5, 1.5, 2.5]
    arr = s["annotation"]["categories"]
    assert isinstance(arr, np.ndarray) and arr.dtype == np.float32
    np.testing.assert_array_equal(arr, np.array([0.5, 1.5, 2.5], dtype=np.float32))

    # Now change type again but keep INT32 and set a new mapping
    new_mapping = {"x": 10, "y": 20, None: -1}
    s.change_type_of_data_and_remove_data(("annotation", "categories"), DALIDataType.INT32, new_mapping)
    assert s["annotation"]["categories"] is None
    # Assign with new mapping
    s["annotation"]["categories"] = ["y", "z", "x"]
    np.testing.assert_array_equal(s["annotation"]["categories"], np.array([20, -1, 10], dtype=np.int32))


def test_change_type_of_data_and_remove_data_on_data_group():
    # Original tree has 'annotation' with bboxes/categories/token
    s = _make_blueprint_for_get_set_tests()
    s["annotation"]["bboxes"] = [[0.0, 0.0, 1.0, 1.0]]
    s["annotation"]["categories"] = ["car"]
    s["annotation"]["token"] = "orig"

    # Define a new data group blueprint to replace 'annotation'
    new_ann = SampleDataGroup()
    new_ann.add_data_field("labels", DALIDataType.INT32)
    new_ann.add_data_field("mask", DALIDataType.BOOL)

    # Change type at top-level name (not tuple path) and remove data
    s.change_type_of_data_and_remove_data("annotation", new_ann)

    # The group is replaced; new fields exist and are empty; old fields are gone
    assert s["annotation"].has_child("labels")
    assert s["annotation"].has_child("mask")
    assert not s["annotation"].has_child("bboxes")
    assert not s["annotation"].has_child("categories")
    assert not s["annotation"].has_child("token")
    assert s["annotation"]["labels"] is None
    assert s["annotation"]["mask"] is None

    # Setting data in the new structure should work
    s["annotation"]["labels"] = [1, 2, 3]
    s["annotation"]["mask"] = [True, False, True]
    np.testing.assert_array_equal(s["annotation"]["labels"], np.array([1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(s["annotation"]["mask"], np.array([1, 0, 1], dtype=np.int8))


def test_get_flat_index_first_discrepancy_to_other_identical():
    a = _make_blueprint_for_get_set_tests()
    b = _make_blueprint_for_get_set_tests()
    assert a.get_flat_index_first_discrepancy_to_other(b) == -1
    assert b.get_flat_index_first_discrepancy_to_other(a) == -1


def test_get_flat_index_first_discrepancy_to_other_detects_type_change_index():
    base = _make_blueprint_for_get_set_tests()
    modified = _make_blueprint_for_get_set_tests()

    # Change type of a specific leaf while keeping structure and names identical
    modified.change_type_of_data_and_remove_data(("annotation", "token"), DALIDataType.INT32)

    # Determine expected index for 'annotation.token' in the flattened order
    names = base.field_names_flat
    target_name = "annotation.token"
    expected_idx = list(names).index(target_name)

    assert base.get_flat_index_first_discrepancy_to_other(modified) == expected_idx
    assert modified.get_flat_index_first_discrepancy_to_other(base) == expected_idx


def test_get_flat_index_first_discrepancy_to_other_detects_structural_addition_index():
    base = _make_blueprint_for_get_set_tests()
    modified = _make_blueprint_for_get_set_tests()

    # Add a new field inside 'meta' so the discrepancy appears before 'annotation.*'
    modified["meta"].add_data_field("extra", DALIDataType.FLOAT)

    base_names = base.field_names_flat
    modified_names = modified.field_names_flat

    # The first difference should occur where 'meta.extra' appears in modified,
    # which is at the index where base transitions to 'annotation.bboxes'.
    expected_idx = list(base_names).index("annotation.bboxes")
    assert modified_names[expected_idx] == "meta.extra"

    assert base.get_flat_index_first_discrepancy_to_other(modified) == expected_idx
    assert modified.get_flat_index_first_discrepancy_to_other(base) == expected_idx


def test_find_all_occurrences_with_duplicate_names():
    """Test that find_all_occurrences correctly finds all fields with the same name in nested structures."""
    # Create a SampleDataGroup with nested structure containing duplicate field names
    root = SampleDataGroup()

    # Main annotation group
    annotation = SampleDataGroup()
    annotation.add_data_field("num_lidar_points", DALIDataType.INT32)
    annotation.add_data_field("num_radar_points", DALIDataType.INT32)
    annotation.add_data_field("visibility_levels", DALIDataType.INT32)

    # Nested annotation group inside the main annotation (same name for testing find_all_occurrences)
    nested_annotation = SampleDataGroup()
    nested_annotation.add_data_field("num_lidar_points", DALIDataType.INT32)
    nested_annotation.add_data_field("num_radar_points", DALIDataType.INT32)
    nested_annotation.add_data_field("visibility_levels", DALIDataType.INT32)
    annotation.add_data_group_field("annotation", nested_annotation)

    root.add_data_group_field("annotation", annotation)

    # Test find_all_occurrences behavior with duplicate names
    annotation_paths = root.find_all_occurrences("annotation")

    # This should find both annotation fields
    assert len(annotation_paths) == 2
    assert ("annotation",) in annotation_paths  # Top-level annotation
    assert ("annotation", "annotation") in annotation_paths  # Nested annotation

    # Test finding other fields
    lidar_paths = root.find_all_occurrences("num_lidar_points")

    # This should find both num_lidar_points fields
    assert len(lidar_paths) == 2
    assert ("annotation", "num_lidar_points") in lidar_paths  # Top-level annotation
    assert ("annotation", "annotation", "num_lidar_points") in lidar_paths  # Nested annotation

    # Test finding a field that only exists in one location
    other_data_paths = root.find_all_occurrences("other_data")

    # This should find no paths since other_data doesn't exist
    assert len(other_data_paths) == 0


def test_find_all_occurrences_with_data():
    """Test that find_all_occurrences works correctly when data is populated."""
    # Create a SampleDataGroup with nested structure containing duplicate field names
    root = SampleDataGroup()

    # Main annotation group
    annotation = SampleDataGroup()
    annotation.add_data_field("num_lidar_points", DALIDataType.INT32)
    annotation.add_data_field("num_radar_points", DALIDataType.INT32)
    annotation.add_data_field("visibility_levels", DALIDataType.INT32)

    # Nested annotation group inside the main annotation (same name for testing find_all_occurrences)
    nested_annotation = SampleDataGroup()
    nested_annotation.add_data_field("num_lidar_points", DALIDataType.INT32)
    nested_annotation.add_data_field("num_radar_points", DALIDataType.INT32)
    nested_annotation.add_data_field("visibility_levels", DALIDataType.INT32)
    annotation.add_data_group_field("annotation", nested_annotation)

    root.add_data_group_field("annotation", annotation)

    # Populate with data
    root["annotation"]["num_lidar_points"] = np.array([0, 1, 0, 1, 5, 10])
    root["annotation"]["num_radar_points"] = np.array([9, 10, 0, 1, 0, 0])
    root["annotation"]["visibility_levels"] = np.array([1, 1, 1, 0, 2, 3])

    root["annotation"]["annotation"]["num_lidar_points"] = np.array([0, 0, 0, 1])
    root["annotation"]["annotation"]["num_radar_points"] = np.array([0, 100, 0, 1])
    root["annotation"]["annotation"]["visibility_levels"] = np.array([1, 2, 1, -1])

    # Test find_all_occurrences behavior with data populated
    annotation_paths = root.find_all_occurrences("annotation")

    # This should find both annotation fields even with data populated
    assert len(annotation_paths) == 2
    assert ("annotation",) in annotation_paths  # Top-level annotation
    assert ("annotation", "annotation") in annotation_paths  # Nested annotation

    # Test accessing data through the found paths
    for path in annotation_paths:
        item = root.get_item_in_path(path)
        assert isinstance(item, SampleDataGroup)
        assert item.has_child("num_lidar_points")
        assert item.has_child("num_radar_points")
        assert item.has_child("visibility_levels")


def test_check_has_children_success_and_errors():
    s = SampleDataGroup()
    s.add_data_field("id", DALIDataType.INT32)

    meta = SampleDataGroup()
    meta.add_data_field("score", DALIDataType.FLOAT)
    meta.add_data_field("flag", DALIDataType.BOOL)
    s.add_data_group_field("meta", meta)

    # Array of data groups under "sequence"
    seq_elem = SampleDataGroup()
    seq_elem.add_data_field("value", DALIDataType.FLOAT)
    s.add_data_group_field_array("sequence", seq_elem, 2)

    # Correct expectations
    s.check_has_children(
        data_field_children=["id"],
        data_group_field_children=["meta"],
        data_group_field_array_children=["sequence"],
        current_name="root",
    )

    # Wrong type: expect data field array but have data group array
    with pytest.raises(ValueError, match="is not a data field array"):
        s.check_has_children(
            data_field_children=["id"],
            data_group_field_children=["meta"],
            data_field_array_children=[
                "sequence"
            ],  # sequence is array of data groups, not data fields → should NOT pass
        )

    # Missing child
    with pytest.raises(ValueError, match="does not have child `missing`"):
        s.check_has_children(data_field_children=["missing"])

    # Wrong type: expect data field but have data group
    with pytest.raises(ValueError, match="is not a data field"):
        s.check_has_children(data_field_children=["meta"])

    # Wrong type: expect data group but have data field
    with pytest.raises(ValueError, match="is not a data group field"):
        s.check_has_children(data_group_field_children=["id"])

    # Wrong type: expect data field array but have data group array
    with pytest.raises(ValueError, match="is not a data field array"):
        s.check_has_children(data_field_array_children=["sequence"])


def test_path_helpers_get_set_parent_and_existence():
    s = SampleDataGroup()
    inner = SampleDataGroup()
    leaf = SampleDataGroup()
    leaf2 = SampleDataGroup()
    leaf.add_data_field("val", DALIDataType.FLOAT)
    leaf.add_data_field("val2", DALIDataType.FLOAT)
    leaf2.add_data_field("val", DALIDataType.FLOAT)
    inner.add_data_group_field("leaf", leaf)
    inner.add_data_group_field("leaf2", leaf2)
    s.add_data_group_field("inner", inner)

    # Set via path and get via path
    s.set_item_in_path(["inner", "leaf", "val"], 3.5)
    got = s.get_item_in_path(("inner", "leaf", "val"))
    got2 = s["inner"]["leaf"]["val"]
    assert np.isclose(got, 3.5)
    assert np.isclose(got2, 3.5)

    # Parent of path
    parent = s.get_parent_of_path(("inner", "leaf", "val"))
    assert isinstance(parent, SampleDataGroup)
    assert parent.has_child("val")

    # Existence checks
    assert s.path_exists(("inner", "leaf", "val"))
    assert not s.path_exists(("inner", "leaf", "missing"))
    assert s.path_exists_and_is_data_group_field(("inner", "leaf"))
    assert not s.path_exists_and_is_data_group_field(("inner", "leaf", "val"))

    # Type queries
    assert s.get_type_of_item_in_path(["inner", "leaf"]) == SampleDataGroup
    assert s.get_type_of_item_in_path(["inner", "leaf", "val"]) == DALIDataType.FLOAT
    assert s.get_type_of_field("inner") == SampleDataGroup


def test_path_helpers_error_cases():
    s = SampleDataGroup()
    child = SampleDataGroup()
    child.add_data_field("x", DALIDataType.INT32)
    s.add_data_group_field("child", child)

    # set_item_in_path with missing head
    with pytest.raises(KeyError, match="No field with name 'missing'"):
        s.set_item_in_path(["missing", "x"], 1)

    # get_item_in_path with missing head
    with pytest.raises(KeyError, match="No field with name 'missing'"):
        s.get_item_in_path(["missing", "x"])

    # get_parent_of_path with missing element
    with pytest.raises(KeyError, match="No element 'missing' is present"):
        s.get_parent_of_path("missing")

    # get_parent_of_path with empty path → assertion
    with pytest.raises(AssertionError):
        s.get_parent_of_path([])


def test_remove_field_and_remove_all_occurrences():
    s = SampleDataGroup()
    s.add_data_field("keep", DALIDataType.FLOAT)
    grp = SampleDataGroup()
    grp.add_data_field("dup", DALIDataType.INT32)
    s.add_data_group_field("g1", grp)
    s.add_data_group_field("g2", grp)

    # Direct remove_field
    s.remove_field("keep")
    assert not s.has_child("keep")

    # Removing non-existing raises
    with pytest.raises(KeyError, match="Cannot delete field 'missing'"):
        s.remove_field("missing")

    # Remove all occurrences of 'dup' in nested groups
    assert s.get_num_occurrences("dup") == 2
    s.remove_all_occurrences("dup")
    assert s.get_num_occurrences("dup") == 0


def test_is_data_field_array_and_is_data_group_field_array():
    # Data field array
    arr = SampleDataGroup.create_data_field_array(DALIDataType.FLOAT, num_fields=3)
    s = SampleDataGroup()
    s.add_data_group_field("arr", arr)
    assert s.is_data_field_array("arr")
    assert not s.is_data_group_field_array("arr")

    # Data group field array
    elem = SampleDataGroup()
    elem.add_data_field("x", DALIDataType.INT32)
    s.add_data_group_field_array("grp_arr", elem, 2)
    assert s.is_data_group_field_array("grp_arr")
    assert not s.is_data_field_array("grp_arr")


def test_get_numpy_type_for_dali_type_success_and_error():
    # Known mapping
    np_type = SampleDataGroup.get_numpy_type_for_dali_type(DALIDataType.FLOAT)
    assert np_type == np.float32

    # Unknown mapping should raise
    with pytest.raises(ValueError, match="does not have a corresponding numpy type"):
        SampleDataGroup.get_numpy_type_for_dali_type(DALIDataType.STRING)


def test_getitem_setitem_errors_on_nonexistent_fields():
    s = SampleDataGroup()
    # Define a valid field to ensure class is initialized
    s.add_data_field("x", DALIDataType.FLOAT)

    # __getitem__ on non-existent should raise KeyError
    with pytest.raises(KeyError, match="No field with name 'missing'"):
        _ = s["missing"]

    # __setitem__ on non-existent should raise KeyError
    with pytest.raises(KeyError, match="No field with name 'missing'"):
        s["missing"] = 1.0

    # __setitem__ on existing should succeed and convert
    s["x"] = 1
    assert isinstance(s["x"], np.ndarray)
    np.testing.assert_array_equal(s["x"], np.array(1.0, dtype=np.float32))


if __name__ == "__main__":
    pytest.main([__file__])
