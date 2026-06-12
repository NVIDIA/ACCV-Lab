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

import copy
import numbers
import warnings

from typing import Union, Tuple, List, Dict, Any, Sequence, Optional

import numpy as np
import cupy
import torch

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types

from ..internal_helpers import check_type, get_mapped


class SampleDataGroup:
    '''Structured container for sample data. Can also be used as a blueprint to describe the data format.

    Data is organized as a tree containing:

      - **Data fields**: Leaf nodes that hold the actual data.
      - **Data group fields**: Non-leaf nodes that group related items.

    Example:

        An example for accessing the data field ``"bounding_boxes"`` inside nested data group fields
        ``"camera"`` and ``"annotations"``:

        >>> bounding_boxes = data["camera"]["annotations"]["bounding_boxes"]

        Note that accessing the data is done as for a nested dictionary. Here, the data group fields are
        analogous to :class:`dict` objects and data fields correspond to the actual stored values at
        the leaves.

    Capabilities (see individual method docs for details):

      - Enforce a predefined data format (field names, order, and types). Format changes need to be performed
        explicitly.
      - Inside the input callable/iterable and outside the DALI pipeline the following can be performed (both
        can be disabled):

        - Apply automatic type conversions (e.g., integers to floats) on assignment
        - Apply optional custom string-to-numeric mappings on assignment for selected fields (see
          :meth:`add_data_field`, :meth:`add_data_field_array`, and :meth:`set_apply_mapping`).
      - Inside the pipeline: Apply automatic type checks on assignment.
      - Render the tree in a human-readable form via ``print(obj)``.
      - Flatten values to a sequence and reconstruct from a sequence (see :meth:`get_data`, :meth:`set_data`,
        and :meth:`set_data_from_dali_generic_iterator_output`).
        This is useful when passing the data from the input callable/iterable to the pipeline, and when
        returning data from the pipeline, as nested data structures are not supported there. Also
        see :class:`DALIStructuredOutputIterator` for an output iterator which re-assembles the data from
        the flattened output into a :class:`SampleDataGroup` instance or nested dictionaries before returning
        it.
      - Compare formats of two instances (see :meth:`type_matches`). This also ensures that the flattened
        data obtained from one instance can be used to fill the data of another instance.
      - Utilities that facilitate implementation of pipeline steps: find/remove all occurrences of fields
        with a given name, add/remove/change fields and types, etc. (e.g. see
        :meth:`find_all_occurrences`). Note that the search is performed at DALI graph construction time, so
        there is no overhead during the pipeline execution.
      - Supports passing strings through the DALI pipeline and obtaining them as strings in the pipeline
        output. Note that strings are not supported inside the DALI pipeline. They can be accessed/assigned
        as strings in the input callable/iterable and outside the DALI pipeline, but appear as uint8 tensors
        inside the pipeline itself (alternative: use a mapping to numeric values as described above).

    Usage modes:

      - **Blueprint**: describes the data format (fields and types) but contains no values. This allows
        inferring downstream formats without running data processing (e.g., to initialize a DALI iterator).
        When only passing of flattened data is possible, a blueprint can be filled from flattened values
        (see :meth:`get_data`, :meth:`set_data`).
      - **Container**: holds actual values. When accessing the data, behaves similarly to a nested dictionary.
        When assigning data, additional checks/conversions are potentially performed.


    Important:
        **Assigning a Field Value**

        Assignment means using the indexed assignment operator ``obj[name] = value`` or the
        method ``obj.set_item_in_path(path, value)``.

        When assigning data fields, the following holds:

          - Mappings and conversions will be performed on assignment (inside the input callable/iterable
            and outside the DALI pipeline; if not disabled). Inside the DALI pipeline itself, no mapping
            or conversion is applied.
          - Inside the DALI pipeline, type checks are performed instead on assignment and an error is
            raised if the type is not correct.
          - Assigning strings is only supported in the input callable/iterable and outside the DALI
            pipeline. String fields are handled as uint8 tensors inside the DALI pipeline.

        When assigning to data group fields, the following holds:

          - The assignment succeeds only if the new value's format matches the previous format,
            i.e. if ``obj[name].type_matches(value)`` holds. Otherwise, a :class:`KeyError` is raised.
            This is done to prevent changing the data format implicitly by assigning a different type.
          - If the type needs to be changed, this needs to be done explicitly first (e.g., using
            :meth:`change_type_of_data_and_remove_data`).

    Important:
        **Getting a Field Value**

        Getting a field value means using the indexed access operator ``obj[name]`` or the method
        ``obj.get_item_in_path(path)``.

        Accessing strings inside the DALI pipeline (except for the input callable/iterable) will return
        the underlying uint8 tensor instead. Using strings directly is only supported in the input
        callable/iterable and outside the DALI pipeline.

    Important:
        **Changing the Data Format**

        Changing the data format is always explicit. For example, adding a field and assigning values is a
        two-step process: create the field first, then assign data. When defining a blueprint, fields are
        created but left empty.

    Important:
        **Type Checking**

        Type checking is performed on assignment to ensure that the data type is correct (inside the DALI
        pipeline). This is useful when developing the pipeline/processing step, but adds some overhead. Type
        checking is enabled by default (see :meth:`set_do_check_type`).

    Note:
        Additional information:

           - When converting a :class:`SampleDataGroup` to a string (e.g., using ``print(obj)``), the data format
             as well as some details (e.g., for which fields a mapping is defined, which fields are empty,
             data types of the fields) are printed. The actual stored values are not printed.
             For a more simple output, see :meth:`get_string_no_details`.
           - When obtaining the length of a :class:`SampleDataGroup` (e.g., using ``len(obj)``), the number of
             direct children (data fields and data group fields) is returned.
    '''

    _type_mapping = {
        types.DALIDataType.BOOL: bool,
        types.DALIDataType.FLOAT: np.float32,
        types.DALIDataType.FLOAT16: np.float16,
        types.DALIDataType.FLOAT64: np.float64,
        types.DALIDataType.INT8: np.int8,
        types.DALIDataType.INT16: np.int16,
        types.DALIDataType.INT32: np.int32,
        types.DALIDataType.INT64: np.int64,
        types.DALIDataType.UINT8: np.uint8,
        types.DALIDataType.UINT16: np.uint16,
        types.DALIDataType.UINT32: np.uint32,
        types.DALIDataType.UINT64: np.uint64,
    }

    def __init__(self):
        self._mappings = {}
        self._value_order = tuple()
        self._types_order = tuple()
        self._values = {}
        self._types = {}
        for i in range(len(self._value_order)):
            val = self._value_order[i]
            self._values[val] = None
            self._types[val] = self._types_order[i]
        self._do_apply_mapping = True
        self._do_convert = True
        self._do_check_type = True

    @staticmethod
    def create_data_field_array(
        type: types.DALIDataType,
        num_fields: int,
        mapping: Optional[Dict[Union[str, None], Union[int, float, np.number, bool]]] = None,
    ) -> SampleDataGroup:
        '''Create a :class:`SampleDataGroup` containing multiple data fields of the same type.

        The data fields have numerical (integer) names in the range ``[0; num_fields - 1]``. This means that
        the returned :class:`SampleDataGroup` behaves as an array of data fields.

        Args:
            type: Type of the fields to add
            num_fields: Number of fields to add to the array data group field
            mapping: Optional mapping for the fields (see :meth:`add_data_field` for details on mappings).

        See also:
            :meth:`create_data_group_field_array`
            :meth:`add_data_field_array`
            :meth:`add_data_group_field_array`

        Returns:
            Resulting array :class:`SampleDataGroup` object
        '''

        res = SampleDataGroup()
        for i in range(num_fields):
            res.add_data_field(i, type, mapping)
        return res

    @staticmethod
    def create_data_group_field_array(sample_data_group: SampleDataGroup, num_fields: int) -> SampleDataGroup:
        '''Create a :class:`SampleDataGroup` containing multiple data group fields (themselves :class:`SampleDataGroup` instances).

        Note that the created data group fields will be initialized as blueprints, i.e. they will not contain
        any actual data even if ``sample_data_group`` does. This is done to cleanly separate
        this step (defining the data format) from actually filling the data.

        Args:
            sample_data_group: Blueprint representing the element format. Any actual data present in
                ``sample_data_group`` will be ignored; the resulting elements will be empty of data.
            num_fields: Number of fields to create

        See also:
            :meth:`create_data_field_array`
            :meth:`add_data_field_array`
            :meth:`add_data_group_field_array`

        Returns:
            Resulting array :class:`SampleDataGroup` object
        '''

        res = SampleDataGroup()
        for i in range(num_fields):
            res.add_data_group_field(i, sample_data_group)
        return res

    def set_apply_mapping(self, apply: bool):
        '''Set whether to apply string to numeric mapping (for data fields where such a mapping is defined).

        This setting will be propagated to descendants (data group fields) of the data group field for which
        it is called.

        Note:
            The mapping is applied in the input callable/iterable and outside the DALI pipeline.
            Inside the DALI pipeline itself, the mapping is not applied. If apply mapping is set to ``True``
            and an assignment is performed inside the pipeline, a warning will be issued, and the assignment
            will be performed without mapping (if it is already in the correct format; an error will be
            raised if the format is not correct).

        Args:
            apply: Whether to apply the mapping (for fields where a mapping is set).
        '''
        self._do_apply_mapping = apply
        # Also set (recursively) in SampleDataGroup elements
        for type, name in zip(self._types_order, self._value_order):
            if type == SampleDataGroup:
                self[name].set_apply_mapping(apply)

    def set_do_convert(self, convert: bool):
        '''Set whether to convert data in the data fields to the types set up when creating those fields.

        This setting will be propagated to descendants (data group fields) of the data group field for which
        it is called.

        Note:
            The conversion is applied in the input callable/iterable and outside the DALI pipeline.
            Inside the DALI pipeline itself, the conversion is not applied. Instead, type checks are performed
            (regardless of this setting).

        Args:
            convert: Whether to perform automatic type conversions (e.g., integers to floats) on assignment.
        '''
        self._do_convert = convert
        # Also set (recursively) in SampleDataGroup elements
        for type, name in zip(self._types_order, self._value_order):
            if type == SampleDataGroup:
                self[name].set_do_convert(convert)

    def set_do_check_type(self, check_type: bool):
        '''Set whether to perform type checking on assignment.

        This setting will be propagated to descendants (data group fields) of the data group field for which
        it is called.

        Note:
            The type checking is useful when developing the pipeline/processing step, but adds some overhead.
            Therefore, it is advisable to disable it in production.

        Args:
            check_type: Whether to perform type checking (in the DALI pipeline) on assignment.
        '''
        self._do_check_type = check_type
        # Also set (recursively) in SampleDataGroup elements
        for type, name in zip(self._types_order, self._value_order):
            if type == SampleDataGroup:
                self[name].set_do_check_type(check_type)

    def get_empty_like_self(self) -> SampleDataGroup:
        '''Get an object with the same structure (same nested data group fields and data fields), but no values.

        Obtain a blueprint either from another blueprint or from a populated object (ignoring values and
        initializing all data fields as empty). This can be regarded as a deep-copy of the original object,
        but with the actual data removed.

        Returns:
            Resulting blueprint :class:`SampleDataGroup` object.
        '''

        res = self._get_copy_except_values()
        # Values should be empty (except which are SampleDataGroups themselves, see next step), but the
        # correct fields should be set up and filled with 'None'
        res._values = {}
        for key in self._values:
            if res._types[key] == SampleDataGroup:
                # If the element is itself a SampleDataGroup, ensure we get the correct empty like the element
                res._values[key] = self._values[key].get_empty_like_self()
            else:
                # If the element is a primitive, set it to None
                res._values[key] = None
        return res

    def get_copy(self) -> SampleDataGroup:
        '''Get a copy.

        Create a copy: equivalent to :meth:`get_empty_like_self` followed by filling the data from the
        original object. Note that for the actual data, references to the original data are used, i.e. the
        data itself is not deep-copied. However, the data group fields making up the data format are
        deep-copied.

        This means that modifying the data in place will modify the data in the original. However, assigning
        new data to fields, adding or deleting fields, changing their type etc. will not affect the original.

        Returns:
            Resulting copy
        '''
        res = self._get_copy_except_values()
        res._values = {}
        # Values should be re-used. However, the individual values should be not deep-copied.
        # Instead, build a new dictionary (so that adding or removing keys will not affect the original),
        # but use references to the original objects for the actual values except for SampleDataGroup
        # elements, which are handled differently (see below)
        for key in self._values:
            if res._types[key] == SampleDataGroup:
                # If the element is itself a SampleDataGroup, ensure it is copied correctly by calling
                # get_copy recursively.
                res._values[key] = self._values[key].get_copy()
            else:
                # If the element is not a SampleDataGroup, set reference to the original object.
                res._values[key] = self._values[key]
        return res

    def type_matches(self, other: SampleDataGroup) -> bool:
        '''Check whether the data type defined by two objects of :class:`SampleDataGroup` is the same.

        The following is not considered when checking for equality as it is not considered to be part of the
        type described by the object:

          - The actual data stored in the data fields
          - Whether mapping and conversion should be performed
          - Whether mappings are available for the same fields and whether mappings themselves are the same

        Important:
            Note that it is checked whether the fields appear in the same order in the two objects.
            This is the case if the objects are constructed from the same blueprint (or if they were
            constructed by adding the individual fields in the same order). This is important as it defines
            whether the flattened data, e.g. obtained by :meth:`get_data` from one of the objects can be used
            to fill the data into the other one, e.g. using :meth:`set_data`.
        '''
        match = self._value_order == other._value_order

        if match:
            for type, name, i in zip(self._types_order, self._value_order, range(len(self))):
                if type == SampleDataGroup:
                    match = self[name].type_matches(other[name])
                else:
                    match = type == other._types_order[i]
                if not match:
                    break

        return match

    def __setitem__(self, name: Union[str, int], value: Any):
        # Documented as part of the class docstring

        assert isinstance(name, str) or isinstance(name, int), f"'name' has unsupported type: `{type(name)}`"
        if not name in self._values:
            raise KeyError(f"No field with name '{name}'")

        if self._types[name] == SampleDataGroup:
            if (self._types[name] == SampleDataGroup) and (not self[name].type_matches(value)):
                raise KeyError(
                    f"Tried to set a data group field '{name}' "
                    f"(fields of type SampleDataGroup), but types do not match."
                )

        if self._types[name] == types.DALIDataType.STRING and not self._is_dali_data_node(value):
            self._values[name] = self._convert_from_string(value)
        else:
            self._values[name] = self._apply_mapping_check_and_convert(name, value)

    def set_item_in_path(
        self, path: Union[str, int, Tuple[Union[str, int]], List[Union[str, int]]], value: Any
    ):
        '''Assign a field value at a (nested) path.

        The path is a sequence of field names/keys. For example, if the path is
        ``path = ("name_1", "name_2", "name_3")``, the following are equivalent:

          - ``obj.set_item_in_path(path, value_to_set)``
          - ``obj["name_1"]["name_2"]["name_3"] = value_to_set``

        Important:
            See the class docstring for details on the assignment behavior.

        Args:
            path: Path of the item to be set.
            value: Value to be set.
        '''
        assert (
            isinstance(path, str)
            or isinstance(path, int)
            or isinstance(path, tuple)
            or isinstance(path, list)
        ), "'path' has unsupported type"
        if isinstance(path, list) or isinstance(path, tuple):
            assert (
                len(path) > 0
            ), "Only setting of children is supported. Therefore, 'path' cannot be a tuple/list with length 0."
            if not path[0] in self._values:
                raise KeyError(f"No field with name '{path[0]}'")
            if len(path) == 1:
                # The path is a tuple/list, but has only 1 entry. This is equivalent to using a string/number
                # directly
                self[path[0]] = value
            else:
                # (recursively) use set_item_in_path() and walk the path, until the remaining path is a
                # single name
                self._values[path[0]].set_item_in_path(path[1:], value)
        else:
            self[path] = value

    def __getitem__(self, name: Union[str, int]) -> Any:
        # Documented as part of the class docstring

        assert isinstance(name, str) or isinstance(name, int), "'name' has unsupported type"

        if not name in self._values:
            raise KeyError(f"No field with name '{name}'")

        value = self._values[name]
        if self._types[name] == types.DALIDataType.STRING and not self._is_dali_data_node(value):
            return self._convert_to_string(value)
        return value

    def get_item_in_path(self, path: Union[str, int, Tuple[Union[str, int]], List[Union[str, int]]]) -> Any:
        '''Get a field value at a nested path.

        The path is a sequence of field names/keys. For example, if ``path = ("name_1", "name_2", "name_3")``,
        the following are equivalent:

          - ``value = obj.get_item_in_path(path)``
          - ``value = obj["name_1"]["name_2"]["name_3"]``

        Note:
            Accessing strings inside the DALI pipeline (except for the input callable/iterable) will return
            the underlying uint8 tensor instead. Using strings directly is only supported in the input
            callable/iterable and outside the DALI pipeline.

        Args:
            path: Path of the item to get.

        Returns:
            Item at ``path``.
        '''
        assert (
            isinstance(path, str)
            or isinstance(path, int)
            or isinstance(path, tuple)
            or isinstance(path, list)
        ), "'path' has unsupported type"
        if isinstance(path, list) or isinstance(path, tuple):
            if len(path) == 0:
                return self
            if not path[0] in self._values:
                raise KeyError(f"No field with name '{path[0]}'")
            if len(path) == 1:
                # The path is a tuple/list, but has only 1 entry. This is equivalent to using a string/number
                # directly
                return self[path[0]]
            else:
                # (recursively) use get_item_in_path() and walk the path, until the remaining path is a
                # single name
                return self._values[path[0]].get_item_in_path(path[1:])
        else:
            return self[path]

    def get_parent_of_path(
        self, path: Union[int, str, Tuple[Union[str, int]], List[Union[str, int]]]
    ) -> SampleDataGroup:
        '''Get the parent of an element described in path.

        The following are equivalent:
          - ``obj.get_parent_of_path(path)``
          - ``obj.get_item_in_path(path[:-1])``

        Note:
            As a parent node cannot be a data field (i.e. a leaf node), the returned value is always a
            :class:`SampleDataGroup` instance.

        Args:
            path: Path for which to get the parent.

        Returns:
            Parent of the path.
        '''

        if self.path_is_single_name(path):
            if not self.has_child(path):
                raise KeyError(f"No element '{path}' is present.")
            return self
        else:
            assert len(path) > 0, (
                "Cannot get parent of element with path len 0, as path len 0 corresponds to the element for "
                "which the method was called."
            )
            return self.get_item_in_path(path[:-1])

    def get_type_of_item_in_path(
        self, path: Union[Tuple[Union[str, int]], List[Union[str, int]]]
    ) -> Union[types.DALIDataType, type]:
        '''Get the type of the item at a nested path.

        Args:
            path: Path to the item.

        See also:
          - :meth:`SampleDataGroup.get_item_in_path` for a description of the `path` parameter.
          - :meth:`SampleDataGroup.get_type_of_field` for a description of how type information is returned
            (which applies to this method as well).

        Returns:
            Data type of the field. For data group fields, :class:`SampleDataGroup`. For data fields, the
            corresponding :class:`nvidia.dali.types.DALIDataType`. If ``path`` is empty, returns
            ``self``.

        '''
        assert isinstance(path, tuple) or isinstance(path, list), "'path' has to be tuple or list"
        if len(path) > 0:
            if len(path) > 1:
                to_check_in = self.get_parent_of_path(path)
            else:
                to_check_in = self
            res_type = to_check_in.get_type_of_field(path[-1])
        else:
            # Path is refering to `self`. As this is a SampleDataGroup, the data type is `SampleDataGroup`
            res_type = SampleDataGroup
        return res_type

    @staticmethod
    def path_is_single_name(path: Union[str, int, Tuple[Union[str, int]], List[Union[str, int]]]) -> bool:
        '''Check if the path given is a single name.

        Args:
            path: Path to check. Can be a single name/key or a sequence of names.

        Returns:
            ``True`` if ``path`` is a single name/key (i.e., a string or integer, not a sequence),
            ``False`` otherwise.
        '''
        is_name = isinstance(path, str) or not isinstance(path, Sequence)
        return is_name

    def path_exists(self, path: Union[str, int, Tuple[Union[str, int]], List[Union[str, int]]]) -> bool:
        '''Check if a field with the given path exists.

        Args:
            path: Path to check.

        Returns:
            Whether field with given path exists.
        '''

        if self.path_is_single_name(path):
            exists = self.has_child(path)
        elif len(path) == 0:
            exists = True
        elif len(path) == 1:
            exists = self.has_child(path[0])
        else:
            remaining_path = path[1:]
            if len(remaining_path) == 1:
                remaining_path = remaining_path[0]
            exists = self[path[0]].path_exists(remaining_path)
        return exists

    def path_exists_and_is_data_group_field(
        self, path: Union[str, int, Tuple[Union[str, int]], List[Union[str, int]]]
    ) -> bool:
        '''Check if a field with the given path exists and is a data group field.

        Args:
            path: Path to check

        Returns:
            ``True`` if field at path exists and is a data group field, ``False`` otherwise
        '''

        exists = self.path_exists(path)
        if exists:
            # Path is referring to 'self', which is a SampleDataGroup and therefore a data group field
            if len(path) == 0:
                res = True
            else:
                # Get the last name in the path
                if self.path_is_single_name(path):
                    last_name = path
                else:
                    last_name = path[-1]
                # Get the parent element of the path (is a SampleDataGroup, as it contains children and
                # therefore is a data group field)
                parent = self.get_parent_of_path(path)
                # Check whether the element with name `last_name` is a data group field using its parent
                # node (as the node itself may be a data node).
                res = parent.is_data_group_field(last_name)
        return res

    def get_type_of_field(self, name: Union[str, int]) -> Union[types.DALIDataType, type]:
        '''Get type of a field.

        The type is either expressed as a :class:`nvidia.dali.types.DALIDataType` (data fields) or
        :class:`SampleDataGroup` (data group fields).

        Args:
            name: Name of the field.

        Returns:
            Type of the field. For string fields this returns
            :class:`nvidia.dali.types.DALIDataType.STRING`. Note that this is different from flattened
            contexts (e.g., :attr:`field_types_flat`), where strings are represented as
            :class:`nvidia.dali.types.DALIDataType.UINT8`. This is as the flattened data is used internally
            to pass data between :class:`SampleDataGroup` objects where the object itself cannot be passed and
            consequently, the string data is passed as stored internally (i.e. the underlying uint8 tensors).
            Here, the actual type as configured (e.g. by :meth:`add_data_field`) is returned.

        '''
        return self._types[name]

    def get_string_no_details(self) -> str:
        '''Get string representing the :class:`SampleDataGroup` instance, omitting details.

        Omits per-field details such as whether a value is set and whether a mapping is available.
        '''
        res_str = "{\n" + self._to_string_with_indent(2, False) + "}\n"
        return res_str

    def __str__(self) -> str:
        # Documented as part of the class docstring
        res_str = "{\n" + self._to_string_with_indent(2, True) + "}\n"
        return res_str

    def __len__(self) -> int:
        # Documented as part of the class docstring
        return len(self._value_order)

    def is_array(self, field: Optional[Union[str, int]] = None) -> bool:
        '''Check whether (self or child) object can be regarded as an array.

        This is the case if all of the following hold:
          - The field names have integer numeric names.
          - Each element in the range ``[0; len(self) - 1]`` is present as a name.
          - The value order is such that for each element, the name increases by 1, i.e.
            ``self.contained_top_level_field_names == (0, 1, 2, 3, ...)``.

        Args:
            field: If set, perform the check for the named child. Otherwise, check ``self``.

        Returns:
            Whether the object can be considered an array.
        '''
        if field is None:
            for i in range(len(self)):
                if not self._value_order[i] == i:
                    return False
            return True
        else:
            return self[field].is_array()

    def is_data_field_array(self, field: Optional[Union[str, int]] = None) -> bool:
        """Check whether (self or child) object is an array whose elements are all data fields (no data group fields).

        See documentation of :meth:`is_array` for conditions for a data group field to be regarded as an
        array.

        Args:
            field: If set, perform the check for the named child. Otherwise, check ``self``.

        Returns:
            Whether the object is an array of data fields.
        """
        if field is None:
            for i in range(len(self)):
                if not self._value_order[i] == i:
                    return False
                if not self.is_data_field(i):
                    return False
            return True
        else:
            if not self.is_data_group_field(field):
                return False
            else:
                return self[field].is_data_field_array()

    def is_data_group_field_array(self, field: Optional[Union[str, int]] = None) -> bool:
        """Check whether (self or child) object is an array whose elements are all data group fields (no data fields).

        See documentation of :meth:`is_array` for conditions for a data group field to be regarded as an
        array.

        Args:
            field: If set, perform the check for the named child. Otherwise, check ``self``.

        Returns:
            Whether the object is an array of data group fields.
        """
        if field is None:
            for i in range(len(self)):
                if not self._value_order[i] == i:
                    return False
                if not self.is_data_group_field(i):
                    return False
            return True
        else:
            return self[field].is_data_group_field_array()

    @property
    def contained_top_level_field_names(self) -> Tuple[Union[str, int]]:
        """Get the names of the contained top-level fields.

        The order of the fields corresponds to the order in which they were added.

        Returns:
            Names of contained fields.
        """
        return self._value_order

    @property
    def field_top_level_types(self) -> Tuple[Union[types.DALIDataType, type]]:
        """Types of the top-level fields.

        The order of the fields corresponds to the order in which they were added (and to the order
        of the elements returned by :attr:`contained_top_level_field_names`).

        Types fields are :class:`nvidia.dali.types.DALIDataType` instances for data
        fields and :class:`SampleDataGroup` blueprints for data group fields.
        """
        return self._types_order

    @property
    def field_names_flat(self) -> Tuple[str]:
        """Names of contained data fields flattened (all leaf nodes, not only direct children).

        Each element corresponds to a data field (leaf node). Original nesting is reflected in the names
        (concatenated with "." between parent and child). Numerical names are converted to strings to ensure
        that they can be used as names in other places (e.g. DALI generic iterator). For example, the numeric
        name ``5`` would become ``"[5]"``. For example, if there is a data field in the original object in the
        path ``object["name_0"][1]["name_2"]``, the name used in the flattened tuple of names would be
        ``"name_0.[1].name_2"``.

        The order of the elements corresponds to the order used in :meth:`get_data`, so that the names
        obtained here correspond to the values obtained there.

        No names are added for data group fields themselves. If they contain descendants which are data
        fields, their name will appear in the name of the descendants (before "."). However, if a data
        group field does not contain any data field descendants, it will not contribute a name to the output.

        Note:
            The names themselves reflect the hierarchy of the data, so that the names are unique, even
            if there are multiple fields with the same name in the structure.
        """
        res = tuple(self._get_contained_field_names_flat(""))
        return res

    @property
    def field_types_flat(self) -> Tuple[types.DALIDataType]:
        """Types of contained data fields flattened (all leaf nodes, not only direct children).

        Each element corresponds to a leaf node.

        The order of the elements corresponds to the order used in :meth:`get_data`, so that the types
        obtained here correspond to the values obtained there.

        No types are added for data group fields themselves. If they contain descendants which are data
        fields, the types of these descendants will be added. However, if a data group field does not
        contain any data field descendants, it will not contribute a type to the output.

        Note:
            As only the leaf nodes containing data are considered, no entries directly corresponding to data
            group fields will be added.

            String fields are represented as :class:`nvidia.dali.types.DALIDataType.UINT8`, matching their
            in-pipeline representation. Note that this is different from e.g. :meth:`get_type_of_field`,
            but consistent with :meth:`get_data` (see :meth:`get_data` for details on the rationale).
        """
        res = tuple(self._get_contained_types_flat())
        return res

    def get_data(self, as_list_type: bool = False) -> Union[tuple, list]:
        """Get values of all data fields as a flattened sequence (all leaf nodes, not only direct children).

        The order of the elements is the order of a depth-first traversal with the order of the children at
        each node corresponding to the order in which the elements were added (consistent with, e.g.,
        :attr:`contained_top_level_field_names`). The order is the same as in
        :attr:`field_names_flat` and :attr:`field_types_flat`, so that these can be used to obtain
        information about the individual elements of the obtained sequence of values. Only data fields
        (leaf nodes that are not :class:`SampleDataGroup`) contribute values. Data group fields are not
        included directly, but their data field descendants contribute values.

        Note:
            The tuple returned by this function can be used directly to
              - Pass parameters from an input callable/iterable to the DALI pipeline.
              - Return the final output of the DALI pipeline.

            In these cases, the returned sequence can be used to fill the original data structure
            (using :meth:`set_data` or :meth:`set_data_from_dali_generic_iterator_output`) into a
            :class:`SampleDataGroup` blueprint object with the same format as ``self``.

        Important:
            For string data fields, the values are the underlying uint8 arrays/tensors (or DataNodes), not
            Python ``str`` objects (both inside and outside the DALI pipeline). This method is designed to
            exchange data between :class:`SampleDataGroup` objects and directly returns the underlying data,
            with the encoded strings. The conversion to Python ``str`` objects is performed when the data is
            obtained, e.g. using the indexed access operator ``[]`` or :meth:`get_item_in_path`.

        Args:
            as_list_type: If ``True``, return a list (tuple otherwise).

        Returns:
            Sequence of values of all data fields.
        """
        res = []
        for type, name in zip(self.field_top_level_types, self.contained_top_level_field_names):
            if type == SampleDataGroup:
                res_i = self[name].get_data(True)
                res = res + res_i
            else:
                res.append(self._getitem_without_conversions(name))

        if not as_list_type:
            res = tuple(res)

        return res

    def set_data(self, data: Union[tuple, list]):
        '''Set values of all descendant data fields from a flattened sequence.

        The sequence needs to contain the data in the same order as indicated by
        :attr:`field_names_flat`. If the flat data was obtained by :meth:`get_data` from a
        :class:`SampleDataGroup` object with the same data format as ``self``, this will always be the case.
        The compatibility between the object from which the flattened data was obtained and this instance can
        be checked with :meth:`type_matches`.

        Important:
            When setting data in this way, no conversions or mappings are applied (both inside and outside
            the DALI pipeline). This method is designed to exchange data between :class:`SampleDataGroup`
            objects and expects the data as stored in the :class:`SampleDataGroup` object (i.e., already
            converted and with mappings applied) as input.

        Args:
            data: Flat sequence of values to use.
        '''
        self._set_data_and_get_num_used_data_elements(data)

    def set_data_from_dali_generic_iterator_output(self, data: List[Dict[str, Any]], index: int):
        '''Set values from the output of a DALI generic iterator.

        The DALI generic iterator refers to :class:`nvidia.dali.plugin.pytorch.DALIGenericIterator` or
        any other iterator which follows the same interface (tensor types may be from a different framework).

        The iterator (and therefore, the underlying DALI pipeline) must output the flattened data in the
        format as this instance (using :meth:`get_data`), with names assigned in the iterator to the
        individual fields matching :attr:`field_names_flat` of this object. The compatibility
        between the object from which the flattened data was obtained and this instance can be checked
        with :meth:`type_matches`.

        See also:
            :meth:`get_like_self_filled_from_iterator_output`

        Note:
            Values for string fields are uint8 arrays/tensors (not Python strings). For details, see
            :meth:`get_data`.

        Args:
            data: Output of the DALI generic iterator.
            index: Index inside data from which to fill the data.
        '''
        name_order = self.field_names_flat
        data_as_sequence = [data[index][name] for name in name_order]
        self.set_data(data_as_sequence)

    def has_child(self, name: Union[str, int]) -> bool:
        '''Check whether a direct child with the given name exists.

        Args:
            name: Name of the child to check

        Returns:
            Whether child exists.
        '''
        res = name in self._values
        return res

    def add_data_field(
        self,
        name: Union[str, int],
        type: types.DALIDataType,
        mapping: Optional[Union[Dict[Union[str, None], Union[int, float, np.number, bool]]]] = None,
    ):
        '''Add a data field as a direct child.

        Data field means that the field contains actual data, i.e. is not another data group field
        (:class:`SampleDataGroup` instance).

        Note:
            If a mapping is defined, it is applied both to strings and to (possibly nested, multi-dimensional)
            sequences of strings (lists/tuples/arrays). The mapping is a dictionary from original string
            values to numeric values. The special key ``None`` provides a default value for unmatched inputs.

            The mapping is only applied when data is assigned inside the input callable/iterable or
            outside the DALI pipeline. The mapping is not performed for assignments inside the actual
            DALI pipeline (and setting data there is only supported directly using numerical values).

        Note:
            Alternatively to using a mapping, strings can be directly assigned to data fields
            by setting the data type to :class:`nvidia.dali.types.DALIDataType.STRING`. However,

              - String processing in this way is only supported inside the input callable/iterable and
                outside the DALI pipeline, and such strings appear as uint8 tensors inside the DALI
                pipeline.
              - Only single strings can be assigned, not sequences of strings (although outputting
                1D sequences of strings is supported to enable output of batch-wise data).
              - Often, using a mapping is advantageous to meaningfully process the data in the pipeline
                and also needs to be performed for other reasons (e.g. to convert class labels from
                strings to integers to be used in the loss computation).

            This way of handling strings is e.g. useful to pass sample tags or other high-level
            descriptors through the pipeline.

        Args:
            name: Name of the field to add
            type: Type of (the elements of) the field to add. If a mapping is used, this is the type after
                mapping is applied.
            mapping: Mapping defining the mapping from input string values to numerical values. The conversion
                from string to numeric happens at data assignment (if applying mapping is not disabled).
                ``None`` can be added as a key to the mapping. In this case, the respective value is used if
                the input string(s) do not match any of the other keys. Mapping is applied both if a single
                string is assigned, but also for (n-dimensional) sequences of strings. Note that if a mapping
                is set, numeric values can still be assigned directly to the data field alternatively to
                strings.
        '''
        assert not isinstance(type, SampleDataGroup), (
            "The method add_data_field() cannot be used to add data group fields (type: SampleDataGroup). "
            "Use add_data_group_field() instead."
        )
        assert (
            type != types.DALIDataType.STRING or mapping is None
        ), "Cannot set a mapping for data fields of type types.DALIDataType.STRING"

        if name in self._value_order:
            raise KeyError(f"Field '{name}' cannot be added as it already exists.")
        self._value_order = self._value_order + (name,)
        self._types_order = self._types_order + (type,)
        self._values[name] = None
        self._types[name] = type
        if mapping is not None:
            self._mappings[name] = mapping

    def add_data_group_field(self, name: str, blueprint_sample_data_group: SampleDataGroup):
        '''Add a data group field as a direct child.

        Data group field means a child of the type :class:`SampleDataGroup`, which itself can contain data
        fields and/or data group fields. Data group fields are used to group elements together logically.

        ``blueprint_sample_data_group`` acts as a blueprint. A new empty instance with the same format is
        created and added as the child. Values can be assigned later directly (or via
        :meth:`set_item_in_path`).

        Args:
            name: Name of the new field.
            blueprint_sample_data_group: :class:`SampleDataGroup` instance describing the field format to add.
        '''

        if name in self._value_order:
            raise KeyError(f"Field '{name}' cannot be added as it already exists.")
        self._value_order = self._value_order + (name,)
        self._types_order = self._types_order + (SampleDataGroup,)
        to_add = blueprint_sample_data_group.get_empty_like_self()
        to_add.set_apply_mapping(self._do_apply_mapping)
        to_add.set_do_convert(self._do_convert)
        self._values[name] = to_add
        self._types[name] = SampleDataGroup

    def add_data_field_array(
        self,
        name: str,
        type: types.DALIDataType,
        num_fields: int,
        mapping: Optional[Dict[Union[str, None], Union[int, float, np.number, bool]]] = None,
    ):
        '''Add a data field array.

        Add a child data group field (type :class:`SampleDataGroup`) that contains ``num_fields`` elements,
        each with the type and mapping defined here. Elements are added with integer names from ``0`` to
        ``num_fields - 1``, so the child behaves like an array.

        Note:
            If a blueprint of the array is already created as another, independent  blueprint, you can use
            :meth:`add_data_group_field` to add the blueprint to this object.

        See also:
            :meth:`add_data_group_field_array`
            :meth:`create_data_field_array`
            :meth:`create_data_group_field_array`

        Args:
            name: Name of the array data group field to add
            type: Type of the fields to add to the array data group field
            num_fields: Number of fields to add to the array data group field
            mapping: Optional mapping for the fields (see :meth:`add_data_field` for details on mappings).
        '''

        data_group_to_add = self.create_data_field_array(type, num_fields, mapping)
        self.add_data_group_field(name, data_group_to_add)

    def add_data_group_field_array(
        self, name: str, blueprint_sample_data_group: SampleDataGroup, num_fields: int
    ):
        '''Add a data group field array.

        Add a child data group field (type :class:`SampleDataGroup`) that contains ``num_fields`` elements,
        each matching the provided blueprint. Elements are added with integer names from ``0`` to
        ``num_fields - 1`` so the child behaves like an array.

        Note:
            If a blueprint of the array is already created as another, independent  blueprint, you can use
            :meth:`add_data_group_field` to add the blueprint to this object.

        See also:
            :meth:`add_data_field_array`
            :meth:`create_data_field_array`
            :meth:`create_data_group_field_array`

        Args:
            name: Name of the array data group field to add
            blueprint_sample_data_group: :class:`SampleDataGroup` describing the element format (each
                element is initialized from ``get_empty_like_self()`` of the blueprint).
            num_fields: Number of elements to add.
        '''
        data_group_to_add = self.create_data_group_field_array(blueprint_sample_data_group, num_fields)
        self.add_data_group_field(name, data_group_to_add)

    def remove_field(self, name: Union[str, int]):
        '''Delete the direct child with the given name.

        See also:
            :meth:`remove_all_occurrences`

        Args:
            name: Name of the child to remove.
        '''
        if not name in self._value_order:
            raise KeyError(f"Cannot delete field '{name}' as it is not present.")
        index = self._value_order.index(name)
        self._value_order = self._value_order[0:index] + self._value_order[index + 1 :]
        self._types_order = self._types_order[0:index] + self._types_order[index + 1 :]
        if name in self._mappings:
            self._mappings.pop(name)
        self._values.pop(name)
        self._types.pop(name)

    def remove_all_occurrences(self, name_to_remove: Union[str, int]):
        '''Remove all fields with a given name.

        All fields with a given name are removed in the tree of which ``self`` is the root, i.e. of this node
        and its descendants.

        See also:
            :meth:`remove_field`

        Args:
            name_to_remove: Name of the field(s) to remove
        '''
        # If a child with the matching name exists, remove it
        if self.has_child(name_to_remove):
            self.remove_field(name_to_remove)

        # Also make sure to remove in children (recursively)
        for type, name in zip(self._types_order, self._value_order):
            if type == SampleDataGroup:
                self[name].remove_all_occurrences(name_to_remove)

    def find_all_occurrences(self, name_to_find: Union[str, int]) -> Tuple[Tuple[Union[str, int]]]:
        '''Find all occurrences of fields with a given name.

        The search is performed in the tree where ``self`` is the root, i.e. of this node and its descendants.

        See also:
            :meth:`get_num_occurrences`

        Args:
            name_to_find: Name of the field(s) to find

        Returns:
            Paths to the found fields. If none were found, an empty tuple is returned. The individual paths
            are themselves tuples. For example, the path ``("name_1", "name_2", "name_3")`` would denote the
            element ``self["name_1"]["name_2"]["name_3"]``.
        '''
        res = []
        self._find_all_occurrences_rec(name_to_find, [], res)
        # convert the individual paths to tuples
        if len(res) > 0:
            res = [tuple(r) for r in res]
        return res

    def get_num_occurrences(self, name_to_find: Union[str, int]) -> int:
        '''Get the number of occurrences of fields with a given name.

        Returns the number of occurrences in the tree where ``self`` is the root, i.e. of this node and its
        descendants.

        See also:
            :meth:`find_all_occurrences`

        Args:
            name_to_find: Name to search for.

        Returns:
            Number of occurrences
        '''
        occurences = self.find_all_occurrences(name_to_find)
        num_occ = len(occurences)
        return num_occ

    def change_type_of_data_and_remove_data(
        self,
        path: Union[Tuple[Union[str, int]], str, int],
        new_type: Union[types.DALIDataType, SampleDataGroup],
        new_mapping: Optional[Union[Dict[Union[str, None], Union[int, float, np.number, bool]]]] = None,
    ):
        """Change the type of a child field and remove its data.

        The data is removed as it is incompatible with the new type. Note that removing the data means
        resetting the reference, not actively deleting the data.

        Example:
            A typical use case would be:

            1) Get the data of which the type should be changed, e.g.: ``data = obj["name"]``
            2) Change the data type

                a) Change the data type as stored in the structure, e.g.:
                   ``obj.change_type_of_data_and_remove_data("name", dali.types.DALIDataType.FLOAT)``
                b) Convert the actual data, e.g.: ``data = dali.fn.cast(data, dtype=types.DALIDataType.FLOAT)``
            3) Write data back, e.g.: ``obj["name"] = data``

        Note that instead of ``"name"``, a nested path can be used.

        Args:
            path: Either a child name or a nested path (sequence of names).
            new_type: For data fields, a :class:`types.DALIDataType`. For data group fields, a
                :class:`SampleDataGroup` used as a blueprint describing the new format.
            new_mapping: New mapping for data fields (see :meth:`add_data_field`). Must be ``None`` for data
                group fields.
        """

        old_element = self.get_item_in_path(path)

        assert isinstance(new_type, SampleDataGroup) == isinstance(old_element, SampleDataGroup), (
            "Data group field array type can only be changed to another data group field type and data field "
            "type only to another data field type."
        )

        if isinstance(new_type, SampleDataGroup):
            assert new_mapping is None, (
                "When changing type of data group field (i.e. SampleDataGroup node), `new_mapping` has to be "
                "`None`"
            )
            parent = self.get_parent_of_path(path)
            if not self.path_is_single_name(path):
                name = path[-1]
            else:
                name = path
            parent._change_data_group_field_type_to(name, new_type)
        else:
            parent = self.get_parent_of_path(path)
            if not self.path_is_single_name(path):
                name = path[-1]
            else:
                name = path
            element_idx = parent._value_order.index(name)

            value_order_to_set = list(parent._value_order)
            value_order_to_set[element_idx] = name
            parent._value_order = tuple(value_order_to_set)
            parent._values[name] = None

            type_order_to_set = list(parent._types_order)
            type_order_to_set[element_idx] = new_type
            parent._types_order = tuple(type_order_to_set)
            parent._types[name] = new_type

            if new_mapping is not None:
                parent._mappings[name] = new_mapping
            elif name in parent._mappings:
                del parent._mappings[name]

    def get_flat_index_first_discrepancy_to_other(self, other: SampleDataGroup) -> int:
        """Get the first flat index where two instances differ in field structure, name, or type.

        Compares flattened field names and types (see :attr:`field_names_flat`,
        :attr:`field_types_flat`). The flattened names include full paths, making structural differences
        visible. Empty sample data group nodes (no data field descendants) are ignored.

        Args:
            other: Other SampleDataGroup instance to compare to.

        Returns:
            Index where the first difference is present, or -1 if there are no differences. Note that
            string fields are compared as :class:`nvidia.dali.types.DALIDataType.UINT8` in the flattened
            types, matching :attr:`field_types_flat`.
        """

        self_types = self.field_types_flat
        self_names = self.field_names_flat

        other_types = other.field_types_flat
        other_names = other.field_names_flat

        types_match = self_types == other_types
        names_match = self_names == other_names
        if types_match and names_match:
            return -1

        length = np.min([len(self_types), len(other_types)])

        for i in range(length):
            if self_names[i] != other_names[i] or self_types[i] != other_types[i]:
                return i

        # If none of the other return statements were executed, this means that the length is different.
        # In this case, 'length' is the first index in which the SampleDataGroup instances differ, as this is
        # the length of the shorter one and therefore points to the first element in the long one for which
        # there is no correspondence in the short one.
        return length

    def ensure_uniform_size_in_batch(self, fill_value: Union[int, float]):
        '''For each data field, ensure uniform size in batch by padding with ``fill_value``.

        This is equivalent to calling ``dali.fn.pad(field_values)`` for all contained data fields (in this
        data group field, and its descendants).

        Warning:
          - This method needs to be called inside the DALI pipeline (except the input callable/iterable).
          - Scalar (i.e. 0D) tensors are not supported. If such tensors are present, an error will be raised.

        Args:
            fill_value: Fill value to be used for the padded region.

        '''
        for type, name in zip(self._types_order, self._value_order):
            if type == SampleDataGroup:
                # Recursively apply to SampleDataNode children
                self[name].ensure_uniform_size_in_batch(fill_value)
            else:
                self._values[name] = fn.pad(self._values[name], fill_value=fill_value)

    def ensure_uniform_size_in_batch_for_all_strings(self):
        '''Ensure uniform size in batch for all string data fields.

        This is useful before outputting from the DALI pipeline in a format that expects uniform size.
        A padding with 0-values is performed for all string data fields. This is done for all contained
        string data fields (in this data group field, and its descendants).

        Note:
            When obtaining the data as strings, the padding is removed and only the actual data is returned.
        '''
        for type, name in zip(self._types_order, self._value_order):
            if type == types.DALIDataType.STRING:
                self._values[name] = fn.pad(self._values[name], fill_value=0)
            elif type == SampleDataGroup:
                self[name].ensure_uniform_size_in_batch_for_all_strings()

    def is_data_field(self, name: Union[str, int]) -> bool:
        '''Check whether a child field is a data field.

        Args:
            name: Name of the child field to check.

        Returns:
            Whether the child field is a data field (contains values) as opposed to a data group field
            (field of type :class:`SampleDataGroup`).
        '''

        if not name in self._value_order:
            raise KeyError(f"No element with name '{name}' is present.")
        is_leaf = not (self._types[name] == SampleDataGroup)
        return is_leaf

    def is_data_group_field(self, name: Union[str, int]) -> bool:
        '''Check whether a child field is a data group field.

        Args:
            name: Name of the child field to check.

        Returns:
            Whether the child field is a data group field (field of type :class:`SampleDataGroup`).
        '''
        return not self.is_data_field(name)

    def to_dictionary(self) -> dict:
        '''Get a nested dictionary with the same (nested) data structure and contained values.

        This and descendants :class:`SampleDataGroup` objects are converted to :class:`dict` objects.
        Contained strings are returned as Python strings.

        Returns:
            Resulting dictionary.
        '''
        res = {}
        for name, type in zip(self._value_order, self._types_order):
            if type == SampleDataGroup:
                res[name] = self[name].to_dictionary()
            else:
                res[name] = self[name]
        return res

    @staticmethod
    def get_numpy_type_for_dali_type(dali_type: types.DALIDataType) -> type:
        '''Get the numpy dtype corresponding to a DALI data type.

        Note:
            Only numeric and boolean DALI types are supported. A ``ValueError`` is raised
            for unsupported types.
        '''
        if not dali_type in SampleDataGroup._type_mapping:
            raise ValueError(
                f"The DALI type ({dali_type}) does not have a corresponding numpy type set in SampleDataGroup"
            )
        res = SampleDataGroup._type_mapping[dali_type]
        return res

    def check_has_children(
        self,
        data_field_children: Optional[Union[Sequence[Union[str, int]], str, int]] = None,
        data_group_field_children: Optional[Union[Sequence[Union[str, int]], str, int]] = None,
        data_field_array_children: Optional[Union[Sequence[Union[str, int]], str, int]] = None,
        data_group_field_array_children: Optional[Union[Sequence[Union[str, int]], str, int]] = None,
        current_name: Optional[str] = None,
    ):
        '''Check that required children are present; raise ``ValueError`` if not.

        Convenience helper for validating presence and kinds of children.

        Args:
            data_field_children: Required child names which must be data fields.
            data_group_field_children: Required child names which must be data group fields.
            data_field_array_children: Required child names which must be arrays of data fields.
            data_group_field_array_children: Required child names which must be arrays of data group fields.
            current_name: Name of the current element. Optional, only used to provide clearer error messages.

        Raises:
            ValueError: If a required child is not present or is not of the expected type.

        '''
        if current_name is None:
            name_to_insert = ""
        else:
            name_to_insert = f"'{current_name}'"

        if data_field_children is not None:
            if isinstance(data_field_children, (str, int)):
                data_field_children = [data_field_children]
            for dfc in data_field_children:
                if not self.has_child(dfc):
                    raise ValueError(f"Data Group field {name_to_insert} does not have child `{dfc}`.")
                if not self.is_data_field(dfc):
                    raise ValueError(f"Data Group field {name_to_insert}: child `{dfc}` is not a data field.")

        if data_group_field_children is not None:
            if isinstance(data_group_field_children, (str, int)):
                data_group_field_children = [data_group_field_children]
            for dgfc in data_group_field_children:
                if not self.has_child(dgfc):
                    raise ValueError(f"Data Group field {name_to_insert} does not have child `{dgfc}`.")
                if not self.is_data_group_field(dgfc):
                    raise ValueError(
                        f"Data Group field {name_to_insert}: child `{dgfc}` is not a data group field."
                    )

        if data_field_array_children is not None:
            if isinstance(data_field_array_children, (str, int)):
                data_field_array_children = [data_field_array_children]
            for dfca in data_field_array_children:
                if not self.has_child(dfca):
                    raise ValueError(f"Data Group field {name_to_insert} does not have child `{dfca}`.")
                if not (self.is_data_group_field(dfca) and self[dfca].is_data_field_array()):
                    raise ValueError(
                        f"Data Group field {name_to_insert}: child `{dfca}` is not a data field array."
                    )

        if data_group_field_array_children is not None:
            if isinstance(data_group_field_array_children, (str, int)):
                data_group_field_array_children = [data_group_field_array_children]
            for dgfca in data_group_field_array_children:
                if not self.has_child(dgfca):
                    raise ValueError(f"Data Group field {name_to_insert} does not have child `{dgfca}`.")
                if not (self.is_data_group_field(dgfca) and self[dgfca].is_data_group_field_array()):
                    raise ValueError(
                        f"Data Group field {name_to_insert}: child `{dgfca}` is not a data group field array."
                    )

    # ----- Private helper functions from here on -----

    def _to_string_with_indent(self, indent: int, with_details: bool) -> str:
        ident_string = " " * indent
        res_str = ""
        space_details = " " * 2
        for type, name in zip(self._types_order, self._value_order):
            if type == SampleDataGroup:
                res_str += (
                    f"{ident_string}{name}:\n{ident_string}"
                    + "{"
                    + f"\n{self[name]._to_string_with_indent(indent + 2, with_details)}"
                )
                res_str += ident_string + "}\n"
            else:
                res_str += f"{ident_string}{name}: {str(self._types[name])}{space_details} " + (
                    f"(is set: {self._values[name] is not None}; "
                    f"mapping available: {name in self._mappings})\n"
                    if with_details
                    else "\n"
                )
        return res_str

    def _get_contained_field_names_flat(self, prefix: str) -> List[str]:
        res = []
        for type, name in zip(self.field_top_level_types, self.contained_top_level_field_names):
            if type == SampleDataGroup:
                if isinstance(name, str):
                    prefix_to_add = f"{name}."
                else:
                    prefix_to_add = f"[{name}]."
                res_i = self[name]._get_contained_field_names_flat(prefix + prefix_to_add)
                res = res + res_i
            else:
                if isinstance(name, str):
                    name_to_use = prefix + name
                else:
                    name_to_use = prefix + f"[{name}]"
                res.append(name_to_use)
        return res

    def _get_contained_types_flat(self) -> List[types.DALIDataType]:
        res = []
        for type, name in zip(self.field_top_level_types, self.contained_top_level_field_names):
            if type == SampleDataGroup:
                res_i = self[name]._get_contained_types_flat()
                res = res + res_i
            elif type == types.DALIDataType.STRING:
                res.append(types.DALIDataType.UINT8)
            else:
                res.append(type)
        return res

    def _set_data_and_get_num_used_data_elements(self, data: Union[tuple, list]) -> int:
        curr_element = 0
        for type, name in zip(self.field_top_level_types, self.contained_top_level_field_names):
            if type == SampleDataGroup:
                num_elements_used = self[name]._set_data_and_get_num_used_data_elements(data[curr_element:])
                curr_element += num_elements_used
            else:
                self._setitem_without_conversions(name, data[curr_element])
                curr_element += 1
        return curr_element

    def _find_all_occurrences_rec(
        self,
        name_to_find: Union[str, int],
        prefix: List[Union[str, int]],
        results_ref: List[List[Union[str, int]]],
    ):
        if name_to_find in self._value_order:
            # Copy necessary as otherwise, the prefix used by the outer recursion levels would be altered
            path = copy.deepcopy(prefix)
            path.append(name_to_find)
            results_ref.append(path)
        for type, name in zip(self._types_order, self._value_order):
            if type == SampleDataGroup:
                # Copy prefix to not modify the original, which will is still needed by the caller
                prefix_for_next = copy.deepcopy(prefix)
                # Include the current child which which we re going to call `_find_all_occurrences_rec(...)`
                # to the prefix
                prefix_for_next.append(name)
                # Call `_find_all_occurrences_rec(...)` of the current child
                self[name]._find_all_occurrences_rec(name_to_find, prefix_for_next, results_ref)

    def _get_copy_except_values(self) -> SampleDataGroup:
        # First, make a shallow copy to obtain the object itself.
        res = copy.copy(self)
        # Then, deep copy the individual properties where possible & needed
        res._mappings = copy.deepcopy(self._mappings)
        res._value_order = copy.deepcopy(self._value_order)
        res._types_order = copy.deepcopy(self._types_order)
        res._types = copy.deepcopy(self._types)
        # 'res._values' should not be filled in this function. Set it to 'None'
        # as otherwise, it is a shallow copy of 'self._values'
        res._values = None
        return res

    def _apply_mapping_check_and_convert(self, name: Union[str, int], value: Any) -> Any:
        if self._do_apply_mapping:
            value = self._apply_mapping_if_set(name, value)
        res = self._check_or_convert_types(name, value)
        return res

    def _apply_mapping_if_set(self, name: Union[str, int], data: Any) -> Any:
        if name in self._mappings:
            if isinstance(data, SampleDataGroup):
                warnings.warn(
                    "Mapping cannot be applied inside the DALI pipeline; call "
                    "set_apply_mapping(False) first to disable. Proceeding without mapping."
                )
                res = data
            else:
                res = get_mapped(data, self._mappings[name])
        else:
            res = data
        return res

    def _check_or_convert_types(self, name: Union[str, int], data: Any) -> Any:
        # Get the expected type of the data field
        dali_type = self._types[name]
        is_data_node = self._is_dali_data_node(data)

        if dali_type == types.DALIDataType.STRING and is_data_node:
            # String fields are represented as uint8 tensors inside the DALI graph. String handling
            # outside the DALI pipeline remains restricted to regular string values.
            if self._do_check_type:
                return check_type(data, np.uint8, name)
            return data

        # Only perform runtime type checking inside the DALI pipeline when explicitly enabled.
        # Skipping this preserves tensor layout metadata (important for steps like AxesLayoutSetter).
        if self._do_check_type:
            # If we are inside the DALI pipeline, we need to check that the data type is correct (regardless of
            # the `_do_convert` flag).
            if is_data_node:
                # Ensure the check op is part of the graph by using its output
                res = check_type(data, self._type_mapping[dali_type], name)
                return res

        if not self._do_convert:
            return data

        # If the set element is a data group, there is no conversion needed
        if dali_type == SampleDataGroup:
            return data

        np_type = self._type_mapping[dali_type]

        # note that `numbers.Number` includes Booleans, but not `np.bool_`, so do not check for python
        # booleans explicitly, but check for `np.bool_`
        if (
            isinstance(data, list)
            or isinstance(data, tuple)
            or isinstance(data, np.ndarray)
            or isinstance(data, np.matrix)
            or isinstance(data, numbers.Number)
            or isinstance(data, np.bool_)
        ):
            data = np.array(data, dtype=np_type)
        elif isinstance(data, cupy.ndarray):
            data = cupy.array(data, dtype=np_type)

        return data

    @staticmethod
    def _is_dali_data_node(data: Any) -> bool:
        return isinstance(data, dali.data_node.DataNode) or isinstance(
            data, getattr(getattr(dali, "_debug_mode", object()), "DataNodeDebug", ())
        )

    def _convert_from_string(
        self, data: Union[dali.pipeline.DataNode, str, Sequence[str], None]
    ) -> Union[dali.pipeline.DataNode, np.ndarray, None]:
        if isinstance(data, dali.pipeline.DataNode):
            res = data
        elif isinstance(data, str):
            as_bytes = data.encode("utf-8")
            res = np.frombuffer(as_bytes, dtype=np.uint8)
        elif data is None:
            res = None
        else:
            raise ValueError(f"Expected a string or a DataNode, but got {type(data)}")

        return res

    def _convert_to_string(
        self, data: Union[dali.pipeline.DataNode, np.ndarray, cupy.ndarray, torch.Tensor, None]
    ) -> Union[dali.pipeline.DataNode, str, List[str], None]:
        if isinstance(data, dali.pipeline.DataNode):
            return data
        if isinstance(data, np.ndarray):
            np_data = data
        elif isinstance(data, cupy.ndarray):
            np_data = np.array(data.get())
        elif isinstance(data, torch.Tensor):
            np_data = data.detach().cpu().numpy()
        elif data is None:
            return None
        else:
            raise ValueError(
                f"Expected a numpy array, cupy array, a torch tensor, or a DataNode, but got {type(data)}"
            )

        # If this is an encoded string, the first element will contain a number
        # If it is a (possibly nested) sequence of strings, the first element will be again a sequence (and
        # the else branch is executed)
        if isinstance(np_data[0], np.number):
            as_bytes = np_data.tobytes().strip(b'\x00')
            res = str(as_bytes.decode("utf-8"))
        else:
            # Elements are themseves (spossibly nested) sequences of strings (see comment above `if``). In
            # this case, process each entry (recursively, until the actual strings are reached).
            res = [self._convert_to_string(d) for d in np_data]

        return res

    def _change_data_group_field_type_to(self, name: Union[str, int], value: SampleDataGroup):
        if self._types[name] != SampleDataGroup:
            raise ValueError("Called _change_data_group_field_type_to() for a non-SampleDataGroup element.")
        blueprint = value.get_empty_like_self()
        blueprint.set_apply_mapping(self._apply_mapping_if_set)
        blueprint.set_do_convert(self._do_convert)
        self._values[name] = blueprint

    def _setitem_without_conversions(self, name: Union[str, int], value: Any):
        if not name in self._values:
            raise KeyError(f"No field with name '{name}'")
        if self._types[name] == SampleDataGroup:
            if (self._types[name] == SampleDataGroup) and (not self[name].type_matches(value)):
                raise KeyError(
                    f"Tried to set a data group field '{name}' (fields of type SampleDataGroup), but types "
                    "do not match."
                )
        self._values[name] = value

    def _getitem_without_conversions(self, name: Union[str, int]) -> Any:
        if not name in self._values:
            raise KeyError(f"No field with name '{name}'")
        return self._values[name]
