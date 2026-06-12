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

from typing import List, Optional, Sequence, Set, Tuple, Union

import nvidia.dali.fn as fn

from .sample_data_group import SampleDataGroup

PathElement = Union[str, int]
PathTuple = Tuple[PathElement, ...]
PathType = Union[PathElement, Sequence[PathElement]]


class _InsertCopyForPassthrough:
    '''Insert ``fn.copy`` on selected pipeline outputs.

    Workaround for cases where parallel external-source outputs would otherwise be returned
    directly from the pipeline (see DALI dynamic-executor parallel ES notes), which can lead
    to data corruption in certain cases.

    This helper is used internally by the pipeline construction code. Fields to copy can be selected by
    name, by branch path, or by name within selected branches. If no selectors are configured,
    every output data field is copied.
    '''

    def __init__(
        self,
        data_empty: SampleDataGroup,
        field_names: Optional[Sequence[Union[str, int]]] = None,
        field_names_scope_paths: Optional[Sequence[PathType]] = None,
        branch_paths: Optional[Sequence[PathType]] = None,
    ):
        '''

        Args:
            data_empty: Final output data format blueprint.
            field_names: Names of data fields to copy. By default, every occurrence in the final output
                structure is copied. Use ``field_names_scope_paths`` to limit the search to specific
                sub-trees.
            field_names_scope_paths: Optional paths to sub-trees (data group fields) under which
                ``field_names`` are resolved. Each entry is a path to a data group; name lookup is
                performed only inside that group and its descendants. Ignored when ``field_names``
                is ``None``.
            branch_paths: Paths selecting branches to copy. If a path refers to a data field, that
                field is copied. If it refers to a data group field, every data field in that
                sub-tree (recursively) is copied.

        Raises:
            ValueError: If ``field_names_scope_paths`` is set without ``field_names``, or if a configured
                path does not exist or has the wrong node kind (e.g. a scope path that is not a data group
                field).
        '''

        self._field_names = tuple(field_names) if field_names is not None else None
        self._field_names_scope_paths = (
            tuple(self._normalize_path(path) for path in field_names_scope_paths)
            if field_names_scope_paths is not None
            else None
        )
        self._branch_paths = (
            tuple(self._normalize_path(path) for path in branch_paths) if branch_paths is not None else None
        )

        if (
            self._field_names_scope_paths is not None
            and len(self._field_names_scope_paths) > 0
            and (self._field_names is None or len(self._field_names) == 0)
        ):
            raise ValueError(
                "`field_names_scope_paths` can only be used together with non-empty `field_names`."
            )

        self._paths_to_copy = self._sort_paths(self._resolve_paths_to_copy(data_empty))

    def __call__(self, data: SampleDataGroup) -> SampleDataGroup:
        '''Apply ``fn.copy`` to the configured output fields.

        Args:
            data: Final pipeline output structure before flattening.

        Returns:
            The input structure with selected output fields replaced by copied data nodes.
        '''

        for path in self._paths_to_copy:
            copy = fn.copy(data.get_item_in_path(path))
            data.set_item_in_path(path, copy)
        return data

    def _resolve_paths_to_copy(self, data: SampleDataGroup) -> Set[PathTuple]:
        '''Resolve configured selectors to concrete output data field paths.'''

        paths: Set[PathTuple] = set()

        # Field-name selectors resolve every matching output leaf, optionally constrained by scope paths.
        if self._field_names is not None:
            paths.update(self._resolve_field_name_paths(data))

        # Branch selectors resolve explicit paths; group paths are expanded to their contained leaves.
        if self._branch_paths is not None:
            paths.update(self._resolve_branch_paths(data))

        # No configured selector means copy every output leaf; explicit empty selectors copy none.
        if not self._has_selection():
            paths.update(self._collect_data_field_paths_under_group(data, ()))

        return paths

    def _has_selection(self) -> bool:
        '''Check whether any copy selector was explicitly configured.'''

        # Use presence rather than truthiness: an explicit empty selector means "copy nothing",
        # while omitting all selectors means "copy every output field".
        has_name_selection = self._field_names is not None
        has_branch_selection = self._branch_paths is not None
        return has_name_selection or has_branch_selection

    def _resolve_field_name_paths(self, data: SampleDataGroup) -> Set[PathTuple]:
        '''Resolve field-name selectors to matching output data field paths.'''

        assert self._field_names is not None
        paths: Set[PathTuple] = set()

        # If no scope paths are configured, resolve field names directly against the entire data tree.
        if self._field_names_scope_paths is None:
            for name in self._field_names:
                paths.update(tuple(path) for path in data.find_all_occurrences(name))
            return paths

        # Otherwise, resolve field names within each configured scope group.
        for scope_path in self._field_names_scope_paths:
            self._ensure_path_exists(data, scope_path)
            if not data.path_exists_and_is_data_group_field(scope_path):
                raise ValueError(f"Field name scope path `{scope_path}` must refer to a data group field.")
            subtree = data.get_item_in_path(scope_path)
            for name in self._field_names:
                paths.update(
                    scope_path + tuple(relative_path) for relative_path in subtree.find_all_occurrences(name)
                )
        return paths

    def _resolve_branch_paths(self, data: SampleDataGroup) -> Set[PathTuple]:
        '''Resolve branch selectors to selected output data field paths.'''

        assert self._branch_paths is not None
        paths: Set[PathTuple] = set()

        for branch_path in self._branch_paths:
            # Reject missing selector paths during setup rather than during graph execution.
            self._ensure_path_exists(data, branch_path)
            if data.path_exists_and_is_data_group_field(branch_path):
                # A branch path to a group selects all data leaves below that group, but a branch path to
                # a leaf selects only that exact field.
                subtree = data.get_item_in_path(branch_path)
                paths.update(
                    branch_path + relative_path
                    for relative_path in self._collect_data_field_paths_under_group(subtree, ())
                )
            else:
                # A branch path that already points to a data field is copied as-is.
                paths.add(branch_path)
        return paths

    @classmethod
    def _collect_data_field_paths_under_group(
        cls, group: SampleDataGroup, prefix: PathTuple
    ) -> Tuple[PathTuple, ...]:
        '''Collect tuple paths for all data fields under a group.'''

        # Branch selections can point to a data group; expand those groups to concrete leaf data fields so
        # ``__call__`` can replace each selected value in-place.
        paths: List[PathTuple] = []
        for name in group.contained_top_level_field_names:
            current = prefix + (name,)
            if group.is_data_group_field(name):
                paths.extend(cls._collect_data_field_paths_under_group(group[name], current))
            else:
                paths.append(current)
        return tuple(paths)

    @staticmethod
    def _normalize_path(path: PathType) -> PathTuple:
        '''Convert a single name or path sequence into a tuple path.'''

        if SampleDataGroup.path_is_single_name(path):
            return (path,)
        return tuple(path)

    @staticmethod
    def _ensure_path_exists(data: SampleDataGroup, path: PathTuple) -> None:
        '''Raise an error if a path does not exist in the data format.'''

        if not data.path_exists(path):
            raise ValueError(f"Path `{path}` does not exist in the output data format.")

    @staticmethod
    def _sort_paths(paths: Set[PathTuple]) -> Tuple[PathTuple, ...]:
        '''Sort paths deterministically across mixed string and integer names.'''

        return tuple(sorted(paths, key=lambda path: tuple(str(part) for part in path)))
