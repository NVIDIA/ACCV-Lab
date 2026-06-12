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

from typing import Sequence, TYPE_CHECKING, Optional, Union

import warnings

from nvidia.dali.pipeline.experimental import pipeline_def

# from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali as dali

from .sample_data_group import SampleDataGroup
from ._insert_copy_for_passthrough import (
    PathType,
    _InsertCopyForPassthrough,
)
from ..processing_steps import PipelineStepBase

if TYPE_CHECKING:
    from ..inputs import CallableBase, IterableBase


class PipelineDefinition:
    '''Definition for the data loading and pre-processing pipeline.

    Configure with a data-loading functor and an ordered list of processing steps. Exposes utilities to
    retrieve the input data format (blueprint), infer the output data format by applying each step's
    format-checking logic, and build a DALI pipeline that combined the data loading functor and the
    processing steps.
    '''

    def __init__(
        self,
        data_loading_callable_iterable: Union['CallableBase', 'IterableBase'],
        preprocess_functors: Optional[Sequence[Optional[PipelineStepBase]]] = None,
        check_data_format: bool = True,
        use_parallel_external_source: bool = True,
        prefetch_queue_depth: int = 2,
        print_sample_data_group_format: bool = False,
        copy_external_source_passthrough_outputs: Optional[bool] = None,
        passthrough_copy_field_names: Optional[Sequence[Union[str, int]]] = None,
        passthrough_copy_field_names_scope_paths: Optional[Sequence[PathType]] = None,
        passthrough_copy_branch_paths: Optional[Sequence[PathType]] = None,
    ):
        '''

        Args:
            data_loading_callable_iterable: Callable or iterable performing the loading of the data.
            preprocess_functors: Functors for the individual processing steps which will be executed in
                sequence on the input data. May contain ``None``-elements, which are ignored.
                Optional, if not set, the loaded data is returned as is.
            use_parallel_external_source: Whether to use the parallel external source.
            prefetch_queue_depth: The depth of the prefetch queue. Only used if `use_parallel_external_source`
                is True.
            print_sample_data_group_format: Whether to print the sample data group formats after each
                processing step during the setup of the pipeline (e.g. for debugging purposes).
            copy_external_source_passthrough_outputs: Optional control for copying final pipeline outputs before
                returning them. When omitted and no pass-through copy selectors are configured, all final output
                data fields are copied and a warning about the possible unintended overhead is issued. Set to ``True`` to
                explicitly copy outputs according to the configured selectors, or all final output data fields
                if no selectors are configured. Set to ``False`` to disable copying. This can be used as a
                workaround for DALI pass-through outputs from parallel external sources. See the
                :ref:`Important note about DALI pass-through outputs <dali-pipeline-framework-external-source-pass-through-note>`
                for details.
            passthrough_copy_field_names: Optional data field names to copy in the final output structure. If
                omitted together with the other copy selectors while ``copy_external_source_passthrough_outputs``
                is enabled, all final output data fields are copied.
            passthrough_copy_field_names_scope_paths: Optional final output group paths under which
                ``passthrough_copy_field_names`` are resolved. For each configured scope path, all configured
                field names are searched recursively inside that scope. Paths use the same format as in
                processing steps: a single field name or a sequence of field names for nested fields, e.g.
                ``("camera", "annotations")``.
            passthrough_copy_branch_paths: Optional final output paths to copy. Paths use the same format as
                in processing steps: a single field name or a sequence of field names for nested fields, e.g.
                ``("camera", "image")``. A path to a data group copies all data fields under that group.
        '''

        self._data_loading_callable_iterable = data_loading_callable_iterable
        # Remove pre-processing steps which are None
        self._preprocess_functors = [pf for pf in preprocess_functors or [] if pf is not None]
        self._check_data_format = check_data_format
        self._use_parallel_external_source = use_parallel_external_source
        self._prefetch_queue_depth = prefetch_queue_depth
        self._print_sample_data_group_format = print_sample_data_group_format

        has_passthrough_copy_selection = (
            passthrough_copy_field_names is not None
            or passthrough_copy_field_names_scope_paths is not None
            or passthrough_copy_branch_paths is not None
        )
        if copy_external_source_passthrough_outputs is None:
            if has_passthrough_copy_selection:
                raise ValueError(
                    "Pass-through output copy selectors require explicitly setting "
                    "`copy_external_source_passthrough_outputs=True`."
                )
            warnings.warn(
                "`copy_external_source_passthrough_outputs` was not set. Copying all final pipeline "
                "outputs by default to avoid potential pipeline pass-through output corruption. This may add "
                "overhead. To reduce overhead, see the `PipelineDefinition` API docs for further details and "
                "configure `passthrough_copy_field_names`, `passthrough_copy_field_names_scope_paths`, or "
                "`passthrough_copy_branch_paths`; or explicitly set `copy_external_source_passthrough_outputs=False` "
                "to disable copying."
            )
            copy_external_source_passthrough_outputs = True
        if has_passthrough_copy_selection and not copy_external_source_passthrough_outputs:
            raise ValueError(
                "Pass-through output copy selectors require "
                "`copy_external_source_passthrough_outputs=True`."
            )

        self._copy_external_source_passthrough_outputs = copy_external_source_passthrough_outputs
        self._passthrough_copy_field_names = passthrough_copy_field_names
        self._passthrough_copy_field_names_scope_paths = passthrough_copy_field_names_scope_paths
        self._passthrough_copy_branch_paths = passthrough_copy_branch_paths
        self._passthrough_output_copy: Optional[_InsertCopyForPassthrough] = None

        if self._check_data_format:
            warnings.warn(
                "Data format checking is enabled. This may add some overhead. "
                "It is recommended to disable it in production. "
                "You can disable it by setting `check_data_format=False` in the constructor of the "
                "`PipelineDefinition` class."
            )

    @property
    def input_data_structure(self) -> SampleDataGroup:
        '''Get the input data format (blueprint).

        The input blueprint is provided by the data-loading functor passed at construction time.

        Returns:
            :class:`SampleDataGroup` blueprint object describing the input data format (no actual data).
        '''
        return self._data_loading_callable_iterable.used_sample_data_structure

    def check_and_get_output_data_structure(self) -> SampleDataGroup:
        '''Infer and return the output data format (blueprint).

        Starting from the input blueprint provided by the loading functor, each processing step validates
        compatibility and transforms the blueprint (e.g., adding fields or changing types). Steps are applied
        in sequence to obtain the final output blueprint. If an incompatibility is detected, an exception is
        raised.

        Returns:
            :class:`SampleDataGroup` blueprint object describing the output data format (no actual data).

        Raises:
            ValueError: If the data loading functor is not compatible with the first processing step.

        '''

        # Get the input data format blueprint
        intermediate_setup = self.input_data_structure

        # For each of the processing steps, check if it is compatible with the blueprint and adjust the blueprint to how the data format will look like after
        # the processing step is applied.
        if self._preprocess_functors is not None:
            for pf in self._preprocess_functors:
                intermediate_setup = pf.check_input_data_format_and_set_output_data_format(intermediate_setup)

        if self._copy_external_source_passthrough_outputs:
            self._prepare_passthrough_output_copy(intermediate_setup)

        return intermediate_setup

    def get_dali_pipeline(self, *args, **kwargs) -> dali.pipeline.Pipeline:
        '''Get the DALI pipeline as configured.

        Note:
            This calls a function decorated with ``@pipeline_def`` used by DALI to create a pipeline object.
            The resulting pipeline object is returned. For more information on the possible arguments
            (i.e. ``*args`` and ``**kwargs`` in this function), see the documentation of
            the :func:`nvidia.dali.pipeline.experimental.pipeline_def` decorator.

        Args:
            *args: Arguments for the DALI pipeline.
            **kwargs: Keyword arguments for the DALI pipeline.

        Returns:
            The DALI pipeline as configured.
        '''
        if self._print_sample_data_group_format:
            input_data_structure = self.input_data_structure
            print("DALI Pipeline ///////////////////////////////////////////////////////////////")
            print("Input format:")
            print(f"\n{input_data_structure.get_string_no_details()}\n")
            for i, pf in enumerate(self._preprocess_functors):
                print(f"After processing step # {i} ({pf}):")
                input_data_structure = pf.check_input_data_format_and_set_output_data_format(
                    input_data_structure
                )
                print(f"\n{input_data_structure.get_string_no_details()}\n")
            if self._copy_external_source_passthrough_outputs:
                self._prepare_passthrough_output_copy(input_data_structure)
            print("///////////////////////////////////////////////////////////////")
        else:
            # If no pre-processing steps are provided, we still need to check the compatibility of the data
            # loading functors with the output of the previous step (or data provider).
            self.check_and_get_output_data_structure()

        return self._get_dali_pipeline_inner(*args, **kwargs)

    @pipeline_def
    def _get_dali_pipeline_inner(self) -> dali.pipeline.Pipeline:
        '''Get the DALI pipeline as configured.

        This function is not called directly. As it is decorated with ``@pipeline_def``, it can be used by DALI to create a pipeline object,
        which then can be used to obtain batch after batch of data.

        Returns:
            The DALI pipeline as configured.
        '''

        # Get the input data format blueprint
        data_structure_used = self.input_data_structure
        data_structure_used.set_apply_mapping(False)
        data_structure_used.set_do_check_type(self._check_data_format)

        # Get the input data types (as a flat sequence) as well as the number of data fields
        input_types = data_structure_used.field_types_flat
        num_data = len(input_types)

        # Some parameters of the external source depend on whether the data loading functor is a callable or
        # an iterable. Set these parameters up here.
        is_callable = callable(self._data_loading_callable_iterable)
        batch_mode = not is_callable
        cycle_mode = "raise" if not is_callable else None

        # Use the external source. Note how it uses input_types. The output will also be a flat sequence of data.
        # The data loading functor is expected to use the same blueprint as data_structure_used, fill it with data,
        # and output a flattened version (using the `get_data()` method).
        data = fn.external_source(
            source=self._data_loading_callable_iterable,
            num_outputs=num_data,
            dtype=input_types,
            batch=batch_mode,
            parallel=self._use_parallel_external_source,
            prefetch_queue_depth=self._prefetch_queue_depth if self._use_parallel_external_source else None,
            cycle=cycle_mode,
        )

        # We got the flattened output; fill it back to the blueprint to obtain a correctly filled structure.
        data_structure_used.set_data(data)

        # Perform the individual processing steps
        for func in self._preprocess_functors:
            data_structure_used = func(data_structure_used)

        # Pad each string fiels to the same length in the batch
        data_structure_used.ensure_uniform_size_in_batch_for_all_strings()

        data_structure_used = self._copy_passthrough_outputs_if_enabled(data_structure_used)

        # Get the data as a flat sequence. Similar to the external source, we can only output sequences of DataNode elements, no nested data structures.
        data_out = data_structure_used.get_data()
        # And return the flat data.
        return data_out

    def _copy_passthrough_outputs_if_enabled(self, data: SampleDataGroup) -> SampleDataGroup:
        if not self._copy_external_source_passthrough_outputs:
            return data
        if self._passthrough_output_copy is None:
            self._prepare_passthrough_output_copy(data.get_empty_like_self())
        assert self._passthrough_output_copy is not None
        return self._passthrough_output_copy(data)

    def _prepare_passthrough_output_copy(self, data_empty: SampleDataGroup) -> None:
        if not self._copy_external_source_passthrough_outputs:
            return
        try:
            self._passthrough_output_copy = _InsertCopyForPassthrough(
                data_empty,
                field_names=self._passthrough_copy_field_names,
                field_names_scope_paths=self._passthrough_copy_field_names_scope_paths,
                branch_paths=self._passthrough_copy_branch_paths,
            )
        except ValueError as error:
            raise ValueError(
                "Invalid pass-through output copy configuration for final output format: " f"{error}"
            ) from error
