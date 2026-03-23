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

import os
from enum import Enum
from typing import Union, Sequence, Dict, Any, Callable, List, TYPE_CHECKING, TypeAlias, Optional
import json
import numbers
import math
import importlib.util
import warnings
import copy
import pickle

import torch
import numpy as np

# Type aliases for commonly used complex types
TensorData: TypeAlias = Union[torch.Tensor, Any]
TensorDataStructure: TypeAlias = Union[
    TensorData, Sequence[Union[TensorData, Sequence, Dict]], Dict[str, Union[TensorData, Sequence, Dict]]
]

FormattedTensorData: TypeAlias = Union['TensorDumper._TensorWithFormat', Any]
FormattedDataStructure: TypeAlias = Union[
    FormattedTensorData,
    Sequence[Union[FormattedTensorData, Sequence, Dict]],
    Dict[str, Union[FormattedTensorData, Sequence, Dict]],
]

GenericDataStructure: TypeAlias = Union[
    Any, Sequence[Union[Any, Sequence, Dict]], Dict[str, Union[Any, Sequence, Dict]]
]

# Additional type aliases for common patterns
OptionalSequence: TypeAlias = Optional[Sequence[int]]
OptionalTypeDict: TypeAlias = Optional[Dict[str, 'TensorDumper.Type']]
OptionalPermuteDict: TypeAlias = Optional[Dict[str, OptionalSequence]]
OptionalPath: TypeAlias = Optional[str]

if TYPE_CHECKING:
    try:
        from accvlab.batching_helpers import RaggedBatch
    except ImportError:
        # Dummy to avoid errors in case the `RaggedBatch` class is not available.
        # This can happen if the `batching_helpers` package is not installed.
        # In this case, it is still possible to use the `TensorDumper` class, but
        # dumping of `RaggedBatch` data is not supported.
        class RaggedBatch:
            pass


if __name__ != "__main__":
    from .singleton_base import SingletonBase
else:
    from singleton_base import SingletonBase


class TensorDumper(SingletonBase):
    '''Singleton class for dumping tensor & gradient data to a directory and comparing to previously dumped
    data.

    This class provides a way to dump tensor data to a directory in a structured format.

    The dumper is able to dump tensors, gradients, :class:`RaggedBatch` objects, as well as data with
    user-defined & auto-applied converters. Furthermore, it supports custom processing prior to dumping (e.g.
    converting of bounding boxes to images containing the bounding boxes), which is performed only if the
    dumper is enabled, and does not incur overhead if the dumper is not enabled.

    Main JSON files are created for each dump (one for the data and one for the gradients). The individual
    tensors (or converted data) can be stored inside the main JSON file, or in separate binary/image files
    (can be configured, and can vary for individual data entries). In case of the binary/image files, the main
    JSON file contains a reference to the file, and the file is stored in the same directory as the main JSON
    file.

    The dumper can also be used to compare to previously dumped data, to detect mismatches. This can be useful
    for debugging e.g. to rerun the same code multiple times, while always comparing to the same dumped data.
    This can be use used when modifying (e.g. optimizing) the implementation, or to check for determinism.

    Important:
        The dumper is a singleton, so that it can be used in different source files without having to pass the
        instance around.

    Note:
        The comparison is only supported if all data is dumped in the ``Type.JSON`` format. This can be
        enforced by calling :meth:`set_dump_type_for_all` before dumping/comparing the data (so easy
        switching between dumping for manual inspection and comparison is possible).

    Note:
        When in the disabled state, all dumping-related methods (dump, add data, compare to dumped data etc)
        are empty methods, which means they have no effect and minimal overhead.

    Note:
        When obtaining an object using (``TensorDumper()``) the singleton is returned if already
        created.

        If parameters are provided when calling ``TensorDumper()``, this will enable the dumper
        (equivalent to calling :meth:`enable`). Note that enabling can only be done once, and will
        lead to an error if attempted a second time.
    '''

    class Type(Enum):
        '''Dump format types.

        The format type determines how tensor data is serialized when dumped.

        Note:
            For binary types except ``PICKLE`` (i.e. ``BINARY``, ``IMAGE_RGB``, ``IMAGE_BGR``, ``IMAGE_I``),
            entries are added to the main JSON file indicating the filenames of the stored data. Also, files
            containing meta-data are created and stored in the same directory. For ``BINARY``, the meta-data
            is the shape and dtype of the tensor. For ``IMAGE_*``, the meta-data is the original range of the
            image data (min and max value) and the image format (RGB, BGR, Intensity). For ``PICKLE``, no
            meta-data is written as the pickled object is self-contained.

        Note:
            For ``BINARY``, ``IMAGE_*`` and ``PICKLE`` formats, entries are added to the main JSON file
            indicating the filenames of the stored data. The filenames for these cases are:

              - blob/image data: ``[<main_json_file_name>]<path_to_data_in_dumped_structure>.<file_type>``
              - meta-data (if applicable):
                ``[<main_json_file_name>]<path_to_data_in_dumped_structure>.<file_type>.meta.json``

        Note:
            For images containing multiple channels, the color channel is assumed to be the last dimension. If
            this is not the case, permutation of the axes needs to be applied to move the color channel to the
            last dimension. The permutation can be applied using the ``permute_axes`` parameter, e.g. of
            :meth:`add_tensor_data`.

            If a tensor contains more than the necessary number of dimensions (3 for color images,
            2 for grayscale images), the leading dimensions are treated as iterating over the images,
            and multiple images are dumped (with the indices of the leading dimensions indicated in the
            filename).
        '''

        #: Tensor data is serialized into the JSON file as nested lists.
        #: Suitable for small tensors and provides human-readable output.
        JSON = 0
        #: Tensor data saved as binary files with metadata in separate JSON files.
        #: Efficient for large tensors; preserves exact numerical precision.
        BINARY = 1
        #: Tensor data converted to PNG image format (RGB, 3 channels).
        #: Channel must be the last dimension; permute axes if necessary.
        IMAGE_RGB = 2
        #: Tensor data converted to PNG image format (BGR, 3 channels).
        #: Channel must be the last dimension; permute axes if necessary.
        IMAGE_BGR = 3
        #: Tensor data converted to PNG image format (grayscale).
        #: Single channel; no explicit channel dimension.
        IMAGE_I = 4
        #: Tensor data saved as pickle files.
        PICKLE = 5

        @classmethod
        def is_image(cls, dump_type: 'TensorDumper.Type') -> bool:
            return dump_type in [cls.IMAGE_RGB, cls.IMAGE_BGR, cls.IMAGE_I]

    class _ComparisonConfig:
        def __init__(
            self,
            eps_numerical_data: float = 1e-6,
            num_errors_per_tensor_to_show: int = 1,
            allow_missing_data_in_current: bool = False,
            allow_missing_data_in_previous: bool = False,
        ):
            self.eps_numerical_data = eps_numerical_data
            self.num_errors_per_tensor_to_show = num_errors_per_tensor_to_show
            self.allow_missing_data_in_current = allow_missing_data_in_current
            self.allow_missing_data_in_previous = allow_missing_data_in_previous

    class _TensorWithFormat:
        def __init__(self, tensor: Any, dump_type: 'TensorDumper.Type', permute_axes: OptionalSequence):
            self.tensor = tensor
            self.dump_type = dump_type
            self.permute_axes = permute_axes

    class _CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.complexfloating):
                return complex(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                obj_as_list = obj.tolist()
                return obj_as_list
            elif isinstance(obj, torch.dtype):
                obj_as_str = str(obj)
                return obj_as_str
            elif isinstance(obj, type):
                return str(obj)
            else:
                return super(TensorDumper._CustomEncoder, self).default(obj)

    def __init__(self, *args, **kwargs):
        '''
        Args:
            dump_dir: The directory to dump the data to. If provided, the dumper will be enabled
                automatically. If not provided, the dumper will be disabled and can be enabled later by
                calling :meth:`enable`.
        '''
        if not hasattr(self, '_initialized'):
            # Only import accvlab.batching_helpers if it is available. If it is not, show a warning about
            # dumping of RaggedBatch data being not supported.
            accvlab_spec = importlib.util.find_spec("accvlab")
            batching_helpers_spec = None
            if accvlab_spec is not None and accvlab_spec.submodule_search_locations is not None:
                batching_helpers_spec = importlib.util.find_spec("accvlab.batching_helpers")
            if batching_helpers_spec is None:
                warnings.warn(
                    "`accvlab.batching_helpers` is not available. Dumping of `RaggedBatch` data is not "
                    "supported."
                )
            self._initialized = True
            self._enabled = False
            self._SET_DOCSTRINGS_OF_ENABLED_METHOD_VARIANTS()
        if len(args) > 0 or len(kwargs) > 0:
            self.enable(*args, **kwargs)

    def enable(self, dump_dir: str):
        '''Enable the TensorDumper singleton.

        This method can be called only once and enables the TensorDumper singleton.
        Any use of the singleton before enabling it is ignored.

        Args:
            dump_dir: The directory to dump the data to.
        '''
        if self._enabled:
            raise RuntimeError("`TensorDumper` is already enabled. Can only be enabled once.")
        self._dump_dir = dump_dir
        self._range_stack = []
        self._dump_count = 0
        self._tensor_struct = {}
        self._grad_struct = {}
        self._grad_computed = False
        self._enabled = True
        self._after_dump_count_actions = {}
        self._custom_converters = {np.ndarray: lambda x: torch.from_numpy(x)}
        # If not None, this overrides *all* dump type settings (per-call dump_type and dump_type_override).
        self._dump_type_for_all = None

        self._dump_is_compare = False
        self._dump_is_compare_params = None

        # Set the methods
        self.set_dump_is_compare = self._set_dump_is_compare_enabled
        self.push_range = self._push_range_enabled
        self.pop_range = self._pop_range_enabled
        self.add_tensor_data = self._add_tensor_data_enabled
        self.add_grad_data = self._add_grad_data_enabled
        self.set_dump_type_for_all = self._set_dump_type_for_all_enabled
        self.dump = self._dump_enabled
        self.compare_to_dumped_data = self._compare_to_dumped_data_enabled
        self.set_gradients = self._set_gradients_enabled
        self.reset_dump_count = self._reset_dump_count_enabled
        self.set_dump_count = self._set_dump_count_enabled
        self.perform_after_dump_count = self._perform_after_dump_count_enabled
        self.register_custom_converter = self._register_custom_converter_enabled
        self.enable_ragged_batch_dumping = self._enable_ragged_batch_dumping_enabled
        self.run_if_enabled = self._run_if_enabled_enabled

    def push_range(self, range_name: Union[str, Callable[[], str]]):
        '''Push a range to the range stack.

        Multiple ranges can be pushed and popped in a nested manner. The ranges will be prepended to the dump
        path (see e.g. the ``path`` argument in :meth:`add_tensor_data`) in the order in which they were
        pushed.

        The ranges can be used to conveniently disambiguate the names of data entries where the same
        name is used in multiple contexts (e.g. multiple iterations of a loop, function called from
        multiple places, etc.).

        Important:
            To ensure that the formatting is performed only if the tensor dumper is enabled,
            the range should not be formatted when passing the argument. Instead, the formatting happens
            inside the method, and the additional arguments (``args``) are used to format the range name.

        Args:
            range_name: The name of the range to push.
            *args: Additional arguments to format the range name. If not provided, the range name is used
                as is.
        '''
        pass

    def pop_range(self):
        '''Pop the last range from the range stack.

        Args:
            range_name: The name of the range to pop.
        '''
        pass

    def set_dump_is_compare(
        self,
        eps_numerical_data: float = 1e-6,
        num_errors_per_tensor_to_show: int = 1,
        allow_missing_data_in_current: bool = False,
        allow_missing_data_in_previous: bool = False,
        as_warning: bool = False,
    ) -> bool:
        '''Automatically replace calls to :meth:`dump` with calls to :meth:`compare_to_dumped_data`.

        Note:
            The parameters defined in this method will be forwarded to :meth:`compare_to_dumped_data`.
            Note that ``compare_if_empty`` is not passed here. Instead, the value passed to :meth:`dump` is
            used.

        See Also:
            Please see the documentation of :meth:`compare_to_dumped_data` for more details.

        Args:
            eps_numerical_data: The numerical tolerance for the comparison of numerical data.
            num_errors_per_tensor_to_show: The number of most significant errors to show per tensor.
            allow_missing_data_in_current: If ``True``, the comparison will not raise an error if the current data is missing
                some keys which are present in the reference data.
            allow_missing_data_in_previous: If ``True``, the comparison will not raise an error if the reference data is missing
                some keys which are present in the current data.
            as_warning: If ``True``, no error is raised in case of a mismatch and instead, a warning is printed.
                If ``False``, an error is raised.
        '''
        pass

    @property
    def is_enabled(self) -> bool:
        '''Whether the TensorDumper is enabled'''
        return self._enabled

    def add_tensor_data(
        self,
        path: str,
        data: Union[TensorDataStructure, Callable[[], TensorDataStructure]],
        dump_type: 'TensorDumper.Type',
        dump_type_override: OptionalTypeDict = None,
        permute_axes: OptionalSequence = None,
        permute_axes_override: OptionalPermuteDict = None,
        exclude: Union[Sequence[str], None] = None,
    ):
        '''
        Add tensor data to the dump.

        The data is formatted and inserted into the dump structure.

        Args:
            path: Path where the data will be inserted. If the path does not exist, it will be created.
                If `data` is a dictionary, the path may be already present in the structure,
                but the direct children of `data` need to be non-existent in the element the path points to.
                If `data` is not a dictionary, the path must not be present in the structure and the
                data will be inserted at the path.
            data: The tensor data to add
            dump_type: The type of dump to use
            dump_type_override: A dictionary mapping names to dump types.
                If a name is present in the dictionary, the dump type for all tensors with that name in the path
                (i.e. either the name itself or the name of a parent along the path) will be overridden with the
                value in the dictionary. If multiple names match the path, the match closest to the tensor
                (i.e. further inside the structure) is used.
                If ``None``, no override is applied.
            permute_axes: Permutation of axes to apply to the tensor data.
                If ``None``, no permutation is applied.
            permute_axes_override: A dictionary mapping names to permute axes.
                If a name is present in the dictionary, the permute axes for all tensors with that name in the path
                (i.e. either the name itself or the name of a parent along the path) will be overridden with the
                value in the dictionary. If multiple names match the path, the match closest to the tensor
                (i.e. further inside the structure) is used.
                If ``None``, no override is applied.
            exclude: List of entries to exclude from the dump. There entries are specified by names and may apply to any level of the data structure.
        '''
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def add_grad_data(
        self,
        path: str,
        data: Union[TensorDataStructure, Callable[[], TensorDataStructure]],
        dump_type: 'TensorDumper.Type',
        dump_type_override: OptionalTypeDict = None,
        permute_grad_axes: OptionalSequence = None,
        permute_grad_axes_override: OptionalPermuteDict = None,
        exclude: Union[Sequence[str], None] = None,
    ):
        '''Add gradient data of the given tensor(s) to dump.

        Note that if this method is called, :meth:`set_gradients` must be called before dumping the next time.

        The gradients are computed using :func:`torch.autograd.grad`, and do not influence the gradients
        as computed/used elsewhere in the code (e.g. in the training loop).

        Note that tensors which do not require gradients or which are not part of the computation graph
        can be included in the dump, but no actual gradients will be computed for them. Instead,
        a note will be written to the json dump in case that ``requires_grad`` is ``False``. If the tensor
        is not part of the computation graph, the written gradient will be ``null``, and no image/binary file
        will be written for that tensor regardless of the ``dump_type`` setting.

        Args:
            path: Path where the gradient data will be inserted. See :meth:`add_tensor_data` for more details.
            data: The tensor data for which to dump the gradients.
            dump_type: The type of dump to use
            dump_type_override: A dictionary mapping names to dump types.
                If a name is present in the dictionary, the dump type for all gradients of tensors with that
                name in the path (i.e. either the name itself or the name of a parent along the path) will be
                overridden with the value in the dictionary. If multiple names match the path, the match
                closest to the tensor (i.e. further inside the structure) is used.
                If ``None``, no override is applied.
            permute_grad_axes: Permutation of axes to apply to the gradient data.
                If ``None``, no permutation is applied.
            permute_grad_axes_override: A dictionary mapping names to permute axes.
                If a name is present in the dictionary, the permute axes for all gradients of tensors with
                that name in the path (i.e. either the name itself or the name of a parent along the path)
                will be overridden with the value in the dictionary. If multiple names match the path, the
                match closest to the tensor (i.e. further inside the structure) is used.
                If ``None``, no override is applied.
            exclude: List of entries to exclude from the dump. There entries are specified by names and may
                apply to any level of the data structure.
        '''
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def set_dump_type_for_all(
        self, dump_type: 'TensorDumper.Type', include_tensors: bool = True, include_grads: bool = True
    ):
        '''Set the dump type for all tensors and gradients.

        This method is e.g. useful to quickly change the dump type to ``Type.JSON`` to generate reference data
        for comparison (using :meth:`compare_to_dumped_data`) without the need to go through the code and change the dump type
        for each tensor manually.

        Important:
            This method can sets the dumping type for the data which is already added.
            The dump type of data which is added after this method is called will not be affected.

        Args:
            dump_type: The type of dump to use
            include_tensors: Whether to include tensors in the dump
            include_grads: Whether to include gradients in the dump
        '''
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def dump(self, dump_if_empty: bool = True):
        '''Dump the data to the dump directory.

        Note:
            Setting ``dump_if_empty`` to ``False`` is useful to not count iterations where no data is dumped
            as a separate iteration.

        Args:
            dump_if_empty: If ``True``, the data will be dumped even if it is empty.
                If ``False``, the data will not be dumped if it is empty, and the dump count will not be
                incremented.
        '''
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def compare_to_dumped_data(
        self,
        eps_numerical_data: float = 1e-6,
        num_errors_per_tensor_to_show: int = 1,
        allow_missing_data_in_current: bool = False,
        allow_missing_data_in_previous: bool = False,
        as_warning: bool = False,
        compare_if_empty: bool = True,
    ):
        '''Compare the data to previously dumped data.

        In case of a mismatch, a ``ValueError`` is raised with a detailed error message.

        Important:
            Only comparisons to data stored in the JSON format (Type.JSON) are supported.
            Therefore, the reference data must be stored with the ``Type.JSON`` both
            when generating the reference data and when comparing to it.

            An easy way to ensure that the reference data is stored in the JSON format without
            modifying multiple places in the code is to call :meth:`set_dump_type_for_all`
            when generating the reference data.

        Important:
            The `compare_if_empty` parameter needs to be consistent with the `dump_if_empty` parameter of
            :meth:`dump` of the calls to :meth:`dump` which are used to dump the reference data to compare
            to.

        Note:
            The comparison can be set to allow missing keys in the current and/or reference data by setting
            ``allow_missing_data_in_current`` and/or ``allow_missing_data_in_previous`` to ``True``.
            This is e.g. useful if the current data is based on an implementation in progress, so that some
            of the data is not yet available, or if the current run produces additional data which is not
            needed for the comparison (and not present in the reference).

        Args:
            eps_numerical_data: The numerical tolerance for the comparison of numerical data.
            num_errors_per_tensor_to_show: The number of most significant errors to show per tensor.
            allow_missing_data_in_current: If ``True``, the comparison will not raise an error if the current
                data is missing some keys which are present in the reference data.
            allow_missing_data_in_previous: If ``True``, the comparison will not raise an error if the
                reference data is missing some keys which are present in the current data.
            as_warning: If ``True``, no error is raised in case of a mismatch and instead, a warning is
                printed. If ``False``, an error is raised.
            compare_if_empty: If ``True``, the comparison will be performed even if the current data is empty.
                If ``False``, the comparison will not be performed if the current data is empty, and the dump
                count will not be incremented.
        '''
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def set_gradients(self, function_values: Union[torch.Tensor, List[torch.Tensor]]):
        '''Set gradients for the tensors in the dump.

        The gradients are computed using :func:`torch.autograd.grad`, and do not influence the gradients
        computed elsewhere (e.g. in the training loop).

        This method must be called before dumping if :meth:`add_grad_data` was called since the last dump.

        Args:
            function_values: The value(s) of the function(s) to compute the gradients for.
                This can be a single tensor or a list of tensors.
        '''
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def reset_dump_count(self):
        '''Reset the dump count.

        This method can be used to reset the dump count to 0.
        This is useful for debugging (e.g. when comparing to previously dumped data) to start from the first
        dump.

        See Also:

            This method is equivalent to calling :meth:`set_dump_count` with a value of 0. Please see
            the documentation of :meth:`set_dump_count` for more details.

        '''
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def set_dump_count(self, count: int):
        '''Set the dump count.

        This method can be used to set the dump count to a specific value.
        This is useful for debugging (e.g. when comparing to previously dumped data) to jump to a specific
        iteration.

        Note:
            If any actions are registered to be performed after a given number of dumps, they will be
            triggered if the count corresponds to the number of dumps set.

        Args:
            count: The dump count to set.
        '''
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def perform_after_dump_count(self, count: int, action: Callable[[], None]):
        '''Register an action to be performed after a given number of dumps.

        The action will be performed after the dump is completed.

        This can e.g. be used to automatically exit the program after a given number of iterations have
        been dumped (by passing the :func:`exit`-function as the action).

        Important:
            The action is performed after the dump count reaches the given ``count`` value.
            If :meth:`set_dump_count` is called, the dump count is adjusted to a given value,
            and this also influences when the action is performed. For example, if :meth:`set_dump_count` is
            called with a value of 3 and an action is registered to be performed after 5 dumps, the action
            will be performed after another 2 dumps.

        Important:
            This method can be called multiple times with the same count.
            In this case, the action will be overwritten.

        Note that as in case of other methods, this method has no effect if the TensorDumper is not enabled.

        Args:
            count: The number of dumps after which the action should be performed.
            action: The action to perform.
        '''
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def register_custom_converter(self, data_type: type, converter_func: Callable):
        '''Register a custom converter for a given data type.

        This method can be used to register a custom converter function for a given data type.
        The converter function must take a single argument of type ``data_type`` and return one of the following,
        or a nested list/dict structure containing elements of the following types:

          - either a JSON-serializable object,
          - or a tensor,
          - or a numpy array
          - or an object for which a custom converter is registered

        The conversion is performed iteratively, so that chains of conversions can be followed through.

        The conversion is performed before any other processing steps. This means that if the converter returns
        tensors, these are handled in the same way as tensors which are directly added to the dumper.

        Note:
            This is useful when the data to dump in not JSON-serializable by default. This may e.g. be the case
            for custom data types which are used in the training.

        Args:

            data_type: The type of the data to convert.
            converter_func: The function to use for converting the data.

        '''
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def enable_ragged_batch_dumping(self, as_per_sample: bool = False):
        '''Enable dumping of :class:`RaggedBatch` data.

        Note:
            It is possible to dump some :class:`RaggedBatch` data as per sample, and some as a :class:`RaggedBatch`
            structure.
            This can be achieved by calling this method multiple times with different values for ``as_per_sample``,
            before adding the data which should be dumped with the desired format.

        Args:
            as_per_sample: If ``True``, the :class:`RaggedBatch` data is dumped as per sample. Otherwise, it is dumped
                as a :class:`RaggedBatch` structure.
        '''
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def run_if_enabled(self, func: Callable[[], None]):
        '''Run a function if the TensorDumper is enabled.

        This method can be used to run a function only if the TensorDumper is enabled.
        This is useful to avoid running code which is only relevant for debugging.

        The typical use-case for this method is the dumping of data which needs
        to be pre-processed first (e.g. drawing of bounding boxes into an image).
        This is done as follows:

          - Encapsulate the pre-processing logic in a function (inside the function
            which uses the dumper). Note that this means that ``func`` will enclose
            the data accessible in that function and therefore does not need to have
            any arguments. The function ``func`` should

              - Perform any debugging-related pre-processing needed
              - Add the pre-processed data to the dump (e.g. using :meth:`add_tensor_data`)

          - Call :meth:`run_if_enabled` with the function ``func`` as its argument. This will ensure
            that the pre-processing is only performed if the dumper is enabled. Otherwise, the
            pre-processing is omitted, and there is no overhead (apart from calling an empty function).

        Args:
            func: The function to run. The function must take no arguments.
        '''
        pass

    # ===== Methods for Setting the Doc-strings of the Enabled Variants of the Methods =====

    @classmethod
    def _SET_DOCSTRINGS_OF_ENABLED_METHOD_VARIANTS(cls):
        '''Set the docstrings of the enabled method variants.

        This is done to ensure that the correct docstring is present in the methods
        once the TensorDumper is enabled, and the original (disabled) methods are
        replaced by the enabled variants.
        '''
        cls._set_dump_is_compare_enabled.__doc__ = cls.set_dump_is_compare.__doc__
        cls._push_range_enabled.__doc__ = cls.push_range.__doc__
        cls._pop_range_enabled.__doc__ = cls.pop_range.__doc__
        cls._add_tensor_data_enabled.__doc__ = cls.add_tensor_data.__doc__
        cls._add_grad_data_enabled.__doc__ = cls.add_grad_data.__doc__
        cls._set_dump_type_for_all_enabled.__doc__ = cls.set_dump_type_for_all.__doc__
        cls._dump_enabled.__doc__ = cls.dump.__doc__
        cls._compare_to_dumped_data_enabled.__doc__ = cls.compare_to_dumped_data.__doc__
        cls._set_gradients_enabled.__doc__ = cls.set_gradients.__doc__
        cls._reset_dump_count_enabled.__doc__ = cls.reset_dump_count.__doc__
        cls._set_dump_count_enabled.__doc__ = cls.set_dump_count.__doc__
        cls._perform_after_dump_count_enabled.__doc__ = cls.perform_after_dump_count.__doc__
        cls._register_custom_converter_enabled.__doc__ = cls.register_custom_converter.__doc__
        cls._enable_ragged_batch_dumping_enabled.__doc__ = cls.enable_ragged_batch_dumping.__doc__
        cls._run_if_enabled_enabled.__doc__ = cls.run_if_enabled.__doc__

    # ===== Enabled Variants of the Methods =====

    def _set_dump_is_compare_enabled(
        self,
        eps_numerical_data: float = 1e-6,
        num_errors_per_tensor_to_show: int = 1,
        allow_missing_data_in_current: bool = False,
        allow_missing_data_in_previous: bool = False,
        as_warning: bool = False,
    ):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        self._dump_is_compare = True
        self._dump_is_compare_params = {
            "eps_numerical_data": eps_numerical_data,
            "num_errors_per_tensor_to_show": num_errors_per_tensor_to_show,
            "allow_missing_data_in_current": allow_missing_data_in_current,
            "allow_missing_data_in_previous": allow_missing_data_in_previous,
            "as_warning": as_warning,
        }

    def _push_range_enabled(self, range_name: Union[str, Callable[[], str]]):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        if callable(range_name):
            range_name = range_name()
        self._range_stack.append(range_name)

    def _pop_range_enabled(self):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        self._range_stack.pop()

    def _add_tensor_data_enabled(
        self,
        path: str,
        data: Union[TensorDataStructure, Callable[[], TensorDataStructure]],
        dump_type: 'TensorDumper.Type',
        dump_type_override: OptionalTypeDict = None,
        permute_axes: OptionalSequence = None,
        permute_axes_override: OptionalPermuteDict = None,
        exclude: Union[Sequence[str], None] = None,
    ):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        if len(self._range_stack) > 0:
            path = ".".join(self._range_stack) + "." + path
        if callable(data):
            data = data()
        if exclude is not None:
            data = TensorDumper._exclude_elements(data, exclude)
        if len(self._custom_converters) > 0:
            data = TensorDumper._get_with_custom_converters_applied(data, self._custom_converters)
        # If the global dump type is set, it overrides everything else.
        if self._dump_type_for_all is not None:
            dump_type = self._dump_type_for_all
            dump_type_override = None
        data_with_format = TensorDumper._format_data_elements(
            data, dump_type, dump_type_override, permute_axes, permute_axes_override
        )
        TensorDumper._insert_at_path(self._tensor_struct, path, data_with_format)

    def _add_grad_data_enabled(
        self,
        path: str,
        data: Union[TensorDataStructure, Callable[[], TensorDataStructure]],
        dump_type: 'TensorDumper.Type',
        dump_type_override: OptionalTypeDict = None,
        permute_grad_axes: OptionalSequence = None,
        permute_grad_axes_override: OptionalPermuteDict = None,
        exclude: Union[Sequence[str], None] = None,
    ):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        if len(self._range_stack) > 0:
            path = ".".join(self._range_stack) + "." + path
        if callable(data):
            data = data()
        if exclude is not None:
            data = TensorDumper._exclude_elements(data, exclude)
        if len(self._custom_converters) > 0:
            data = TensorDumper._get_with_custom_converters_applied(data, self._custom_converters)
        # If the global dump type is set, it overrides everything else.
        if self._dump_type_for_all is not None:
            dump_type = self._dump_type_for_all
            dump_type_override = None
        for_grads_with_format = TensorDumper._format_data_elements(
            data, dump_type, dump_type_override, permute_grad_axes, permute_grad_axes_override
        )
        # with_grads_enabled = TensorDumper._enable_grad(for_grads_with_format)
        TensorDumper._insert_at_path(self._grad_struct, path, for_grads_with_format)

    def _set_dump_type_for_all_enabled(
        self, dump_type: 'TensorDumper.Type', include_tensors: bool = True, include_grads: bool = True
    ):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        # Store the global dump type for the future
        self._dump_type_for_all = dump_type

        # Apply the dump type to the already set data.
        def set_dump_type(data: TensorDumper._TensorWithFormat) -> TensorDumper._TensorWithFormat:
            data.dump_type = dump_type
            return data

        if include_tensors:
            self._tensor_struct = TensorDumper._traverse_and_apply(
                self._tensor_struct, TensorDumper._TensorWithFormat, set_dump_type
            )
        if include_grads:
            self._grad_struct = TensorDumper._traverse_and_apply(
                self._grad_struct, TensorDumper._TensorWithFormat, set_dump_type
            )

    def _dump_enabled(self, dump_if_empty: bool = True):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''

        if self._dump_is_compare:
            return self.compare_to_dumped_data(
                eps_numerical_data=self._dump_is_compare_params["eps_numerical_data"],
                num_errors_per_tensor_to_show=self._dump_is_compare_params["num_errors_per_tensor_to_show"],
                allow_missing_data_in_current=self._dump_is_compare_params["allow_missing_data_in_current"],
                as_warning=self._dump_is_compare_params["as_warning"],
                compare_if_empty=dump_if_empty,
            )

        # If dumping is disabled for empty dumps, return early (and don't increment the dump count).
        if not dump_if_empty and len(self._tensor_struct) == 0 and len(self._grad_struct) == 0:
            return

        self._dump_struct(self._tensor_struct, "tensors")
        if len(self._grad_struct) > 0:
            if not self._grad_computed:
                raise ValueError(
                    "Gradients were not computed. Call `set_gradients` before dumping if any gradients are included."
                )
            self._dump_struct(self._grad_struct, "grads")
        self._tensor_struct = {}
        self._grad_struct = {}
        self._grad_computed = False
        self._dump_count += 1
        if self._dump_count in self._after_dump_count_actions:
            self._after_dump_count_actions[self._dump_count]()

    def _compare_to_dumped_data_enabled(
        self,
        eps_numerical_data: float = 1e-6,
        num_errors_per_tensor_to_show: int = 1,
        allow_missing_data_in_current: bool = False,
        allow_missing_data_in_previous: bool = False,
        as_warning: bool = False,
        compare_if_empty: bool = True,
    ):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        # If comparison is disabled for empty data and the current data is empty, return early (and don't
        # increment the dump count).
        if not compare_if_empty and len(self._tensor_struct) == 0 and len(self._grad_struct) == 0:
            return
        try:
            # Create config from parameters
            config = TensorDumper._ComparisonConfig(
                eps_numerical_data=eps_numerical_data,
                num_errors_per_tensor_to_show=num_errors_per_tensor_to_show,
                allow_missing_data_in_current=allow_missing_data_in_current,
                allow_missing_data_in_previous=allow_missing_data_in_previous,
            )

            is_tensor_data_consistent = self._compare_to_dumped_data(
                self._tensor_struct,
                "tensors",
                config,
                as_warning,
            )
            has_grad_data = len(self._grad_struct) > 0
            if has_grad_data:
                if not self._grad_computed:
                    raise ValueError(
                        "Gradients were not computed. Call `set_gradients` before comparing to previously dumped data."
                    )
                is_grad_data_consistent = self._compare_to_dumped_data(
                    self._grad_struct,
                    "grads",
                    config,
                    as_warning,
                )
            else:
                is_grad_data_consistent = True
            if is_tensor_data_consistent:
                print(
                    f"`TensorDumper:` Tensor data is consistent with previously dumped data for dump {self._dump_count}."
                )
            if has_grad_data and is_grad_data_consistent:
                print(
                    f"`TensorDumper:` Grad data is consistent with previously dumped data for dump {self._dump_count}."
                )
        finally:
            # Keep compare and dump interchangeable:
            # - always clear buffered data, even if comparison raises
            # - always advance dump_count to stay in sync with a `dump()`-based run loop
            self._tensor_struct = {}
            self._grad_struct = {}
            self._grad_computed = False
            self._dump_count += 1
            if self._dump_count in self._after_dump_count_actions:
                self._after_dump_count_actions[self._dump_count]()

    def _set_gradients_enabled(self, function_values: Union[torch.Tensor, List[torch.Tensor]]):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        self._grad_struct = self._compute_and_set_gradients(self._grad_struct, function_values)
        self._grad_computed = True

    def _reset_dump_count_enabled(self):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        self.set_dump_count(0)

    def _set_dump_count_enabled(self, count: int):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        self._dump_count = count
        if self._dump_count in self._after_dump_count_actions:
            self._after_dump_count_actions[self._dump_count]()

    def _perform_after_dump_count_enabled(self, count: int, action: Callable):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        self._after_dump_count_actions[count] = action

    def _register_custom_converter_enabled(self, data_type: type, converter_func: Callable):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        self._custom_converters[data_type] = converter_func

    def _enable_ragged_batch_dumping_enabled(self, as_per_sample: bool = False):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''

        # Check if the RaggedBatch class is available and raise an error if not.
        try:
            from accvlab.batching_helpers import RaggedBatch
        except ImportError:
            raise ImportError(
                "The `accvlab.batching_helpers` package is not installed. Please install it to use ragged batch dumping."
            )

        # Convert to per-sample format.
        def convert_to_per_sample(data: 'RaggedBatch') -> List[torch.Tensor]:
            res = data.split()
            return res

        # Convert to RaggedBatch descriptor format.
        def convert_to_descriptor(data: 'RaggedBatch') -> Dict[str, Union[torch.Tensor, int]]:
            res = {
                "data": data.tensor,
                "sample_sizes": data.sample_sizes,
                "non_uniform_dim": data.non_uniform_dim,
                "num_batch_dims": data.num_batch_dims,
            }
            return res

        # Register the selected converter.
        if as_per_sample:
            self.register_custom_converter(RaggedBatch, convert_to_per_sample)
        else:
            self.register_custom_converter(RaggedBatch, convert_to_descriptor)

    def _run_if_enabled_enabled(self, func: Callable):
        '''TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        '''
        # Note that the `_run_if_enabled_enabled` method is only called if the TensorDumper is enabled. Therefore,
        # no further checks are needed and we can directly call the passed function.
        func()

    # ===== Private Helper Methods =====

    @staticmethod
    def _exclude_elements(
        data: TensorDataStructure,
        exclude: Sequence[str],
    ) -> TensorDataStructure:
        if isinstance(data, Dict):
            return {
                key: TensorDumper._exclude_elements(data[key], exclude)
                for key in data.keys()
                if not key in exclude
            }
        else:
            return data

    @staticmethod
    def _split_multi_image_data(
        data: FormattedDataStructure,
    ) -> FormattedDataStructure:

        def split_dims_inner(
            data: torch.Tensor, num_dims_to_split: int, wrapper: TensorDumper._TensorWithFormat
        ) -> Union[TensorDumper._TensorWithFormat, List]:
            if num_dims_to_split == 0:
                wrapper = copy.deepcopy(wrapper)
                wrapper.tensor = data
                return wrapper
            else:
                res = [
                    split_dims_inner(data[i], num_dims_to_split - 1, wrapper) for i in range(data.shape[0])
                ]
                return res

        # Create a wrapper for the resulting tensors. Note that:
        # - The tensor is set to None, as it will be replaced by the split tensors
        # - The permutation is set to None, as it will be applied as part of this function
        #   for the tensors which are split, and the wrapper is not used for any other tensors
        def get_image_wrapper(dump_type: 'TensorDumper.Type') -> TensorDumper._TensorWithFormat:
            return TensorDumper._TensorWithFormat(None, dump_type, None)

        def split_dims(data: TensorDumper._TensorWithFormat) -> Union[TensorDumper._TensorWithFormat, List]:
            if TensorDumper.Type.is_image(data.dump_type):
                image_num_dims = 2 if data.dump_type == TensorDumper.Type.IMAGE_I else 3
                if data.tensor.ndim > image_num_dims:
                    data_to_split = data.tensor
                    if data.permute_axes is not None:
                        data_to_split = torch.permute(data_to_split, data.permute_axes)
                    res = split_dims_inner(
                        data_to_split, data.tensor.ndim - image_num_dims, get_image_wrapper(data.dump_type)
                    )
                    return res
                else:
                    return data
            else:
                return data

        res = TensorDumper._traverse_and_apply(data, TensorDumper._TensorWithFormat, split_dims)
        return res

    def _compute_and_set_gradients(
        self,
        struct_with_tensors: TensorDataStructure,
        function_values: Union[torch.Tensor, List[torch.Tensor]],
    ) -> TensorDataStructure:
        list_of_tensors = []

        def traverse_and_collect(
            data: FormattedDataStructure,
            list_of_tensors: List[torch.Tensor],
        ):
            if isinstance(data, TensorDumper._TensorWithFormat):
                list_of_tensors.append(data.tensor)
            elif isinstance(data, Sequence) and not isinstance(data, str):
                for item in data:
                    traverse_and_collect(item, list_of_tensors)
            elif isinstance(data, Dict):
                for key in data.keys():
                    traverse_and_collect(data[key], list_of_tensors)

        def traverse_and_replace_by_grad(
            data: FormattedDataStructure,
            list_of_tensors: List[torch.Tensor],
        ) -> FormattedDataStructure:
            if isinstance(data, TensorDumper._TensorWithFormat):
                to_set = list_of_tensors.pop(0)
                if to_set is None:
                    data = None
                else:
                    data.tensor = to_set
                return data
            elif isinstance(data, Sequence) and not isinstance(data, str):
                res = []
                for item in data:
                    res.append(traverse_and_replace_by_grad(item, list_of_tensors))
                return res
            elif isinstance(data, Dict):
                res = {}
                for key in data.keys():
                    res[key] = traverse_and_replace_by_grad(data[key], list_of_tensors)
                return res
            else:
                return data

        def replace_element_not_requiring_grad(
            data: TensorDumper._TensorWithFormat,
        ) -> Union[TensorDumper._TensorWithFormat, str]:
            if not data.tensor.requires_grad:
                data = "`.requires_grad == False`"
            return data

        struct_with_tensors = TensorDumper._traverse_and_apply(
            struct_with_tensors, TensorDumper._TensorWithFormat, replace_element_not_requiring_grad
        )
        traverse_and_collect(struct_with_tensors, list_of_tensors)
        if isinstance(function_values, torch.Tensor):
            function_values = [function_values]
        grads = torch.autograd.grad(function_values, list_of_tensors, retain_graph=True, allow_unused=True)
        grads = list(grads)
        struct_with_tensors = traverse_and_replace_by_grad(struct_with_tensors, grads)

        return struct_with_tensors

    def _get_dump_dir(self) -> str:
        res = f"{self._dump_dir}/{self._dump_count}"
        return res

    def _get_json_filename(self, type_of_struct: str) -> str:
        return f"{type_of_struct}.json"

    def _dump_struct(self, struct_to_dump: Union[Sequence, Dict], type_of_struct: str):
        struct_to_dump = TensorDumper._split_multi_image_data(struct_to_dump)
        json_struct, binary_files = TensorDumper._apply_format_and_get(struct_to_dump)
        dump_dir = self._get_dump_dir()
        self._ensure_dir_exists(dump_dir)
        json_file_name = self._get_json_filename(type_of_struct)

        json.dump(
            json_struct,
            open(f"{dump_dir}/{json_file_name}", "w"),
            cls=TensorDumper._CustomEncoder,
            indent=2,
        )
        for file_name, file_data in binary_files.items():
            dump_type = file_data["dump_type"]
            if dump_type == TensorDumper.Type.BINARY:
                TensorDumper._dump_binary(f"{dump_dir}/[{json_file_name}]{file_name}", file_data["data"])
            elif TensorDumper.Type.is_image(dump_type):
                TensorDumper._dump_image(
                    f"{dump_dir}/[{json_file_name}]{file_name}", file_data["data"], dump_type
                )
            elif dump_type == TensorDumper.Type.PICKLE:
                TensorDumper._dump_pickle(f"{dump_dir}/[{json_file_name}]{file_name}", file_data["data"])
            else:
                raise ValueError(f"Unsupported file type: {file_name}")

    @staticmethod
    def _walk_and_compare(
        dumped_data: Union[Any, Sequence, Dict],
        json_struct_to_compare: Union[Any, Sequence, Dict],
        non_tensor_struct: Optional[Union[Any, Sequence, Dict]],
        curr_path: str,
        is_parent_tensor: bool,
        config: 'TensorDumper._ComparisonConfig',
    ) -> List['TensorDumper._ComparisonError']:

        class ComparisonError:
            def __init__(self, message: str, weight: float):
                self.message = message
                self.weight = weight

        def order_errors_by_weight(
            errors: List['TensorDumper._ComparisonError'],
        ) -> List['TensorDumper._ComparisonError']:
            return sorted(errors, key=lambda error: error.weight, reverse=True)

        def get_path_to_show(path: str) -> str:
            if len(path) == 0:
                return ". (i.e. root)"
            elif ":" in path:
                return (path.replace(":", "[") + "]")[1:]
            else:
                return path[1:]

        def get_child_path(curr_path: str, key: str, is_self_tensor: bool, is_parent_tensor: bool) -> str:
            if not is_parent_tensor and is_self_tensor:
                child_path = f"{curr_path}:{key}"
            elif is_self_tensor:
                child_path = f"{curr_path},{key}"
            else:
                child_path = f"{curr_path}.{key}"
            return child_path

        if isinstance(dumped_data, Dict):
            res = []
            is_self_tensor = non_tensor_struct is None
            for key in dumped_data.keys():
                if not key in json_struct_to_compare:
                    if not config.allow_missing_data_in_current:
                        res.append(
                            ComparisonError(
                                f"  Missing key '{key}' at path: {get_path_to_show(curr_path)} in current "
                                f"data but present in reference",
                                math.inf,
                            )
                        )
                    continue
                is_child_tensor = is_self_tensor or not key in non_tensor_struct
                non_tensor_struct_child = non_tensor_struct[key] if not is_child_tensor else None
                r = TensorDumper._walk_and_compare(
                    dumped_data[key],
                    json_struct_to_compare[key],
                    non_tensor_struct_child,
                    get_child_path(curr_path, key, is_self_tensor, is_parent_tensor),
                    is_self_tensor,
                    config,
                )
                if not is_self_tensor and is_child_tensor:
                    r = order_errors_by_weight(r)
                    r = r[: config.num_errors_per_tensor_to_show] if len(r) > 0 else []
                res.extend(r)
            if not config.allow_missing_data_in_previous:
                for key in json_struct_to_compare.keys():
                    if not key in dumped_data:
                        res.append(
                            ComparisonError(
                                f"  Extra key '{key}' at path: {get_path_to_show(curr_path)} in current "
                                f"data but not present in reference",
                                math.inf,
                            )
                        )
            return res
        elif isinstance(dumped_data, Sequence):
            res = []
            is_self_tensor = non_tensor_struct is None
            if len(dumped_data) != len(json_struct_to_compare):
                res.append(
                    ComparisonError(
                        f"  Length mismatch at path: {get_path_to_show(curr_path)}\n"
                        f"    Dumped data: {dumped_data}\n    Struct to compare: {json_struct_to_compare}",
                        math.inf,
                    )
                )
                return res
            for i in range(len(dumped_data)):
                is_child_tensor = is_self_tensor or not i in non_tensor_struct
                non_tensor_struct_child = non_tensor_struct[i] if not is_child_tensor else None
                r = TensorDumper._walk_and_compare(
                    dumped_data[i],
                    json_struct_to_compare[i],
                    non_tensor_struct_child,
                    get_child_path(curr_path, i, is_self_tensor, is_parent_tensor),
                    is_self_tensor,
                    config,
                )
                if not is_self_tensor and is_child_tensor:
                    r = order_errors_by_weight(r)
                    r = r[: config.num_errors_per_tensor_to_show] if len(r) > 0 else []
                res.extend(r)
            return res
        elif isinstance(dumped_data, numbers.Number) and isinstance(json_struct_to_compare, numbers.Number):
            if abs(dumped_data - json_struct_to_compare) > config.eps_numerical_data:
                difference = abs(json_struct_to_compare - dumped_data)
                return [
                    ComparisonError(
                        f"  Numerical mismatch at path: {get_path_to_show(curr_path)}\n"
                        f"    Dumped data: {dumped_data}\n    Struct to compare: {json_struct_to_compare}\n"
                        f"    Difference (current - dumped): {json_struct_to_compare - dumped_data}",
                        difference,
                    )
                ]
            else:
                return []
        else:
            if dumped_data != json_struct_to_compare:
                return [
                    ComparisonError(
                        f"  Mismatch at path: {get_path_to_show(curr_path)}\n    Dumped data: {dumped_data}\n"
                        f"    Struct to compare: {json_struct_to_compare}",
                        0.0,
                    )
                ]
            else:
                return []

    def _compare_to_dumped_data(
        self,
        struct_to_compare: Union[Sequence, Dict],
        type_of_struct: str,
        config: 'TensorDumper._ComparisonConfig',
        as_warning: bool = False,
    ) -> bool:
        non_tensor_struct = TensorDumper._get_non_tensor_structure(struct_to_compare)
        json_struct, binary_files = TensorDumper._apply_format_and_get(struct_to_compare)
        # Dump json_struct to a string and read back to ensure format consistency to previously dumped data
        json_struct_str = json.dumps(json_struct, cls=TensorDumper._CustomEncoder, indent=2)
        json_struct = json.loads(json_struct_str)
        if len(binary_files) > 0:
            first_file = list(binary_files.keys())[0]
            first_file_split_at_extension = first_file.rsplit(".", 1)
            first_file_no_extension, extension = first_file_split_at_extension
            raise ValueError(
                f"Cannot compare to dumped data with binary or image format.\nFound image or binary format "
                f"at: {first_file_no_extension}\nwith format: {extension}\nPlease use the JSON format when "
                f"dumping the data for comparison."
            )

        json_file_name = self._get_json_filename(type_of_struct)
        json_file_path = f"{self._get_dump_dir()}/{json_file_name}"
        try:
            with open(json_file_path, "r") as f:
                dumped_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No previously dumped data found for [{type_of_struct}] data for dump {self._dump_count}\n"
                f"under file path: {json_file_path}.\nDump the data first before comparing to previously "
                f"dumped data."
            )

        res_errors = self._walk_and_compare(
            dumped_data,
            json_struct,
            non_tensor_struct,
            "",
            False,
            config,
        )
        if len(res_errors) > 0:
            error_message = "\n".join([error.message for error in res_errors])
            error_message = (
                f"NOTE: The following errors were found for the dumped [{type_of_struct}] data for dump "
                f"{self._dump_count}.\n"
                f"      Up to {config.num_errors_per_tensor_to_show} most significant errors are shown per "
                f"tensor.\n" + error_message
            )
            error_message = (
                f"Comparison of data with previously dumped data failed for [{type_of_struct}] data for dump "
                f"{self._dump_count}.\n{error_message}"
            )
            if as_warning:
                warnings.warn(error_message)
            else:
                raise ValueError(error_message)
            return False
        else:
            return True

    @staticmethod
    def _dump_binary(file_name: str, file_data: torch.Tensor, add_meta_info: bool = True):
        if add_meta_info:
            file_meta_info = {
                "file_data_shape": file_data.shape,
                "file_data_dtype": file_data.dtype,
            }
            json.dump(
                file_meta_info, open(f"{file_name}.meta.json", "w"), indent=2, cls=TensorDumper._CustomEncoder
            )
        data = file_data.detach().cpu().contiguous().numpy().tobytes()
        with open(file_name, 'wb') as f:
            f.write(data)

    @staticmethod
    def _dump_image(
        file_name: str, file_data: torch.Tensor, dump_type: 'TensorDumper.Type', add_meta_info: bool = True
    ):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV (cv2) is not installed, but is required for dumping images via TensorDumper.\n"
                "Please install the ACCV-Lab packages with optional dependencies enabled. "
                "For details, see the Installation Guide (section on installation with optional "
                "dependencies)."
            ) from exc

        def ensure_image_range_ang_get_orig_range(image: torch.Tensor) -> tuple[torch.Tensor, List[float]]:
            min_val = image.min()
            max_val = image.max()
            offset = -min_val
            diff = max_val - min_val
            scaling = (255 / diff) if diff > 0 else 1
            res = (image + offset) * scaling
            return res, [min_val, max_val]

        assert (dump_type == TensorDumper.Type.IMAGE_I and file_data.ndim == 2) or (
            dump_type != TensorDumper.Type.IMAGE_I and file_data.ndim == 3
        ), (
            f"Number of image dimensions does not match the dump type for file:\n{file_name}.\nImage data "
            f"has {file_data.ndim} dimensions; dump type is {dump_type}."
        )
        assert dump_type == TensorDumper.Type.IMAGE_I or file_data.shape[-1] == 3, (
            f"Color image must have 3 channels, but image to be dumped to:\n{file_name}\nhas "
            f"{file_data.shape[-1]} channels."
        )

        file_data, orig_range = ensure_image_range_ang_get_orig_range(file_data)
        file_data = file_data.detach().cpu().contiguous().numpy().astype(np.uint8)

        if dump_type == TensorDumper.Type.IMAGE_RGB:
            file_data = cv2.cvtColor(file_data, cv2.COLOR_RGB2BGR)

        cv2.imwrite(file_name, file_data)

        if add_meta_info:
            if dump_type == TensorDumper.Type.IMAGE_RGB:
                image_format = "RGB"
            elif dump_type == TensorDumper.Type.IMAGE_BGR:
                image_format = "BGR"
            elif dump_type == TensorDumper.Type.IMAGE_I:
                image_format = "Intensity"
            else:
                raise ValueError(f"Unsupported image format: {dump_type}")

            file_meta_info = {
                "original_range": orig_range,
                "image_format": image_format,
            }
            json.dump(
                file_meta_info, open(f"{file_name}.meta.json", "w"), cls=TensorDumper._CustomEncoder, indent=2
            )

    @staticmethod
    def _dump_pickle(file_name: str, file_data: Any):
        with open(file_name, "wb") as f:
            pickle.dump(file_data, f)

    @staticmethod
    def _ensure_dir_exists(dir_name: str):
        os.makedirs(dir_name, exist_ok=True)

    @staticmethod
    def _traverse_and_apply(
        data: GenericDataStructure,
        data_type_element: type,
        func_element: Callable,
    ) -> GenericDataStructure:
        if isinstance(data, data_type_element):
            return func_element(data)
        else:
            if isinstance(data, Sequence) and not isinstance(data, str):
                res = [
                    TensorDumper._traverse_and_apply(item, data_type_element, func_element) for item in data
                ]
                return res
            elif isinstance(data, Dict):
                res = {
                    key: TensorDumper._traverse_and_apply(data[key], data_type_element, func_element)
                    for key in data.keys()
                }
                return res
            else:
                return data

    @staticmethod
    def _traverse_remember_waypoints_and_apply(
        data: GenericDataStructure,
        data_element_type: type,
        default_param: Any,
        param_for_waypoints: Dict[str, Any],
        func_element: Callable,
    ) -> GenericDataStructure:
        if isinstance(data, data_element_type):
            return func_element(data, default_param)
        elif isinstance(data, Sequence) and not isinstance(data, str):
            return [
                TensorDumper._traverse_remember_waypoints_and_apply(
                    item, data_element_type, default_param, param_for_waypoints, func_element
                )
                for item in data
            ]
        elif isinstance(data, Dict):
            res = {}
            for key, value in data.items():
                if str(key) in param_for_waypoints:
                    res[key] = TensorDumper._traverse_remember_waypoints_and_apply(
                        value,
                        data_element_type,
                        param_for_waypoints[str(key)],
                        param_for_waypoints,
                        func_element,
                    )
                else:
                    res[key] = TensorDumper._traverse_remember_waypoints_and_apply(
                        value, data_element_type, default_param, param_for_waypoints, func_element
                    )
            return res
        else:
            return data

    @staticmethod
    def _format_data_elements(
        data: TensorDataStructure,
        dump_type: 'TensorDumper.Type',
        dump_type_override: OptionalTypeDict = None,
        permute_axes: OptionalSequence = None,
        permute_axes_override: OptionalPermuteDict = None,
    ) -> FormattedDataStructure:

        def change_permute_axes(
            data: TensorDumper._TensorWithFormat, selected_permute_axes: OptionalSequence
        ) -> TensorDumper._TensorWithFormat:
            data.permute_axes = selected_permute_axes
            return data

        # If no overrides are provided, we can just apply the default dump type and permute axes to all tensors
        if dump_type_override is None and permute_axes_override is None:
            data = TensorDumper._traverse_and_apply(
                data, torch.Tensor, lambda x: TensorDumper._TensorWithFormat(x, dump_type, permute_axes)
            )
            return data
        # If dump type overrides are provided, we need to traverse the data and apply the overrides to the
        # tensors
        elif dump_type_override is not None:
            data = TensorDumper._traverse_remember_waypoints_and_apply(
                data,
                torch.Tensor,
                dump_type,
                dump_type_override,
                lambda x, selected_dump_type: TensorDumper._TensorWithFormat(
                    x, selected_dump_type, permute_axes
                ),
            )
            # If permute axes overrides are provided additionally,
            # we need to need to make a second pass and apply the overrides
            # on the already converted data
            if permute_axes_override is not None:
                data = TensorDumper._traverse_remember_waypoints_and_apply(
                    data,
                    TensorDumper._TensorWithFormat,
                    permute_axes,
                    permute_axes_override,
                    change_permute_axes,
                )
            return data
        # The `else` case is when only permute axes overrides are provided.
        # In this case, we can apply them directly to the original data
        # and do not need a second pass.
        else:
            data = TensorDumper._traverse_remember_waypoints_and_apply(
                data,
                torch.Tensor,
                permute_axes,
                permute_axes_override,
                lambda x, selected_perm_axes: TensorDumper._TensorWithFormat(
                    x, dump_type, selected_perm_axes
                ),
            )
            return data

    @staticmethod
    def _apply_format_and_get(
        data: FormattedDataStructure,
        path: OptionalPath = None,
    ) -> tuple[Union[Any, Sequence, Dict], Dict[str, Dict[str, Union[torch.Tensor, 'TensorDumper.Type']]]]:

        def get_item_path(item_key: Union[str, int]) -> str:
            return f"{path}.{item_key}" if path is not None else str(item_key)

        res_data = None
        res_files = {}

        if isinstance(data, TensorDumper._TensorWithFormat):
            if data.permute_axes is not None:
                tensor = data.tensor.permute(data.permute_axes)
            else:
                tensor = data.tensor
            if data.dump_type == TensorDumper.Type.JSON:
                res_data = tensor
                res_files = {}
            elif data.dump_type == TensorDumper.Type.BINARY:
                res_data = f"{path}.bin"
                res_files = {res_data: {"data": tensor, "dump_type": data.dump_type}}
            elif TensorDumper.Type.is_image(data.dump_type):
                res_data = f"{path}.png"
                res_files = {res_data: {"data": tensor, "dump_type": data.dump_type}}
            elif data.dump_type == TensorDumper.Type.PICKLE:
                res_data = f"{path}.pkl"
                res_files = {res_data: {"data": tensor, "dump_type": data.dump_type}}
            else:
                raise ValueError(f"Unsupported dump type: {data.dump_type}")
            return res_data, res_files

        elif isinstance(data, Sequence) and not isinstance(data, str):
            res_data = []
            for i, item in enumerate(data):
                path_i = get_item_path(i)
                item_data, item_files = TensorDumper._apply_format_and_get(item, path_i)
                res_data.append(item_data)
                res_files.update(item_files)
            return res_data, res_files

        elif isinstance(data, Dict):
            res_data = {}
            for key, value in data.items():
                path_key = get_item_path(key)
                item_data, item_files = TensorDumper._apply_format_and_get(value, path_key)
                res_data[key] = item_data
                res_files.update(item_files)
            return res_data, res_files

        else:
            return data, {}

    def _get_non_tensor_structure(
        data: FormattedDataStructure,
    ) -> FormattedDataStructure:
        if isinstance(data, TensorDumper._TensorWithFormat):
            return None
        elif isinstance(data, Sequence) and not isinstance(data, str):
            # Note that we are using dictionaries instead of lists here,
            # because this allows us to store only non-tensor elements while
            # preserving the indices of the elements.
            res = {
                i: result
                for i, item in enumerate(data)
                if (result := TensorDumper._get_non_tensor_structure(item)) is not None
            }
            return res
        elif isinstance(data, dict):
            res = {
                key: result
                for key, value in data.items()
                if (result := TensorDumper._get_non_tensor_structure(value)) is not None
            }
            return res
        else:
            return data

    @staticmethod
    def _insert_at_path(
        data: Union[Sequence[Union[Sequence, Dict]], Dict[str, Union[Sequence, Dict]]],
        path: str,
        value: Union[
            Union[_TensorWithFormat, Any],
            Sequence[Union[_TensorWithFormat, Any, Sequence, Dict]],
            Dict[str, Union[_TensorWithFormat, Any, Sequence, Dict]],
        ],
    ):
        path_parts = path.split(".")
        curr_data = data
        parent = None
        # Make sure the path exists
        for part in path_parts:
            if isinstance(curr_data, Sequence):
                assert part.isdigit(), f"Path part {part} is not a number, but parent is a sequence"
                part = int(part)
                assert part < len(curr_data), f"Path part {part} is out of bounds (parent is a sequence)"
            elif isinstance(curr_data, Dict):
                # Convert digits to ints, but only if they are not already in the data as strings
                if not (part in curr_data):
                    curr_data[part] = {}
            parent = curr_data
            curr_data = curr_data[part]

        # Insert the new element
        # We can only insert into a dictionary
        assert isinstance(
            curr_data, Dict
        ), f"Path `{path}` points to an existing element which is not a dictionary. Cannot insert there."
        # If we are inserting a dictionary, we can insert into non-empty or empty dicts
        if isinstance(value, Dict):
            for key in value.keys():
                assert key not in curr_data, (
                    f"Path `{path}` has an existing element with key `{key}`. Cannot insert element from "
                    f"`value` with the same key."
                )
                curr_data[key] = value[key]
        # If we are inserting a tensor or a sequence, we can only insert into an empty dict
        elif isinstance(value, (TensorDumper._TensorWithFormat, Sequence)):
            assert parent is not None, f"Can only insert dictionaries at the root level."
            assert len(curr_data) == 0, (
                f"Path part `{path}` points to an existing non-empty dictionary. Cannot insert tensors or "
                "sequences as this would overwrite the existing elements."
            )
            parent[path_parts[-1]] = value
        else:
            raise ValueError(f"Unsupported data type: {type(value)}")

    @staticmethod
    def _get_with_custom_converters_applied(
        data: Union[Any, Sequence, Dict], custom_converters: Dict[type, Callable]
    ) -> Union[Any, Sequence, Dict]:

        was_changed = False

        def get_with_custom_converters_applied_inner(
            data: Union[Any, Sequence, Dict], custom_converters: Dict[type, Callable]
        ) -> Union[Any, Sequence, Dict]:
            nonlocal was_changed

            if isinstance(data, dict):
                return {
                    key: get_with_custom_converters_applied_inner(value, custom_converters)
                    for key, value in data.items()
                }
            elif isinstance(data, Sequence) and not isinstance(data, str):
                return [get_with_custom_converters_applied_inner(item, custom_converters) for item in data]
            else:
                data_type = type(data)
                if data_type in custom_converters:
                    was_changed = True
                    return custom_converters[data_type](data)
                else:
                    return data

        do_iterate = True
        while do_iterate:
            data = get_with_custom_converters_applied_inner(data, custom_converters)
            do_iterate = was_changed
            was_changed = False
        return data
