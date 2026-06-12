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

from __future__ import annotations

from typing import Type, Union, Any, Callable, Optional

from collections.abc import Iterator as ABCIterator
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchDALIGenericIterator

from torch.utils.data import DataLoader

from .sample_data_group import SampleDataGroup

try:
    from typing import override
except ImportError:
    from typing_extensions import override


class DALIStructuredOutputIterator(object):
    '''Structured access to DALI pipeline output (as a nested dict or :class:`SampleDataGroup`).

    Designed as a drop-in replacement for a :class:`torch.utils.data.DataLoader`. Optionally applies a user-defined
    lightweight post-processing function (e.g., conversions to types not supported by DALI).
    '''

    class SimpleIterator(ABCIterator):
        '''Iterator, which can e.g. be used as a drop-in replacement for a PyTorch DataLoader iterator.

        Note that a single iterator should be used at any point in time.
        if multiple iterators are used, they share the state, i.e. getting a new iterator will reset all other iterators and calling next for one iterator
        will advance all iterators by one element.
        '''

        def __init__(self, obj: DALIStructuredOutputIterator):
            self._obj = obj
            # Make sure the iteration starts anew
            self.reset()

        @override
        def __next__(self):
            '''Get the next element.'''
            return self._obj._next()

        @override
        def __iter__(self):
            '''Get the iterator (i.e. return ``self`` without re-starting the iteration).'''
            return self

        def reset(self):
            '''Reset the iterator.

            Will call :meth:`DALIStructuredOutputIterator.reset` for the parent object.
            '''
            self._obj.reset()

        def __len__(self):
            '''Get the number of elements (from parent object).'''
            return len(self._obj)

    def __init__(
        self,
        num_batches_in_epoch: int,
        pipeline: Pipeline,
        sample_data_structure_blueprint: SampleDataGroup,
        contained_dataset: Optional[Any] = None,
        dali_generic_iterator_class: Union[
            Type[PyTorchDALIGenericIterator], Any
        ] = PyTorchDALIGenericIterator,
        convert_sample_data_group_to_dict: bool = True,
        post_process_func: Optional[
            Callable[[Union[SampleDataGroup, dict]], Union[SampleDataGroup, dict]]
        ] = None,
    ):
        '''
        Args:
            num_batches_in_epoch: Number of batches in an epoch. Note that this value is only used
                to output if ``len(obj)`` is called. It is not used internally and is added here to
                ensure drop-in compatibility with :class:`torch.utils.data.DataLoader`.
            pipeline: DALI pipeline object.
            sample_data_structure_blueprint: Blueprint for the output data structure.
            contained_dataset: Dataset object which will be exposed via :attr:`dataset`
                (mirrors PyTorch ``DataLoader`` behavior). Can be a PyTorch ``Dataset`` or any other
                compatible object. Note that this object is not used internally. Also see :attr:`dataset`.
            dali_generic_iterator_class: Class for the internal DALI generic iterator. Follows the
                :class:`PyTorchDALIGenericIterator` interface but may emit tensors for other frameworks.
                Defaults to :class:`PyTorchDALIGenericIterator`.
            convert_sample_data_group_to_dict: If ``True``, convert output :class:`SampleDataGroup` to a
                nested :class:`dict`. Ensures drop-in compatibility with ``DataLoader`` when no
                post-processing function is provided.
            post_process_func: Optional post-processing function for the output. This can be e.g. used to
                convert data to types not supported by DALI.
                or perform other light-weight steps. The input is a :class:`SampleDataGroup` object if
                ``convert_sample_data_group_to_dict == False`` and a :class:`dict` otherwise.
                Note that this function is executed when the data is accessed in the thread accessing the
                data (typically the thread performing the training). Therefore, this function should be kept
                lightweight to avoid performance penalties.
        '''

        flattened_names = sample_data_structure_blueprint.field_names_flat
        self._num_batches_in_epoch = num_batches_in_epoch
        self._iterator = dali_generic_iterator_class(pipeline, list(flattened_names))
        self._blueprint = sample_data_structure_blueprint.get_empty_like_self()
        self._contained_dataset = contained_dataset
        self._convert_sample_data_group_to_dict = convert_sample_data_group_to_dict
        self._post_process_func = post_process_func

    def __iter__(self) -> DALIStructuredOutputIterator.SimpleIterator:
        '''Get an iterator.

        Note that a single iterator should be used at any point in time.
        if multiple iterators are used, they share the state, i.e. getting a new iterator will reset all other iterators and calling next for one iterator
        will advance all iterators by one element.
        '''
        res = self.SimpleIterator(self)
        return res

    def _next(self) -> Union[SampleDataGroup, dict]:
        data = self._iterator.__next__()
        data_structured = self._blueprint.get_empty_like_self()
        data_structured.set_data_from_dali_generic_iterator_output(data, 0)
        if self._convert_sample_data_group_to_dict:
            data_structured = data_structured.to_dictionary()

        if self._post_process_func is not None:
            data_structured = self._post_process_func(data_structured)

        return data_structured

    def reset(self):
        '''Reset the current iteration progress (start over from the beginning).

        Note that this will reset iterators of the object as well.
        '''
        self._iterator.reset()

    @property
    def sample_data_structure_blueprint(self) -> SampleDataGroup:
        '''Get the output data structure blueprint.

        The blueprint is a :class:`SampleDataGroup` representing the same nested data format as the output,
        without the actual data. See :class:`SampleDataGroup` for details.

        '''
        return self._blueprint.get_empty_like_self()

    @property
    def internal_iterator(self) -> Union[PyTorchDALIGenericIterator, Any]:
        '''Get the actual DALI iterator used to access the output data internally.

        Note that by default, this is a :class:`nvidia.dali.plugin.pytorch.DALIGenericIterator`. However, this
        can be changed in the constructor and in this case, the returned object will be of the type specified
        in the constructor.
        '''
        return self._iterator

    @property
    def dataset(self) -> Any:
        '''Get the dataset object.

        This is the dataset object set in the constructor (if any). If not set, this will return the object
        for which it is called. This property is used for compatibility with
        :class:`torch.utils.data.DataLoader`.
        '''
        if self._contained_dataset is None:
            return self
        else:
            return self._contained_dataset

    def __len__(self):
        '''Number of available batches.

        Important:
            This value is set manually in the constructor, and only used to output it here.
            This is done as it is a part of the :class:`torch.utils.data.DataLoader` interface.
            The value my be not the actual number of batches in the epoch, e.g. for non-epoch based
            pipelines.
        '''
        return self._num_batches_in_epoch

    @classmethod
    def CreateAsDataLoaderObject(cls, *args, **kwargs):
        from ._dali_structured_output_iterator_data_loader_wrapper import get_masked_as_type

        used_type = get_masked_as_type(cls, DataLoader)
        obj = used_type(*args, **kwargs)
        return obj
