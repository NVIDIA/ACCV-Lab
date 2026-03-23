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

import torch
from torch.autograd.function import once_differentiable
from typing import Any, Union, Sequence, Optional
import accvlab.batching_helpers.batched_indexing_access_cuda as batched_indexing_access_cuda
from .data_format import RaggedBatch


class BatchedIndexingAccess(torch.autograd.Function):
    """Batched indexing with non-uniform indices.

    A wrapper function (:func:`batched_indexing_access`) is available and this class should not be used directly.

    For details about the indexing operation, see documentation of :func:`batched_indexing_access` below, which wraps the functionality of this class
    and presents it as a function.

    """

    @staticmethod
    def forward(
        ctx: Any,
        input_data: torch.Tensor,
        input_indices: torch.Tensor,
        input_nums_indices: torch.Tensor,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """Batched indexing with non-uniform indices.

        Detailed documentation see :func:`batched_indexing_access` below, which wraps the functionality of this class
        and presents it as a function (with some additional functionality).

        """
        input_data = input_data.contiguous()
        input_indices = input_indices.contiguous()
        input_nums_indices = input_nums_indices.contiguous()
        result = batched_indexing_access_cuda.forward(
            input_data, input_indices, input_nums_indices, fill_value
        )
        ctx.save_for_backward(input_indices, input_nums_indices)
        num_batch_dim = input_nums_indices.dim()
        ctx.input_num_targets = input_data.shape[num_batch_dim]
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: Union[torch.Tensor, None]):
        """Perform back-propagation of gradients for the performed mapping operation (see documentation of method :meth:`forward`)."""
        if grad is None:
            return None, None, None, None
        else:
            input_indices, input_nums_indices = ctx.saved_tensors
            grad = grad.contiguous()
            grad_input = batched_indexing_access_cuda.backward_new_tensor(
                grad, input_indices, input_nums_indices, ctx.input_num_targets, 0.0, backward_accumulate=True
            )
            return grad_input, None, None, None


class BatchedInverseIndexingAccessNewTensor(torch.autograd.Function):
    """Batched inverse indexing access, i.e. writing data into the indexed location, with non-uniform indices

    For details about the inverse indexing operation, see documentation of :func:`batched_inverse_indexing_access` below,
    which wraps the functionality of this class and presents it as a function (with some additional functionality).
    """

    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        output_indices: torch.Tensor,
        output_nums_indices: torch.Tensor,
        output_num_targets: Sequence,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """Batched indexing, i.e. writing data into the indexed location, with non-uniform indices

        Detailed documentation see :func:`batched_inverse_indexing_access` below, which wraps the functionality of this class
        and presents it as a function.
        """
        input = input.contiguous()
        output_indices = output_indices.contiguous()
        output_nums_indices = output_nums_indices.contiguous()
        result = batched_indexing_access_cuda.backward_new_tensor(
            input,
            output_indices,
            output_nums_indices,
            output_num_targets,
            fill_value,
            backward_accumulate=False,
        )
        ctx.save_for_backward(output_indices, output_nums_indices)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: Union[torch.Tensor, None]):
        """Perform back-propagation of gradients for the performed mapping operation (see documentation of method :meth:`forward`)."""
        if grad is None:
            return None, None, None, None, None
        else:
            output_indices, output_nums_indices = ctx.saved_tensors
            grad = grad.contiguous()
            grad_input = batched_indexing_access_cuda.forward(grad, output_indices, output_nums_indices, 0.0)
            return grad_input, None, None, None, None


class BatchedInverseIndexingAccessInsert(torch.autograd.Function):
    """Batched inverse indexing access, i.e. writing data into the indexed location, with non-uniform indices

    For details about the inverse indexing operation, see documentation of :func:`batched_inverse_indexing_access` below,
    which wraps the functionality of this class and presents it as a function (with some additional functionality).
    """

    @staticmethod
    def forward(
        ctx: Any,
        to_fill: torch.Tensor,
        output_indices: torch.Tensor,
        output_nums_indices: torch.Tensor,
        to_fill_into: torch.Tensor,
    ) -> torch.Tensor:
        """Batched indexing, i.e. writing data into the indexed location, with non-uniform indices

        Detailed documentation see :func:`batched_inverse_indexing_access` below, which wraps the functionality of this class
        and presents it as a function.
        """
        to_fill = to_fill.contiguous()
        to_fill_into = to_fill_into.contiguous()
        output_indices = output_indices.contiguous()
        output_nums_indices = output_nums_indices.contiguous()
        result = batched_indexing_access_cuda.backward_insert(
            to_fill, output_indices, output_nums_indices, to_fill_into
        )
        ctx.save_for_backward(output_indices, output_nums_indices)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: Union[torch.Tensor, None]):
        """Perform back-propagation of gradients for the performed mapping operation (see documentation of method :meth:`forward`)."""
        if grad is None:
            return None, None, None, None
        else:
            output_indices, output_nums_indices = ctx.saved_tensors
            grad = grad.contiguous()
            grad_for_to_insert = batched_indexing_access_cuda.forward(
                grad, output_indices, output_nums_indices, 0.0
            )
            grad_for_to_insert_into = batched_indexing_access_cuda.backward_insert_const(
                0.0, output_indices, output_nums_indices, grad
            )
            return grad_for_to_insert, None, None, grad_for_to_insert_into


def batched_indexing_access(
    input_data: Union[RaggedBatch, torch.Tensor],
    input_indices: RaggedBatch,
    filler_value: float = 0.0,
    dim_to_index_in: Optional[int] = None,
) -> RaggedBatch:
    """Batched indexing access with non-uniform indices.

    :gpu:

    Note that for each sample, the number of resulting entries corresponds to the number of indices. This means that in general,
    the output size will be non-uniform. Therefore, a :class:`RaggedBatch` is returned regardless of the ``input_data`` type.

    Note:
        Note that whether ``input_data`` is a :class:`RaggedBatch` or a :class:`torch.Tensor`, the indexing of ``input_data``
        is performed along ``dim_to_index_in``, which is not necessarily the non-uniform dimension of ``input_data``.

    Warning:
        While the ``filler_value`` parameter can be used to set the value for filler values,
        the filler value may change when processing the resulting :class:`RaggedBatch` further. Therefore,
        care needs to be taken when assuming a certain filler value.

    Args:
        input_data: Data to which the indexing is applied.
        input_indices: For each sample (element along the batch dimension), the indices of entries to obtain from the input.
            Shape: ``(*batch_shape, max_num_indices)``
            Here, ``max_num_indices`` corresponds to the maximum number of indices over all samples.
        filler_value: Filler values for the remaining elements in the output (corresponding to the fillers in ``input_indices``).
            Default: 0.0
        dim_to_index_in: Dimension on which to apply the indexing. Cannot be a batch dimension of the input indices.
            If not set, will corresponds to `input_indices.non_uniform_dim`.

    Returns:
        Result containing the indexed entries from the input tensor. For a sample ``i`` and a valid
        index ``j < input_indices.sample_sizes[i]``, the following holds (assuming ``dim_to_index_in == 1``):
        ``indexed_vals[i, j] == input_data[i, input_indices[i, j]]``

        The shape of the resulting data is:

            - ``indexed_vals.shape[0] == batch_size``
            - ``indexed_vals.shape[dim_to_index_in] == max_num_indices``
            - Remaining dimensions correspond to the input data

    Example:
        In the illustration below:
          - Letters indicate data entries that are indexed in the input (and therefore appear in the output)
          - '-' indicates entries where the actual values are not relevant (in the input).
          - '*' indicates filler values in :class:`RaggedBatch` instances.

        .. image:: images/BatchedIndexing_ragged.png
            :alt: Illustration of the batched indexing operation
            :align: center

        Each depicted entry in the data may represent a single value (in case of
        2D tensors), or itself be a non-scalar entry (in case that ``input_data`` has more than 2 dimensions).

        Note that for ``input_indices``, the entries are always scalar.

        Also, we do not show the ``filler_value`` in the example. It is filled into the '*'-entries in the output.

        In this case, the ``dim_to_index_in`` is 1.

    """
    is_input_ragged_batch = isinstance(input_data, RaggedBatch)

    if is_input_ragged_batch:
        input_data = input_data.tensor

    if dim_to_index_in is None:
        dim_to_index_in = input_indices.non_uniform_dim
        transpose_needed = False
    else:
        assert (
            dim_to_index_in >= input_indices.num_batch_dims
        ), "Cannot index in a batch dimension of the input indices"
        transpose_needed = input_indices.num_batch_dims != dim_to_index_in

    assert (
        dim_to_index_in >= input_indices.num_batch_dims
    ), "Cannot index in a batch dimension of the input indices"

    transpose_needed = input_indices.num_batch_dims != dim_to_index_in
    if transpose_needed:
        input_data = input_data.transpose(input_indices.num_batch_dims, dim_to_index_in)

    res = BatchedIndexingAccess.apply(
        input_data, input_indices.tensor, input_indices.sample_sizes, filler_value
    )

    if transpose_needed:
        res = res.transpose(input_indices.num_batch_dims, dim_to_index_in)

    res = input_indices.create_with_sample_sizes_like_self(res, dim_to_index_in)

    return res


def batched_inverse_indexing_access(
    input_data: Union[RaggedBatch, torch.Tensor],
    output_indices: RaggedBatch,
    output_num_targets: int,
    filler_value: float = 0.0,
    dim_to_index_in: Optional[int] = None,
) -> torch.Tensor:
    """Batched setting of values at given indices, with non-uniform indices.

    :gpu:

    Non-uniform indices means that for each sample, the indices, as well as the number of indices, vary.

    Note:
        This function is similar to :func:`batched_indexing_write`, but instead of using a ``to_write_into``
        tensor, a tensor with a uniform filler value is created first, and the values to set are
        written into that tensor.

    Note:
        Note that whether ``input_data`` is a :class:`RaggedBatch` instance or a tensor, the indexing
        is performed along ``dim_to_index_in``, which is not necessarily the non-uniform dimension
        of ``input_data``.

    Warning:
        This function assumes that for each sample, there are no duplicate indices in
        ``output_indices``, i.e. there are no duplicates in the valid entries in:
        ``output_indices[i, 0:output_indices.sample_sizes[i]]``.

        If this is not the case, the behavior is undefined.

    Args:
        input_data: Data which to write into the given indices.
        output_indices: For each sample (element along the batch dimension), the indices of entries to write to in the output.
            Shape: ``(batch_size, max_num_indices)``
            Here, ``max_num_indices`` corresponds to the maximum number of indices over all samples.
        output_num_targets: Size of the dimension corresponding to the indexed dimension in the output
        filler_value: Filler values for the non-indexed elements in the output. Default: 0.0
        dim_to_index_in: Dimension on which to apply the indexing. Optional, default is the non-uniform
            dimension of the output indices.

    Returns:
        Resulting tensor, containing the filled in values from the input, inserted at the corresponding indices,
        and the filler values everywhere else.

        For each sample ``i`` and each valid index ``j < output_indices.sample_sizes[i]``, the following holds:

            ``output[i, output_indices[i, j]] == input_data[i, j]``

        The shape of the resulting data is:

            - ``output.shape[0] == batch_size``
            - ``output.shape[dim_to_index_in] == output_nums_targets``
            - Remaining dimensions correspond to the input data

    Example:
        In the illustration below:
          - Letters indicate data entries that are indexed in the input (and therefore appear in the output)
          - '-' indicates entries where the actual values are not relevant (in the input).
          - '*' indicates filler values in :class:`RaggedBatch` instances.

        .. image:: images/BatchedInverseIndexing_ragged.png
            :alt: Illustration of the batched inverse indexing operation
            :align: center

        Each depicted entry in the data may represent a single value (in case of
        2D tensors), or itself be a non-scalar entry (in case that the data has more than 2 dimensions).

        Note that for ``output_indices``, the entries are always scalar.

        In this case, the ``dim_to_index_in`` is 1.
    """
    is_input_ragged_batch = isinstance(input_data, RaggedBatch)

    if is_input_ragged_batch:
        input_data = input_data.tensor

    if dim_to_index_in is None:
        dim_to_index_in = output_indices.non_uniform_dim
        transpose_needed = False
    else:
        assert (
            dim_to_index_in >= output_indices.num_batch_dims
        ), "Cannot index in a batch dimension of the output indices"
        transpose_needed = output_indices.num_batch_dims != dim_to_index_in

    if transpose_needed:
        input_data = input_data.transpose(output_indices.num_batch_dims, dim_to_index_in)

    res = BatchedInverseIndexingAccessNewTensor.apply(
        input_data, output_indices.tensor, output_indices.sample_sizes, output_num_targets, filler_value
    )

    if transpose_needed:
        res = res.transpose(output_indices.num_batch_dims, dim_to_index_in)

    return res


def batched_indexing_write(
    to_write: Union[RaggedBatch, torch.Tensor],
    output_indices: RaggedBatch,
    to_write_into: Union[RaggedBatch, torch.Tensor],
    dim_to_index_in: Optional[int] = None,
) -> Union[RaggedBatch, torch.Tensor]:
    """Batched indexing write, i.e. writing data into the indexed location, with non-uniform indices.

    Non-uniform indices means that for each sample, the indices, as well as the number of indices, vary.

    :gpu:

    Note:
        This function is similar to :func:`batched_inverse_indexing_access`, but instead of creating a
        constant tensor and filling the values in there, a ``to_write_into`` tensor is used, which
        may already contain values, and only the values corresponding to the indices are updated.

    Note:
        Note that whether ``to_write`` and ``to_write_into`` are :class:`RaggedBatch` or :class:`torch.Tensor` instances, the indexing
        is performed along ``dim_to_index_in``, which is not necessarily the non-uniform dimension
        of ``to_write`` or ``to_write_into``.

    Warning:
        This function assumes that for each sample, there are no duplicate indices in
        ``output_indices``, i.e. there are no duplicates in the valid entries in:
        ``output_indices[i, 0:output_indices.sample_sizes[i]]``.

        If this is not the case, the behavior is undefined.

    Args:
        to_write: Data which to write into the given indices.
        output_indices: For each sample (element along the batch dimension), the indices of entries to write to in the output.
            Shape: ``(batch_size, max_num_indices)``
            Here, ``max_num_indices`` corresponds to the maximum number of indices over all samples.
        to_write_into: Tensor or RaggedBatch to write into.
        dim_to_index_in: Dimension on which to apply the indexing. Optional, default is the non-uniform
            dimension of the output indices.

    Returns:
        Resulting tensor or :class:`RaggedBatch` instance. Corresponds to ``to_write_into``, with the values from
        ``to_write`` inserted at the corresponding indices, and the original values from ``to_write_into``
        everywhere else.

    Example:
        In the illustration below:
          - Letters indicate data entries that are indexed in the input (and therefore appear in the output)
          - '-' indicates entries where the actual values are not relevant (in the input).
          - '*' indicates filler values in :class:`RaggedBatch` instances.
          - '..' indicates data which remains unchanged, i.e. is the same as in the ``to_write_into`` parameter
            and the output.

        .. image:: images/BatchedIndexWrite_ragged.png
            :alt: Illustration of the batched indexing write operation
            :align: center

        Each depicted entry in the data may represent a single value (in case of
        2D tensors), or itself be a non-scalar entry (in case that the data has more than 2 dimensions).

        Note that for ``output_indices``, the entries are always scalar.

        In this case, the ``dim_to_index_in`` is 1.
    """
    is_input_ragged_batch = isinstance(to_write, RaggedBatch)
    is_output_ragged_batch = isinstance(to_write_into, RaggedBatch)

    if dim_to_index_in is None:
        dim_to_index_in = output_indices.non_uniform_dim
        transpose_needed = False
    else:
        assert (
            dim_to_index_in >= output_indices.num_batch_dims
        ), "Cannot index in a batch dimension of the output indices"
        transpose_needed = output_indices.num_batch_dims != dim_to_index_in

    if is_input_ragged_batch:
        to_write = to_write.tensor
    if is_output_ragged_batch:
        to_write_into_data = to_write_into.tensor
    else:
        to_write_into_data = to_write_into
    if transpose_needed:
        assert dim_to_index_in >= output_indices.num_batch_dims, "Cannot index in any batch dimension"
        to_write = to_write.transpose(output_indices.num_batch_dims, dim_to_index_in)
        to_write_into_data = to_write_into_data.transpose(output_indices.num_batch_dims, dim_to_index_in)

    res = BatchedInverseIndexingAccessInsert.apply(
        to_write, output_indices.tensor, output_indices.sample_sizes, to_write_into_data
    )
    if transpose_needed:
        res = res.transpose(output_indices.num_batch_dims, dim_to_index_in)
    if is_output_ragged_batch:
        res = to_write_into.create_with_sample_sizes_like_self(res)
    return res
