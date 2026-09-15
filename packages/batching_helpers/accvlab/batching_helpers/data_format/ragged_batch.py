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
from typing import Union, List, Tuple, Sequence, Optional, Callable

import torch

from .set_padded_to import SetPaddedTo

# Type aliases for processing function signatures and returns
ReturnTensor = Union[torch.Tensor, Tuple[torch.Tensor, ...]]
ProcStep = Union[
    Callable[[torch.Tensor], ReturnTensor],
    Callable[[torch.Tensor, torch.Tensor], ReturnTensor],
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], ReturnTensor],
]


class RaggedBatch:
    """Class for representing batches with samples with variable size in one dimension.

    The representation of the batch contains 3 tensors:
        - tensor:
            This is the actual data. It has the size of the largest sample
            in the non-uniform dimension, and the other samples are padded on the "right" (i.e.
            at the end containing larger indices) with filler values to match the size of the largest sample.
            While the padding is typically initialized with 0, no values should be assumed for the
            padded region as the values there may change after operations are performed on the
            data. If the non-uniform dimension is `dim==num_batch_dims`, the shape is
            (\*batch_dims_shape, max_sample_size, \*data_shape). More generally, the first dimensions are the
            batch dimensions (one or more). The non-uniform dimension can be any dimension after the batch dimensions
            and the size of the non-uniform dimension always corresponds to the maximum sample
            size in the batch. The remaining dimensions correspond to the shape of the data, which can have
            any number of dimensions, including 0 (per-object scalar data).
        - mask:
            This is the mask indicating which elements are valid (`True`) and which are
            not (`False`). It has dimensions: (\*batch_dims_shape, max_sample_size).
            The dimension after the batch dimensions corresponds to the non-uniform dimension in the data
            tensor.
        - sample_sizes:
            Sizes of the individual samples, i.e. the actual sizes without padding along the
            non-uniform dimension.
            Shape: (\*batch_dims_shape,)

    Additional attributes describing the batch:
        - non_uniform_dim:
            Indicates which dimension is the non-uniform dimension
        - num_batch_dims:
            Number of batch dimensions at the beginning of the tensor

    Note:
        The tensors described above correspond to the :attr:`tensor`, :attr:`mask`, and :attr:`sample_sizes`
        attributes, respectively. The non-uniform dimension can be accessed as :attr:`non_uniform_dim` and the
        number of batch dimensions as :attr:`num_batch_dims`.

    Important:
        The :attr:`mask` and :attr:`non_uniform_dim` attributes may be shared between instances of
        :class:`RaggedBatch` instances with different data tensors, so they should be treated as
        constants and never be changed in-place.

    Example:

        Here, we show an example of a :class:`RaggedBatch` instance.

        In the image::
          - Letters indicate data entries that are valid (i.e. correspond to the actual data).
          - '*' indicates padded filler entries (i.e. invalid entries) in the data.

        .. image:: images/RaggedBatchExample.png
            :alt: Example of a RaggedBatch
            :align: center

        Note that:
          - The example shows a single batch dimension of size 4. More batch and data dimensions are supported.
          - The maximum sample size (i.e. the size of the non-uniform dimension) is 3.
          - Each element in :attr:`self.tensor` may represent a single value (corresponding to scalar data and 0 data dimensions),
            or itself represent a non-scalar entry (in case for one or more data dimensions).
          - Even if more data dimensions are present, the `mask` has always `num_batch_dims + 1` dimensions, as the data dimensions
            are not needed in the mask.
          - The `sample_sizes` have the same shape as the batch dimensions (i.e. `(4,)` in this example), as they contain one value per sample.
          - The `sample_sizes` and `mask` contain the same information. However

            - Dependent on the use case, one of them may be more efficient & convenient to use
            - One can be efficiently computed from the other (as is done as needed in the `RaggedBatch` implementation).

    Note:
        The number of batch dimensions is determined from the shape of the provided `mask` or `sample_sizes` tensor.

    Warning:
        If both `mask` and `sample_sizes` are set, they need to be consistent with each other. This is
        not checked in the constructor. Inconsistent masks and sample sizes will lead to undefined behavior.

    """

    _CPU = torch.device("cpu")

    def __init__(
        self,
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_sizes: Optional[torch.Tensor] = None,
        non_uniform_dim: Optional[int] = None,
    ):
        '''
        Args:
            tensor: Data to be stored (corresponding to the :attr:`tensor` tensor of :class:`RaggedBatch`, see description above)
            mask: Mask indicating which entries are valid (corresponding to the :attr:`mask` tensor of :class:`RaggedBatch`, see description above).
                If not set, `sample_sizes` is internally used to create a mask.
                Note that at least one of `mask` or `sample_sizes` needs to be set.
            sample_sizes: Number of valid entries for all samples (corresponding to the :attr:`sample_sizes` tensor of :class:`RaggedBatch`, see description above).
                If not set, `mask` is internally used to create a sample sizes tensor.
                Note that at least one of `mask` or `sample_sizes` needs to be set.
            non_uniform_dim: Dimension in which the batch is non-uniform, default: 1
        '''

        assert (
            mask is not None or sample_sizes is not None
        ), "At least one of `mask` or `sample_sizes` needs to be set"

        if sample_sizes is not None:
            num_batch_dims = sample_sizes.dim()
        else:
            num_batch_dims = mask.dim() - 1

        assert num_batch_dims > 0, "Number of batch dimensions needs to be greater than 0"
        assert (
            num_batch_dims < tensor.dim()
        ), "The number of dimensions of the tensor needs to be at least num_batch_dims + 1"

        if non_uniform_dim is None:
            non_uniform_dim = num_batch_dims

        assert (
            non_uniform_dim >= num_batch_dims and non_uniform_dim < tensor.dim()
        ), "Non-uniform dimensions needs to be in the range [num_batch_dims; tensor.dim()["

        assert mask is None or (
            mask.shape[:num_batch_dims] == tensor.shape[:num_batch_dims]
            and mask.shape[num_batch_dims] == tensor.shape[non_uniform_dim]
        ), (
            "Shape of `tensor` does not match the required shape:\n"
            f"  According to mask: Batch shape: {mask.shape[:num_batch_dims]}; Maximum sample size: {mask.shape[num_batch_dims]}\n"
            f"  According to tensor: Batch shape: {tensor.shape[:num_batch_dims]}; Maximum sample size: {tensor.shape[non_uniform_dim]}"
        )
        assert sample_sizes is None or (
            sample_sizes.shape[:num_batch_dims] == tensor.shape[:num_batch_dims]
        ), (
            "Batch shape according to `tensor` does not match the size of `sample_sizes`:\n"
            f"  According to tensor: Batch shape: {tensor.shape[:num_batch_dims]}\n"
            f"  According to sample_sizes: Batch shape: {sample_sizes.shape[:num_batch_dims]}"
        )

        self._tensor = tensor
        self._mask = mask
        self._sample_sizes = sample_sizes
        self._non_uniform_dim = non_uniform_dim
        self._num_batch_dims = num_batch_dims
        self._batch_shape = tensor.shape[:num_batch_dims]
        self._total_num_targets = None

    @classmethod
    def FromOversizeTensor(
        cls,
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_sizes: Optional[torch.Tensor] = None,
        non_uniform_dim: Optional[int] = None,
    ) -> RaggedBatch:
        """Create a :class:`RaggedBatch` instance from a tensor which is over-sized in the non-uniform dimension.

        Over-sized means that the non-uniform dimension is larger than the maximum sample size in the batch.

        Args:
            tensor: Data to be stored (corresponding to the :attr:`tensor` tensor of :class:`RaggedBatch`, see description above) except that the
                non-uniform dimension is larger than the maximum sample size in the batch.
                The tensor in the is truncated to the maximum sample size in the batch.
            mask: Mask indicating which entries are valid (corresponding to the :attr:`mask` tensor of :class:`RaggedBatch`, see description above).
                If not set, `sample_sizes` is internally used to create a mask.
                Note that at least one of `mask` or `sample_sizes` needs to be set. The mask is truncated to the maximum sample size in the batch.
            sample_sizes: Number of valid entries for all samples (corresponding to the :attr:`sample_sizes` tensor of :class:`RaggedBatch`, see description above).
                If not set, `mask` is internally used to create a sample sizes tensor.
                Note that at least one of `mask` or `sample_sizes` needs to be set.
            non_uniform_dim: Dimension in which the batch is non-uniform, default: 1

        Note:
            The number of batch dimensions is determined from the shape of the provided `mask` or `sample_sizes` tensor.

        Warning:
            If both `mask` and `sample_sizes` are set, they need to be consistent with each other. This is
            not checked in the constructor. Inconsistent masks and sample sizes will lead to undefined behavior.
        """
        if non_uniform_dim is None:
            if sample_sizes is not None:
                non_uniform_dim = sample_sizes.dim()
            elif mask is not None:
                non_uniform_dim = mask.dim() - 1
            else:
                raise ValueError("Either `sample_sizes` or `mask` needs to be set")

        if sample_sizes is None:
            sample_sizes = torch.sum(mask, dim=non_uniform_dim, dtype=torch.int64)
        max_sample_size = int(torch.max(sample_sizes).item())
        tensor = torch.narrow(tensor, non_uniform_dim, 0, max_sample_size)
        mask = torch.narrow(mask, non_uniform_dim, 0, max_sample_size) if mask is not None else None
        res = cls(tensor, mask, sample_sizes, non_uniform_dim)
        return res

    def _init_mask(self):
        tensor = self._tensor
        mask = torch.ones(
            *tensor.shape[: self._num_batch_dims],
            tensor.shape[self._non_uniform_dim],
            dtype=torch.bool,
            device=tensor.device,
        )
        self._mask = SetPaddedTo.apply(mask, self._sample_sizes, False)

    def _init_sample_sizes(self):
        self._sample_sizes = torch.sum(self._mask, dim=self._non_uniform_dim, dtype=torch.int64)

    @classmethod
    def Empty(
        cls,
        num_dims: int,
        non_uniform_dim: int,
        device: Union[torch.device, str],
        num_batch_dims: Optional[int] = None,
        batch_shape: Optional[Union[Sequence[int], int]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> RaggedBatch:
        """Create an empty instance.

        The so created instance has a size of 0 along all dimensions.

        Note:
            If neither `num_batch_dims` nor `batch_shape` is provided, the number of batch dimensions is 1 and the batch shape is (0,).

        Args:
            num_dims: Total number of dimensions
            non_uniform_dim: The non-uniform dimension
            device: Device to use for the instance
            num_batch_dims: Number of batch dimensions. If provided, `batch_shape` cannot be set and size 0 is
                assumed for all batch dimensions.
            batch_shape: Shape of the batch (can be a sequence of ints or a single int in case of a single batch dimension).
                If not provided, the batch shape is (0,) * num_batch_dims. If provided, `num_batch_dims` cannot be set
                and the number of batch dimensions is inferred from the shape.
            dtype: Data type of the tensor. Default: ``torch.float32``

        Returns:
            The resulting empty :class:`RaggedBatch` instance
        """

        assert (
            num_batch_dims is None or batch_shape is None
        ), "Either num_batch_dims or batch_shape can be provided, but not both"

        if num_batch_dims is None and batch_shape is None:
            num_batch_dims = 1
            batch_shape = (0,)
        elif batch_shape is None:
            assert num_batch_dims > 0, "Number of batch dimensions needs to be greater than 0"
            batch_shape = (0,) * num_batch_dims
        else:
            if isinstance(batch_shape, int):
                batch_shape = (batch_shape,)
            else:
                batch_shape = tuple(batch_shape)
            assert len(batch_shape) > 0, "Batch shape needs to be a non-empty sequence"
            num_batch_dims = len(batch_shape)

        assert (
            len(batch_shape) < num_dims
        ), "Number of batch dimensions needs to be less than the total number of dimensions"

        assert (
            non_uniform_dim >= num_batch_dims and non_uniform_dim < num_dims
        ), "Non-uniform dimension needs to be in the range [num_batch_dims; num_dims["

        sample_sizes_shape = batch_shape
        mask_shape = batch_shape + (0,)
        tensor_shape = batch_shape + (0,) * (num_dims - len(batch_shape))
        tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
        mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
        sizes = torch.zeros(sample_sizes_shape, dtype=torch.int64, device=device)
        res = RaggedBatch(tensor, mask, sizes, non_uniform_dim)
        return res

    @classmethod
    def FromFullTensor(
        cls, full_tensor: torch.Tensor, non_uniform_dim: int = 1, num_batch_dims: int = 1
    ) -> RaggedBatch:
        """Create a :class:`RaggedBatch` instance from a tensor representing a uniform-sized batch.

        Args:
            full_tensor: Tensor to convert into a :class:`RaggedBatch` instance
            non_uniform_dim: Dimension to use as the non-uniform dimension.
                Note that while in this special case, all dimensions are uniform,
                the non-uniform dimension has a special meaning
                (e.g. for :func:`get_non_uniform_dimension_transposed_to`, and many other functions) and
                needs to be set.
            num_batch_dims: Number of batch dimensions in the tensor. Default: 1

        Returns:
            The resulting :class:`RaggedBatch` instance containing the input tensor

        """
        batch_shape = full_tensor.shape[:num_batch_dims]
        sample_size = full_tensor.shape[non_uniform_dim]

        assert num_batch_dims > 0, "Number of batch dimensions needs to be greater than 0"
        assert (
            non_uniform_dim >= num_batch_dims and non_uniform_dim < full_tensor.dim()
        ), f"Non-uniform dimension needs to be in the range [{num_batch_dims}; full_tensor.dim()["

        mask_shape = (*batch_shape, sample_size)
        mask = torch.ones(mask_shape, dtype=torch.bool, device=full_tensor.device)
        sample_sizes = torch.full(batch_shape, sample_size, dtype=torch.int64, device=full_tensor.device)

        res = RaggedBatch(full_tensor, mask, sample_sizes, non_uniform_dim)
        return res

    @property
    def tensor(self) -> torch.Tensor:
        """Get the data tensor

        See the description of :class:`RaggedBatch` for more information on `tensor`.

        For setting the data tensor, use :func:`set_tensor`.

        """
        return self._tensor

    @property
    def mask(self) -> torch.Tensor:
        """Get the mask tensor

        See the description of :class:`RaggedBatch` for more information on `mask`.

        The mask indicates which elements are valid (`True`) and which are not (`False`).
        It has dimensions: ``(*batch_dims_shape, max_sample_size)``.

        """
        if self._mask is None:
            self._init_mask()
        return self._mask

    @property
    def sample_sizes(self) -> torch.Tensor:
        """Get the sample sizes tensor

        See the description of :class:`RaggedBatch` for more information on `sample_sizes`.

        The sample sizes tensor contains the actual sizes of each sample in the batch
        along the non-uniform dimension.
        Its dimensions are ``batch_dims_shape``.

        """
        if self._sample_sizes is None:
            self._init_sample_sizes()
        return self._sample_sizes

    @property
    def non_uniform_dim(self) -> int:
        """Get the non-uniform dimension"""
        return self._non_uniform_dim

    @property
    def num_batch_dims(self) -> int:
        """Get the number of batch dimensions"""
        return self._num_batch_dims

    @property
    def batch_shape(self) -> torch.Size:
        """Get the batch shape"""
        return self._batch_shape

    @property
    def total_num_samples_in_batch(self) -> int:
        """Get the total number of samples in the batch"""
        return torch.prod(torch.tensor(self._batch_shape)).item()

    @property
    def total_num_entries(self) -> int:
        """Get the total number of entries.

        This is the accumulated number of valid entries along the non-uniform dimension over all samples in the batch.
        This information is computed from the :attr:`sample_sizes` tensor when it is first accessed and re-used on subsequent calls.
        """
        if self._total_num_targets is None:
            self._total_num_targets = torch.sum(self.sample_sizes).item()
        return self._total_num_targets

    @property
    def max_sample_size(self) -> int:
        """Get the maximum sample size in the batch"""
        return self._tensor.shape[self._non_uniform_dim]

    def as_self_with_cloned_data(self) -> RaggedBatch:
        """Create a copy, where the data tensor (i.e. :attr:`tensor`) is cloned (while mask and sample sizes are shared)"""
        res = RaggedBatch(self._tensor.clone(), self.mask, self.sample_sizes, self._non_uniform_dim)
        return res

    def create_with_sample_sizes_like_self(
        self,
        tensor: torch.Tensor,
        non_uniform_dim: Optional[int] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> RaggedBatch:
        """Create a :class:`RaggedBatch` instance with the same batch shape and sample sizes as `this`

        Note that while the sample sizes are the same, the total number of dimensions,
        the non-uniform dimension, and the size of the `data` tensor except in the
        batch and the non-uniform dimensions may be different.

        Args:
            tensor: Data to set for the new instance (padded tensor)
            non_uniform_dim: Non-uniform dimension (in `tensor`). Can be set to `None` to use the same
                dimension as `this`. Default: `None`
            device: Device on which to create the resulting :class:`RaggedBatch` instance. If not provided,
                the device of the input `tensor` is used.

        Returns:
            Resulting :class:`RaggedBatch` instance with the same batch shape and sample sizes as `this`.
        """
        # Note that for checking the upper bound for the non-uniform dimension and handling negative values,
        # we use the tensor for the number of dimensions, as this will be the data tensor of the new RaggedBatch
        # to create.
        # For the lower bound, we use the number of batch dimensions of `self`, as the batch dimensions of the new
        # RaggedBatch to create will be the same as the batch dimensions of `self`.
        if non_uniform_dim is None:
            non_uniform_dim = self._non_uniform_dim
        elif non_uniform_dim < 0:
            non_uniform_dim = tensor.dim() + non_uniform_dim

        assert (
            non_uniform_dim >= self._num_batch_dims and non_uniform_dim < tensor.dim()
        ), f"Non-uniform dimension needs to be in the range [{self._num_batch_dims}; {tensor.dim()}["

        # Check that all batch dimensions match
        assert tensor.shape[: self._num_batch_dims] == self.batch_shape, (
            f"Batch shape of tensor does not match required batch shape:\n"
            f"  Expected batch shape: {self.batch_shape}\n"
            f"  Got batch shape: {tensor.shape[:self._num_batch_dims]}"
        )

        assert tensor.shape[non_uniform_dim] == self.shape[self._non_uniform_dim], (
            f"Non-uniform dimension size of tensor does not match required non-uniform dimension size:\n"
            f"  Expected non-uniform dimension size: {self.shape[self._non_uniform_dim]}\n"
            f"  Got non-uniform dimension size: {tensor.shape[non_uniform_dim]}"
        )

        if device is None:
            device = tensor.device
        else:
            tensor = tensor.to(device=device)
        mask = self.mask.to(device=device)
        sample_sizes = self.sample_sizes.to(device=device)
        res = RaggedBatch(tensor, mask, sample_sizes, non_uniform_dim)
        # In case this number is already computed, it can be re-used in the new RaggedBatch
        res._total_num_targets = self._total_num_targets
        return res

    def get_non_uniform_dimension_transposed_to(self, dim: int) -> RaggedBatch:
        """Get with the non-uniform dimension transposed to a given dimension.

        If the given dimension is already the non-uniform dimension, `self` is returned.

        Info:
            The non-uniform dimension cannot be set to a batch dimension (i.e., any dimension < num_batch_dims).

        Args:
            dim: Dimension to transpose the current non-uniform dimension to

        Returns:
            Resulting :class:`RaggedBatch` instance
        """
        assert (
            dim >= self._num_batch_dims and dim < self._tensor.dim()
        ), f"Non-uniform dimensions needs to be in the range [{self._num_batch_dims}; tensor.dim()["
        if dim == self._non_uniform_dim:
            res = self
        else:
            tensor_transposed = self._tensor.transpose(self._non_uniform_dim, dim)
            res = self.create_with_sample_sizes_like_self(tensor_transposed, dim)
        return res

    def get_existence_weights(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get the existence weights

        The existence weights are 1.0 for the contained entries (i.e. entries corresponding to actual
        data as opposed to padded fillers)  and 0.0 for filler entries.

        In contrast to `self.mask`, the dimensionality and shape of the weights
        correspond to the dimensionality and shape of the data. This means that the mask
        can be directly applied to the data tensor (i.e. :attr:`tensor`), regardless of the number of dimensions or
        which dimension is the non-uniform dimension.

        Args:
            dtype: Type for the existence weights. Default: torch.float32

        Returns:
            The resulting weights tensor
        """
        mask = self.mask
        num_dims_extra = self._tensor.dim() - mask.dim()
        shape_weights_reshape = (*mask.shape, *((1,) * num_dims_extra))
        weights = mask.to(dtype=dtype).reshape(shape_weights_reshape)
        if self._non_uniform_dim != self._num_batch_dims:
            weights = weights.transpose(self._num_batch_dims, self._non_uniform_dim)
        weights_num_repeats = list(self._tensor.shape)
        # Don't repeat along batch dimensions
        for i in range(self._num_batch_dims):
            weights_num_repeats[i] = 1
        weights_num_repeats[self._non_uniform_dim] = 1
        weights = weights.repeat(weights_num_repeats)
        return weights

    def with_padded_set_to(self, value_to_set: float) -> RaggedBatch:
        """Set filler/padded entries in the data (i.e. :attr:`tensor`) to a fixed value.

        Note:
            This operation is not performed in-place, i.e. `this.tensor` is not changed. For an in-place
            operation, use :func:`set_padded_to` instead.

        Args:
            value_to_set: Value to set for padded entries.

        Returns:
            Like self, but with the padded values set
        """
        res = self.as_self_with_cloned_data()
        res.set_padded_to(value_to_set)
        return res

    def set_padded_to(self, value_to_set: float) -> None:
        """Set filler/padded entries in the data tensor (i.e. :attr:`tensor`) to a fixed value in-place.

        Note:
            Note that as this operation is in-place. This means that `this.tensor` is changed.
            No new :class:`RaggedBatch` instance is created. If this is not desired, use
            :func:`with_padded_set_to` instead, which is not in-place and returns a new :class:`RaggedBatch` instance.

        Args:
            value_to_set: Value to set for padded entries.
        """
        tensor = self._tensor
        if self._non_uniform_dim != self._num_batch_dims:
            tensor = tensor.transpose(self._num_batch_dims, self._non_uniform_dim)
        tensor = SetPaddedTo.apply(tensor, self.sample_sizes, value_to_set)
        if self._non_uniform_dim != self._num_batch_dims:
            tensor = tensor.transpose(self._num_batch_dims, self._non_uniform_dim)
        self._tensor = tensor

    def repeat_samples(
        self,
        num_repeats: Union[int, Sequence[int]],
        batch_dim: Optional[int] = None,
    ) -> RaggedBatch:
        """Repeat along a single batch dimension

        Args:
            num_repeats: Number of times to repeat. In case of a single value, the dimension in which to repeat
                is specified by `batch_dim`. In case of a sequence, the sequence needs to have the same length as
                the number of batch dimensions and `batch dim` must not be set.
            batch_dim: Which batch dimension to repeat along. Can only be set if `num_repeats` is a single value.
                If not set (and `num_repeats` is a single value), `0` is used.

        Returns:
            Resulting :class:`RaggedBatch` instance with the samples repeated

        """
        if isinstance(num_repeats, int):
            use_num_repeats = True
            assert (
                batch_dim >= 0 and batch_dim < self._num_batch_dims
            ), f"batch_dim must be in range [0, {self._num_batch_dims})"
        else:
            num_repeats = list(num_repeats)
            use_num_repeats = False
            assert (
                len(num_repeats) == self._num_batch_dims
            ), f"num_repeats must be a sequence of length {self._num_batch_dims}"
            assert batch_dim is None, "batch_dim must be None if num_repeats is a sequence"

        if use_num_repeats:
            # Create repeat specifications for the tensor
            tensor_num_reps = [1] * self._tensor.dim()
            tensor_num_reps[batch_dim] = num_repeats
            mask_repeats = [1] * (self._num_batch_dims + 1)  # Initialize all dims to 1
            mask_repeats[batch_dim] = num_repeats  # Repeat only the specified batch dimension
            sample_sizes_repeats = [1] * self._num_batch_dims
            sample_sizes_repeats[batch_dim] = num_repeats
        else:
            tensor_num_reps = num_repeats + [1] * (self._tensor.dim() - self._num_batch_dims)
            mask_repeats = num_repeats + [1]
            sample_sizes_repeats = num_repeats

        tensor = self._tensor.repeat(tensor_num_reps)

        # Mask and sample sizes only need to be updated if they are computed already
        mask = self._mask.repeat(mask_repeats) if self._mask is not None else None
        sample_sizes = (
            self._sample_sizes.repeat(sample_sizes_repeats) if self._sample_sizes is not None else None
        )

        res = RaggedBatch(tensor, mask, sample_sizes, self._non_uniform_dim)
        return res

    def unsqueeze_batch_dim(self, dim: int) -> RaggedBatch:
        """Unsqueeze a batch dimension

        Important:
            The dimension to unsqueeze has to be among the batch dimensions (including adding a new batch dimension
            after the currently last batch dimensions, i.e. `dim=self.num_batch_dims`).

            For unsqueezing a data dimension, use :meth:`unsqueeze_data_dim` instead.

        Note:
            As the batch dimensions are always before the non-uniform dimension, the non-uniform dimension is shifted by 1 accordingly.

        Example:
            >>> example_batch.num_batch_dims
            2
            >>> example_batch.non_uniform_dim
            4
            >>> example_batch_unsqueezed = example_batch.unsqueeze_batch_dim(1)
            >>> example_batch_unsqueezed.non_uniform_dim
            5

        Args:
            dim: Batch dimension to add. Has to be in range [0, :attr:`num_batch_dims`].

        Returns:
            Resulting :class:`RaggedBatch` instance with the batch dimension added
        """
        assert dim >= 0 and dim <= self._num_batch_dims, f"dim must be in range [0, {self._num_batch_dims}]"

        tensor_res = self._tensor.unsqueeze(dim)
        mask_res = self._mask.unsqueeze(dim) if self._mask is not None else None
        sample_sizes_res = self._sample_sizes.unsqueeze(dim) if self._sample_sizes is not None else None
        res_non_uniform_dim = self._non_uniform_dim + 1
        res = RaggedBatch(tensor_res, mask_res, sample_sizes_res, res_non_uniform_dim)
        return res

    def squeeze_batch_dim(self, batch_dim: int) -> RaggedBatch:
        """Squeeze a batch dimension

        Note:
            This operation is not performed in-place, i.e. `this.tensor` is not changed.

        Args:
            batch_dim: Batch dimension to squeeze. Has to be in range [0, :attr:`num_batch_dims`).

        Returns:
            Resulting :class:`RaggedBatch` instance with the batch dimension squeezed
        """
        assert (
            batch_dim >= 0 and batch_dim < self._num_batch_dims
        ), f"batch_dim must be in range [0, {self._num_batch_dims})"
        if self.batch_shape[batch_dim] > 1:
            raise ValueError(
                f"Batch dimension {batch_dim} has size {self.batch_shape[batch_dim]} > 1. Cannot squeeze."
            )

        tensor_res = self._tensor.squeeze(batch_dim)
        mask_res = self._mask.squeeze(batch_dim) if self._mask is not None else None
        sample_sizes_res = self._sample_sizes.squeeze(batch_dim) if self._sample_sizes is not None else None
        res_non_uniform_dim = self._non_uniform_dim - 1
        res = RaggedBatch(tensor_res, mask_res, sample_sizes_res, res_non_uniform_dim)
        return res

    def reshape_batch_dims(self, new_batch_shape: Union[int, Tuple[int, ...]]) -> RaggedBatch:
        """Reshape the batch dimensions

        Note:
            This operation is not performed in-place, i.e. `this.tensor` is not changed.

        Important:
            The non-uniform dimension is adjusted to the new batch shape.

        Args:
            new_batch_shape: New batch shape

        Returns:
            Resulting :class:`RaggedBatch` instance with the batch dimensions reshaped
        """

        if isinstance(new_batch_shape, int):
            new_batch_shape = (new_batch_shape,)

        tensor_res = self._tensor.reshape(*new_batch_shape, *self._tensor.shape[self._num_batch_dims :])
        mask_res = (
            self._mask.reshape(*new_batch_shape, *self._mask.shape[self._num_batch_dims :])
            if self._mask is not None
            else None
        )
        sample_sizes_res = (
            self._sample_sizes.reshape(*new_batch_shape) if self._sample_sizes is not None else None
        )
        res_non_uniform_dim = self._non_uniform_dim - self._num_batch_dims + len(new_batch_shape)
        res = RaggedBatch(tensor_res, mask_res, sample_sizes_res, res_non_uniform_dim)

        return res

    def flatten_batch_dims(self) -> RaggedBatch:
        """Flatten the batch dimensions

        Note:
            This operation is not performed in-place, i.e. `this.tensor` is not changed.
        """
        # Specify the flattened size explicitly: PyTorch cannot infer ``-1``
        # when another dimension is zero and the tensor has no elements.
        res = self.reshape_batch_dims(self.total_num_samples_in_batch)
        return res

    def broadcast_batch_dims_to_shape(self, new_batch_shape: Sequence[int,]) -> RaggedBatch:
        new_batch_shape = torch.tensor(new_batch_shape)
        assert (
            len(new_batch_shape) == self._num_batch_dims
        ), f"New batch shape {new_batch_shape} has {len(new_batch_shape)} dimensions, but {self._num_batch_dims} dimensions are expected."

        batch_shape_tensor = torch.tensor(self._batch_shape)
        multiplier = torch.floor(new_batch_shape / batch_shape_tensor).int()
        assert torch.all(
            multiplier * batch_shape_tensor == new_batch_shape
        ), f"Cannot broadcast batch dimensions of {self._batch_shape} to {new_batch_shape}."
        res = self.repeat_samples(list(multiplier))

        return res

    @staticmethod
    def broadcast_batch_dims(data: Sequence[RaggedBatch]) -> Sequence[RaggedBatch]:
        """Broadcast the batch dimensions of a sequence of :class:`RaggedBatch` instances to common batch dimensions.

        Args:
            data: Sequence of :class:`RaggedBatch` instances

        Returns:
            Sequence of :class:`RaggedBatch` instances with the batch dimensions broadcasted to the common batch dimensions
        """
        batch_shapes = [torch.tensor(dt.batch_shape) for dt in data]
        try:
            batch_shapes_stacked = torch.stack(batch_shapes, dim=0)
        except RuntimeError as e:
            assert False, f"Cannot broadcast as number of batch dimensions does not match."
        max_batch_shape = torch.max(batch_shapes_stacked, dim=0).values
        multipliers = torch.floor(max_batch_shape.unsqueeze(0) / batch_shapes_stacked).int()
        res = [None] * len(data)
        for i, dt in enumerate(data):
            assert torch.all(
                batch_shapes[i] * multipliers[i] == max_batch_shape
            ), f"Cannot broadcast batch dimensions of {dt.batch_shape} to {max_batch_shape}."
            res[i] = dt.repeat_samples(list(multipliers[i]))
        return res

    def to_device(self, device: Union[torch.device, str]) -> RaggedBatch:
        """Get on device"""
        tensor = self._tensor.to(device=device)
        # Mask and sample sizes only need to be updated if they are computed already.
        # Note that while ensuring they are available here would potentially avoid re-computation,
        # it would also mean more memory transfers.
        mask = self._mask.to(device=device) if self._mask is not None else None
        sample_sizes = self._sample_sizes.to(device=device) if self._sample_sizes is not None else None
        res = RaggedBatch(tensor, mask, sample_sizes, self._non_uniform_dim)
        return res

    def cpu(self) -> RaggedBatch:
        """Get on the CPU"""
        return self.to_device(self._CPU)

    def to_dtype(self, dtype: torch.dtype) -> RaggedBatch:
        """Get with :attr:`tensor` converted to given data type"""
        tensor = self._tensor.to(dtype=dtype)
        # Mask and sample sizes updated (to avoid double computation if needed for both `this` and the new instance)
        res = RaggedBatch(tensor, self.mask, self.sample_sizes, self._non_uniform_dim)
        return res

    def detach(self) -> RaggedBatch:
        """Get with detached :attr:`tensor`"""
        res = RaggedBatch(self._tensor.detach(), self.mask, self.sample_sizes, self._non_uniform_dim)
        return res

    def apply(self, proc_step: ProcStep) -> Union[RaggedBatch, Tuple[RaggedBatch, ...]]:
        """Apply a function to :attr:`tensor` and get results as new `RaggedBatch` instance(s).

        See the `proc_step` parameter for requirements for the used function.

        Important:
            It is important to make sure that the tensors returned by `proc_step` fulfill the output requirements
            regarding the non-uniform dimension, sample sizes, and regarding the valid entries being stored first (i.e. lower indices),
            followed by filler values along the non-uniform dimension to ensure that the resulting :class:`RaggedBatch` instances are correct.
            See the `proc_step` parameter for more details.

        Args:
            proc_step: Function to process the data tensor. All the defined inputs (see below) are
                expected to be positional arguments.

                Args:
                    tensor: Will contain :attr:`tensor` of `this`
                    mask: If part of the function signature, will contain :attr:`mask` of `this`
                    sample_sizes: As a positional argument, this can only be part of the function
                        signature if `mask` is. If used, will contain :attr:`sample_sizes` of `this`
                Returns:
                    Either a tensor or a tuple of tensors. For each tensor, a
                    :class:`RaggedBatch` instance will be output from `apply()`. Note that for each
                    returned tensor, the non-uniform dimension, as well as the number of entries along
                    that dimension, must correspond to `this`. Also, for each sample, the the valid
                    entries must be located before any filler entries along the non-uniform dimension (as is in general the case for
                    the data stored in a :class:`RaggedBatch`, see documentation of the class). Note that the last
                    point is generally fulfilled if no permutations are applied to the data tensor, as the input `tensor`
                    contains valid entries first, followed by filler entries.

        Returns:
            :class:`RaggedBatch` instance or tuple of :class:`RaggedBatch` instances (depending on the output of `proc_step`),
            with the function applied to the data (i.e. to :attr:`tensor`).

        """
        num_args = proc_step.__code__.co_argcount
        if num_args == 1:
            args = (self._tensor,)
        elif num_args == 2:
            args = (self._tensor, self.mask)
        elif num_args == 3:
            args = (self._tensor, self.mask, self.sample_sizes)
        else:
            raise ValueError(
                f"Function {proc_step} has {num_args} arguments, but only 1, 2, or 3 are supported."
            )
        res_tensor = proc_step(*args)
        if isinstance(res_tensor, tuple):
            res = tuple(
                [RaggedBatch(rd, self.mask, self.sample_sizes, self._non_uniform_dim) for rd in res_tensor]
            )
        else:
            res = RaggedBatch(res_tensor, self.mask, self.sample_sizes, self._non_uniform_dim)
        return res

    def set_tensor(self, tensor: torch.Tensor):
        """Set :attr:`tensor`.

        Important:
            The batch shape, the non-uniform dimension, and the number of entries along
            that dimension must correspond to `this`. Also, for each sample, the valid
            entries must be located before any filler entries (as is in general the case for
            the data stored in a :class:`RaggedBatch` instance, see documentation of the class).

        Args:
            tensor: Data tensor to set

        """
        # Check that all batch dimensions match
        assert (
            tensor.shape[: self._num_batch_dims] == self._tensor.shape[: self._num_batch_dims]
        ), f"Batch shape of data to set {tensor.shape[:self._num_batch_dims]} does not match current batch shape {self._tensor.shape[:self._num_batch_dims]}."

        # Check that the non-uniform dimension size matches
        assert (
            tensor.shape[self._non_uniform_dim] == self._tensor.shape[self._non_uniform_dim]
        ), f"Maximum sample size of data to set ({tensor.shape[self._non_uniform_dim]}) does not match current maximum sample size ({self._tensor.shape[self._non_uniform_dim]})."

        assert (
            self._tensor.device == tensor.device
        ), f"Device of the data to set ({tensor.device}) does not match current device ({self._tensor.device})."
        self._tensor = tensor

    def split(self) -> Union[List[torch.Tensor], List[List]]:
        """Split contained data (i.e. the data in :attr:`tensor`) into individual samples.

        The batch dimensions are preserved in the nested list structure.
        For example, if the batch shape is (2, 3), the result will be a list of 2 lists,
        each containing 3 tensors.

        The returned samples are cropped to not contain any filler entries. This means that the
        returned tensors correspond to the actual sample sizes.

        Example:

            In the example below, the :meth:`split` operation is applied to a :class:`RaggedBatch` instance with
            a batch size of 4 (single batch dimension) and a maximum sample size of 3, resulting in a list
            of 4 tensors, and each tensor corresponding to a single sample without padded filler entries.
            Note that in the image below:

              - Letters indicate data entries that are valid (i.e. correspond to the actual data).
              - '*' Indicates padded filler entries (i.e. invalid entries) in the data.

            .. image:: images/Split_ragged.png
                :alt: Illustration of the split operation
                :align: center

            Each depicted element may represent a single value (corresponding to scalar data and 0 data dimensions),
            or itself represent a non-scalar entry (in case for one or more data dimensions).

        Returns:
            The individual samples in a nested list structure that reflects the original batch shape.
            The individual tensors correspond to the actual sample sizes, and do not contain padded filler entries.

            For a single batch dimension, returns a flat list of tensors.
            For multiple batch dimensions, returns a nested list structure mirroring the batch dimensions.
        """
        # Ensure non_uniform_dim is at the position right after the batch dimensions for easier processing
        need_transpose = self._non_uniform_dim != self._num_batch_dims

        if need_transpose:
            original_non_uniform_dim = self._non_uniform_dim
            self_preshaped = self.get_non_uniform_dimension_transposed_to(self._num_batch_dims)
        else:
            self_preshaped = self

        tensor_to_use = self_preshaped.tensor
        sample_sizes_to_use = self_preshaped.sample_sizes

        def _recursive_split(tensor, sample_sizes, batch_idx=(), batch_dim=0):
            if batch_dim == self._num_batch_dims:
                # At the innermost level, extract the actual sample by slicing to the correct size
                sample_size = sample_sizes[batch_idx]
                sample = tensor[batch_idx][:sample_size]
                if need_transpose:
                    original_non_uniform_dim_unbatched = original_non_uniform_dim - self._num_batch_dims
                    sample = sample.transpose(0, original_non_uniform_dim_unbatched)
                return sample
            else:
                # At intermediate levels, create a list for this batch dimension
                return [
                    _recursive_split(tensor, sample_sizes, batch_idx + (i,), batch_dim + 1)
                    for i in range(tensor.shape[batch_dim])
                ]

        # Use recursive function to create nested list structure
        result = _recursive_split(tensor_to_use, sample_sizes_to_use)
        return result

    def unsqueeze_data_dim(self, dim: int) -> RaggedBatch:
        """Unsqueeze the data tensor (i.e. :attr:`tensor`) along a dimension.

        Important:
            The dimension to unsqueeze has to be after the batch dimensions (including adding a new data dimension
            right after the batch dimensions, i.e. `dim=self.num_batch_dims`).

            For unsqueezing a batch dimension, use :meth:`unsqueeze_batch_dim` instead.

        Note:
            If the new dimension is inserted before the current non-uniform dimension, the
            non-uniform dimension is shifted by 1 accordingly.

        Example:
            >>> example_batch.num_batch_dims
            1
            >>> example_batch.non_uniform_dim
            1
            >>> example_batch_unsqueezed = example_batch.unsqueeze_data_dim(1)
            >>> example_batch_unsqueezed.non_uniform_dim
            2

        Args:
            dim: Dimension index into which to insert the new dimension

        Returns:
            Like self, but with the new dimension added, and the non-uniform
            dimension shifted accordingly if needed
        """
        if dim < 0:
            dim = self._tensor.dim() + 1 + dim
            # Note that dim can be equal to self._tensor.dim(), in which case it is added after the last dimension
            assert dim >= 0 and dim <= self._tensor.dim(), "Dimension outside the available range"
        assert dim >= self._num_batch_dims, "Can only add dimensions after the batch dimensions"

        tensor_res = self._tensor.unsqueeze(dim)
        # Check if the new dimension is inserted before the non-uniform dimension and if this is the case,
        # account for the fact that the non-uniform dimension was moved back by 1 dimension.
        if dim <= self._non_uniform_dim:
            non_uniform_dim_res = self._non_uniform_dim + 1
        else:
            non_uniform_dim_res = self._non_uniform_dim

        res = self.create_with_sample_sizes_like_self(tensor_res, non_uniform_dim_res)
        return res

    def __getitem__(self, item) -> torch.Tensor:
        """Item read access for :attr:`tensor`

        This is a shorthand for: `... = self.tensor[item]`.

        Note than as such, this allows for access to filler entries and does not check whether the accessed
        elements correspond to valid or filler entries.
        """
        return self._tensor[item]

    def __setitem__(self, item, value) -> None:
        """Item write access for :attr:`tensor`

        This is a shorthand for: `self.tensor[item] = ...`.

        Note than as such, this allows for access to filler entries and does not check whether the accessed
        elements correspond to valid or filler entries.
        """
        self._tensor[item] = value

    @property
    def device(self) -> torch.device:
        """Get the used device"""
        return self._tensor.device

    @property
    def shape(self) -> torch.Size:
        """Get the shape of the data tensor (i.e. :attr:`tensor`)

        The non-uniform dimension is reported as the size of the underlying :attr:`tensor`,
        i.e. to the maximum size among all samples.
        """
        return self._tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        """Type of the data elements (i.e. elements of :attr:`tensor`)"""
        return self._tensor.dtype

    @property
    def requires_grad(self) -> bool:
        """Get/set whether :attr:`tensor` requires gradients"""
        return self._tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._tensor.requires_grad = value

    def retain_grad(self) -> None:
        """Retain gradients for :attr:`tensor`"""
        self._tensor.retain_grad()

    @property
    def retains_grad(self) -> bool:
        """Get whether gradients are retained for :attr:`tensor`"""
        return self._tensor.retains_grad

    def size(self, *args, **kwargs):
        """Shorthand for `self.tensor.size(*args, **kwargs)`"""
        return self._tensor.size(*args, **kwargs)

    def dim(self) -> int:
        """Get the number of dimensions (of :attr:`tensor`)"""
        return self._tensor.dim()

    def int(self) -> RaggedBatch:
        """Convert type of :attr:`tensor` elements"""
        return self.create_with_sample_sizes_like_self(self._tensor.int())

    def long(self) -> RaggedBatch:
        """Convert type of :attr:`tensor` elements"""
        return self.create_with_sample_sizes_like_self(self._tensor.long())

    def bool(self) -> RaggedBatch:
        """Convert type of :attr:`tensor` elements"""
        return self.create_with_sample_sizes_like_self(self._tensor.bool())

    def half(self) -> RaggedBatch:
        """Convert type of :attr:`tensor` elements"""
        return self.create_with_sample_sizes_like_self(self._tensor.half())

    def bfloat16(self) -> RaggedBatch:
        """Convert type of :attr:`tensor` elements"""
        return self.create_with_sample_sizes_like_self(self._tensor.bfloat16())

    def float(self) -> RaggedBatch:
        """Convert type of :attr:`tensor` elements"""
        return self.create_with_sample_sizes_like_self(self._tensor.float())

    def double(self) -> RaggedBatch:
        """Convert type of :attr:`tensor` elements"""
        return self.create_with_sample_sizes_like_self(self._tensor.double())

    def cfloat(self) -> RaggedBatch:
        """Convert type of :attr:`tensor` elements"""
        return self.create_with_sample_sizes_like_self(self._tensor.cfloat())

    def cdouble(self) -> RaggedBatch:
        """Convert type of :attr:`tensor` elements"""
        return self.create_with_sample_sizes_like_self(self._tensor.cdouble())

    def to(self, *args, **kwargs) -> RaggedBatch:
        """Create a new :class:`RaggedBatch` instance converted as specified.

        This is a shorthand for:
        `self.create_with_sample_sizes_like_self(self._tensor.to(*args, **kwargs))`.

        Note:
            The conversion is primarily performed on the :attr:`tensor`.
            The :attr:`sample_sizes` and :attr:`mask` are adjusted accordingly if needed
            (e.g. when converting to a different device, but not when converting to a different
            dtype, as the dtype is only relevant for the :attr:`tensor`).

        Args:
            *args: Arguments for :meth:`torch.Tensor.to` as applied to :attr:`tensor`
            **kwargs: Keyword arguments for :meth:`torch.Tensor.to` as applied to :attr:`tensor`

        Returns:
            A new :class:`RaggedBatch` instance with the :attr:`tensor` converted to the given type
            and the :attr:`sample_sizes` and :attr:`mask` adjusted accordingly if needed.
        """
        return self.create_with_sample_sizes_like_self(self._tensor.to(*args, **kwargs))

    def __repr__(self) -> str:
        mask_str = "*uninitialized*" if self._mask is None else f"mask={self._mask}"
        sample_sizes_str = (
            "*uninitialized*" if self._sample_sizes is None else f"sample_sizes={self._sample_sizes}"
        )
        res = f"RaggedBatch(tensor={self._tensor}, mask={mask_str}, samples_sizes={sample_sizes_str}, non_uniform_dim={self._non_uniform_dim}, batch_shape={self.batch_shape})"
        return res
