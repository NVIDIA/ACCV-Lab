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
import torch
from accvlab.batching_helpers.batched_processing_py import RaggedBatch


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
    ],
)
def test_ragged_batch_constructor(dtype, capsys):
    # To see console outputs, use `with capsys.disabled(): ...`
    batch_shape = (16, 2)
    max_sample_size = 50
    num_tries = 1000
    additional_shape = (2, 50, 3)

    for device in ["cuda:0", "cpu"]:
        for _ in range(num_tries):
            shape = (*batch_shape, max_sample_size, *additional_shape)
            if dtype.is_floating_point:
                data = torch.randn(shape, device=device, dtype=dtype)
            else:
                # Use a bounded integer range to avoid overflow in smaller bitwidths
                data = torch.randint(low=-1000, high=1000, size=shape, device=device, dtype=dtype)
            mask = torch.ones((*batch_shape, max_sample_size), dtype=torch.bool, device=device)
            sample_sizes = torch.randint(0, max_sample_size + 1, (*batch_shape,), device=device)
            for s0 in range(batch_shape[0]):
                for s1 in range(batch_shape[1]):
                    mask[s0, s1, sample_sizes[s0, s1] :] = False

            ragged_batch_ref = RaggedBatch(data, mask, sample_sizes)
            ragged_batch_from_mask = RaggedBatch(data, mask=mask, sample_sizes=None)
            ragged_batch_from_sample_sizes = RaggedBatch(data, mask=None, sample_sizes=sample_sizes)

            assert torch.all(
                ragged_batch_ref.mask == ragged_batch_from_sample_sizes.mask
            ), "Detected error in mask generated in constructor when `sample_sizes` is provided"
            assert torch.all(
                ragged_batch_ref.sample_sizes == ragged_batch_from_mask.sample_sizes
            ), "Detected error in sample sizes generated in constructor when `mask` is provided"

            # Dtype/device consistency
            assert ragged_batch_ref.tensor.dtype == dtype
            assert ragged_batch_from_mask.tensor.dtype == dtype
            assert ragged_batch_from_sample_sizes.tensor.dtype == dtype
            assert ragged_batch_ref.tensor.device.type == torch.device(device).type


def test_ragged_batch_properties(capsys):
    """Test basic properties of RaggedBatch"""
    batch_shape = (2, 3)
    max_sample_size = 5
    additional_shape = (2,)

    for device in ["cuda:0", "cpu"]:
        data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device)
        sample_sizes = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)
        ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

        # Test device property
        assert ragged_batch.device == data.device

        # Test shape property
        assert ragged_batch.shape == data.shape

        # Test dtype property
        assert ragged_batch.dtype == data.dtype

        assert ragged_batch.non_uniform_dim == 2
        assert ragged_batch.batch_shape == batch_shape
        assert ragged_batch.num_batch_dims == 2
        assert ragged_batch.total_num_samples_in_batch == 6

        # Test requires_grad property
        ragged_batch.requires_grad = True
        assert ragged_batch.requires_grad == True
        assert ragged_batch.tensor.requires_grad == True

        # Test size method
        assert ragged_batch.size() == data.size()
        assert ragged_batch.size(0) == data.size(0)

        # Test dim method
        assert ragged_batch.dim() == data.dim()


@pytest.mark.parametrize(
    ("kwargs", "expected_batch_shape", "expected_tensor_shape", "expected_dtype"),
    [
        ({}, (0,), (0, 0), torch.float32),
        ({"num_batch_dims": 2}, (0, 0), (0, 0, 0), torch.float32),
        ({"batch_shape": 4}, (4,), (4, 0), torch.float32),
        ({"batch_shape": [2, 3], "dtype": torch.int64}, (2, 3), (2, 3, 0), torch.int64),
    ],
)
def test_ragged_batch_empty(kwargs, expected_batch_shape, expected_tensor_shape, expected_dtype):
    num_dims = len(expected_tensor_shape)
    non_uniform_dim = len(expected_batch_shape)
    ragged_batch = RaggedBatch.Empty(
        num_dims=num_dims,
        non_uniform_dim=non_uniform_dim,
        device="cpu",
        **kwargs,
    )

    assert ragged_batch.batch_shape == expected_batch_shape
    assert ragged_batch.num_batch_dims == len(expected_batch_shape)
    assert ragged_batch.tensor.shape == expected_tensor_shape
    assert ragged_batch.tensor.dtype == expected_dtype
    assert ragged_batch.mask.shape == (*expected_batch_shape, 0)
    assert ragged_batch.sample_sizes.shape == expected_batch_shape
    assert torch.count_nonzero(ragged_batch.sample_sizes) == 0


def test_ragged_batch_flatten_empty_data_batch():
    batch_shape = (2, 3)
    data = torch.empty((*batch_shape, 0, 9))
    sample_sizes = torch.zeros(batch_shape, dtype=torch.int64)
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    flattened = ragged_batch.flatten_batch_dims()

    assert flattened.batch_shape == (6,)
    assert flattened.num_batch_dims == 1
    assert flattened.non_uniform_dim == 1
    assert flattened.tensor.shape == (6, 0, 9)
    assert flattened.mask.shape == (6, 0)
    assert flattened.sample_sizes.shape == (6,)
    assert torch.count_nonzero(flattened.sample_sizes) == 0


def test_ragged_batch_type_conversion(capsys):
    """Test type conversion methods of RaggedBatch"""
    batch_size = 3
    max_sample_size = 5
    additional_shape = (2,)

    for device in ["cuda:0", "cpu"]:
        data = torch.randn((batch_size, max_sample_size, *additional_shape), device=device)
        sample_sizes = torch.tensor([3, 2, 5], dtype=torch.int64, device=device)
        ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

        # Test type conversion methods
        assert ragged_batch.int().dtype == torch.int32
        assert ragged_batch.long().dtype == torch.int64
        assert ragged_batch.bool().dtype == torch.bool
        assert ragged_batch.float().dtype == torch.float32
        assert ragged_batch.double().dtype == torch.float64
        assert ragged_batch.half().dtype == torch.float16
        assert ragged_batch.bfloat16().dtype == torch.bfloat16
        assert ragged_batch.cfloat().dtype == torch.cfloat
        assert ragged_batch.cdouble().dtype == torch.cdouble


def test_ragged_batch_indexing(capsys):
    """Test indexing operations of RaggedBatch"""
    batch_size = 3
    max_sample_size = 5
    additional_shape = (2,)

    for device in ["cuda:0", "cpu"]:
        data = torch.randn((batch_size, max_sample_size, *additional_shape), device=device)
        sample_sizes = torch.tensor([3, 2, 5], dtype=torch.int64, device=device)
        ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

        # Test getitem
        assert torch.all(ragged_batch[0] == data[0])
        assert torch.all(ragged_batch[1, 0] == data[1, 0])

        # Test setitem
        new_value = torch.randn_like(data[0])
        ragged_batch[0] = new_value
        assert torch.all(ragged_batch[0] == new_value)


def test_ragged_batch_gradients(capsys):
    """Test gradient-related functionality of RaggedBatch"""
    batch_size = 3
    max_sample_size = 5
    additional_shape = (2,)

    for device in ["cuda:0", "cpu"]:
        data = torch.randn((batch_size, max_sample_size, *additional_shape), device=device)
        sample_sizes = torch.tensor([3, 2, 5], dtype=torch.int64, device=device)
        ragged_batch_leaf = RaggedBatch(data, sample_sizes=sample_sizes)
        # Use a non-leaf tensor to test retain_grad (which is a no-op for leaf tensors).
        # Note that `data.requires_grad == True` is needed as otherwise, `data_non_leaf`
        # would be a leaf tensor.
        data.requires_grad = True
        data_non_leaf = data + 1.0
        ragged_batch_non_leaf = RaggedBatch(data_non_leaf, sample_sizes=sample_sizes)

        # Test retain_grad
        ragged_batch_non_leaf.retain_grad()
        assert ragged_batch_non_leaf.tensor.retains_grad

        # Test requires_grad setter
        ragged_batch_leaf.requires_grad = False
        assert not ragged_batch_leaf.requires_grad
        assert not ragged_batch_leaf.tensor.requires_grad
        ragged_batch_leaf.requires_grad = True
        assert ragged_batch_leaf.requires_grad
        assert ragged_batch_leaf.tensor.requires_grad


def test_ragged_batch_repr(capsys):
    """Test string representation of RaggedBatch"""
    batch_size = 3
    max_sample_size = 5
    additional_shape = (2,)

    for device in ["cuda:0", "cpu"]:
        data = torch.randn((batch_size, max_sample_size, *additional_shape), device=device)
        sample_sizes = torch.tensor([3, 2, 5], dtype=torch.int64, device=device)
        ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

        # Test string representation
        repr_str = repr(ragged_batch)
        assert "RaggedBatch" in repr_str
        assert "tensor=" in repr_str
        assert "mask=" in repr_str
        assert "sample_sizes=" in repr_str
        assert "non_uniform_dim=" in repr_str
        assert "batch_shape=" in repr_str


def test_ragged_batch_as_self_with_cloned_data(capsys):
    """Test creating a new RaggedBatch with cloned data but same structure"""
    batch_size = 3
    max_sample_size = 5
    additional_shape = (2,)

    for device in ["cuda:0", "cpu"]:
        # Create original ragged batch
        data = torch.randn((batch_size, max_sample_size, *additional_shape), device=device)
        sample_sizes = torch.tensor([3, 2, 5], dtype=torch.int64, device=device)
        ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

        # Create new ragged batch with cloned data
        new_ragged_batch = ragged_batch.as_self_with_cloned_data()

        # Check that structure is preserved
        assert torch.all(new_ragged_batch.sample_sizes == ragged_batch.sample_sizes)
        assert torch.all(new_ragged_batch.mask == ragged_batch.mask)
        assert new_ragged_batch.non_uniform_dim == ragged_batch.non_uniform_dim

        # Check that data is the same
        assert torch.all(new_ragged_batch.tensor == ragged_batch.tensor)
        assert new_ragged_batch.tensor.device == ragged_batch.tensor.device
        assert new_ragged_batch.tensor.dtype == ragged_batch.tensor.dtype

        # Check that the data is not shared
        orig_val = ragged_batch.tensor[0, 0, 0]
        new_ragged_batch.tensor[0, 0, 0] = 100.0
        assert ragged_batch.tensor[0, 0, 0] == orig_val


def test_ragged_batch_create_with_sample_sizes_like_self(capsys):
    """Test creating a new RaggedBatch with same sample sizes but different data"""
    batch_size = 3
    max_sample_size = 5
    additional_shape = (2,)

    for device in ["cuda:0", "cpu"]:
        # Create original ragged batch
        data = torch.randn((batch_size, max_sample_size, *additional_shape), device=device)
        sample_sizes = torch.tensor([3, 2, 5], dtype=torch.int64, device=device)
        ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

        # Create new data with different shape but same batch size and max sample size
        new_additional_shape = (4, 3)
        new_data = torch.randn((batch_size, max_sample_size, *new_additional_shape), device=device)

        # Create new ragged batch with same sample sizes
        new_ragged_batch = ragged_batch.create_with_sample_sizes_like_self(new_data)

        # Check that structure is preserved
        assert torch.all(new_ragged_batch.sample_sizes == ragged_batch.sample_sizes)
        assert torch.all(new_ragged_batch.mask == ragged_batch.mask)
        assert new_ragged_batch.non_uniform_dim == ragged_batch.non_uniform_dim

        # Check data and shape
        assert torch.all(new_ragged_batch.tensor == new_data)
        assert new_ragged_batch.tensor.shape == new_data.shape
        assert new_ragged_batch.tensor.device == new_data.device

        # Test with different non_uniform_dim
        new_non_uniform_dim = 2
        new_data_transposed = new_data.transpose(1, new_non_uniform_dim)
        new_ragged_batch = ragged_batch.create_with_sample_sizes_like_self(
            new_data_transposed, non_uniform_dim=new_non_uniform_dim
        )
        assert new_ragged_batch.non_uniform_dim == new_non_uniform_dim

        # Test with different device
        if device == "cuda:0":
            new_device = "cpu"
        else:
            new_device = "cuda:0"
        new_ragged_batch = ragged_batch.create_with_sample_sizes_like_self(new_data, device=new_device)
        assert new_ragged_batch.tensor.device == torch.device(new_device)
        assert new_ragged_batch.mask.device == torch.device(new_device)
        assert new_ragged_batch.sample_sizes.device == torch.device(new_device)


def test_ragged_batch_get_non_uniform_dimension_transposed_to(capsys):
    """Test transposing the non-uniform dimension to different positions"""
    batch_shape = (2, 3)
    max_sample_size = 5
    additional_shape = (2,)

    for device in ["cuda:0", "cpu"]:
        # Create original ragged batch with non_uniform_dim=1
        data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device)
        sample_sizes = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)
        ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

        # Test transposing to different dimensions
        for target_dim in range(ragged_batch.num_batch_dims, data.dim()):
            transposed = ragged_batch.get_non_uniform_dimension_transposed_to(target_dim)

            # Check that structure is preserved
            assert torch.all(transposed.sample_sizes == ragged_batch.sample_sizes)
            assert transposed.non_uniform_dim == target_dim

            # Check that data is preserved (after accounting for transpose)
            if target_dim == ragged_batch.non_uniform_dim:
                assert torch.all(transposed.tensor == ragged_batch.tensor)
            else:
                # Get the permutation that would transpose non_uniform_dim to target_dim
                dims = list(range(data.dim()))
                dims.remove(ragged_batch.non_uniform_dim)
                dims.insert(target_dim, ragged_batch.non_uniform_dim)
                expected_data = ragged_batch.tensor.permute(dims)
                assert torch.all(transposed.tensor == expected_data)

            # Check that mask remains unchanged
            assert torch.all(transposed.mask == ragged_batch.mask)

        # Test that transposing to dimension 0 raises an error
        with pytest.raises(AssertionError):
            ragged_batch.get_non_uniform_dimension_transposed_to(0)


def test_ragged_batch_get_existence_weights(capsys):
    """Test getting existence weights for ragged batch"""
    batch_shape = (2, 3)
    max_sample_size = 5
    additional_shape = (2,)
    device = "cuda:0"  # Only test on GPU

    # Create ragged batch
    data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device)
    sample_sizes = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    # Test with default dtype (float32)
    weights = ragged_batch.get_existence_weights()
    assert weights.dtype == torch.float32
    assert weights.shape == data.shape
    assert weights.device == data.device

    # Check that weights are 1.0 for valid entries and 0.0 for padded entries
    for s0 in range(batch_shape[0]):
        for s1 in range(batch_shape[1]):
            assert torch.all(weights[s0, s1, : sample_sizes[s0, s1]] == 1.0)
            assert torch.all(weights[s0, s1, sample_sizes[s0, s1] :] == 0.0)

    # Test with different dtype
    weights_double = ragged_batch.get_existence_weights(dtype=torch.float64)
    assert weights_double.dtype == torch.float64
    assert torch.all(weights_double == weights.to(torch.float64))

    # Test with different non-uniform dimension
    ragged_batch_transposed = ragged_batch.get_non_uniform_dimension_transposed_to(2)
    weights_transposed = ragged_batch_transposed.get_existence_weights()
    assert weights_transposed.shape == ragged_batch_transposed.tensor.shape

    # Verify weights are correctly applied to data
    weighted_data = ragged_batch.tensor * weights
    for s0 in range(batch_shape[0]):
        for s1 in range(batch_shape[1]):
            assert torch.all(
                weighted_data[s0, s1, : sample_sizes[s0, s1]]
                == ragged_batch.tensor[s0, s1, : sample_sizes[s0, s1]]
            )
            assert torch.all(weighted_data[s0, s1, sample_sizes[s0, s1] :] == 0.0)


def test_ragged_batch_repeat_samples(capsys):
    """Test repeating samples in ragged batch"""
    batch_shape = (2, 3)
    max_sample_size = 5
    additional_shape = (2,)
    device = "cuda:0"  # Only test on GPU

    # Create ragged batch
    data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device)
    sample_sizes = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    # Test repeating samples in batch dimension 0
    num_repeats = 2
    repeated = ragged_batch.repeat_samples(num_repeats, batch_dim=0)

    # Check shapes
    assert repeated.tensor.shape[0] == batch_shape[0] * num_repeats
    assert repeated.mask.shape[0] == batch_shape[0] * num_repeats
    assert repeated.sample_sizes.shape[0] == batch_shape[0] * num_repeats

    # Check that data is correctly repeated
    for s0 in range(batch_shape[0]):
        for s1 in range(batch_shape[1]):
            for j in range(num_repeats):
                s0_rep = s0 + j * batch_shape[0]
                assert torch.all(repeated.tensor[s0_rep, s1] == data[s0, s1])
            assert torch.all(repeated.mask[s0_rep, s1] == ragged_batch.mask[s0, s1])
            assert repeated.sample_sizes[s0_rep, s1] == sample_sizes[s0, s1]

    # Test repeating samples in batch dimension 1
    num_repeats = 3
    repeated = ragged_batch.repeat_samples(num_repeats, batch_dim=1)

    # Check shapes
    assert repeated.tensor.shape[1] == batch_shape[1] * num_repeats
    assert repeated.mask.shape[1] == batch_shape[1] * num_repeats
    assert repeated.sample_sizes.shape[1] == batch_shape[1] * num_repeats

    # Check that data is correctly repeated
    for s0 in range(batch_shape[0]):
        for s1 in range(batch_shape[1]):
            for j in range(num_repeats):
                s1_rep = s1 + j * batch_shape[1]
                assert torch.all(repeated.tensor[s0, s1_rep] == data[s0, s1])
            assert torch.all(repeated.mask[s0, s1_rep] == ragged_batch.mask[s0, s1])
            assert repeated.sample_sizes[s0, s1_rep] == sample_sizes[s0, s1]


def test_ragged_batch_apply_single_input(capsys):
    """Test applying a function to ragged batch data"""
    batch_size = 3
    max_sample_size = 5
    additional_shape = (2,)
    device = "cuda:0"  # Only test on GPU

    # Create ragged batch
    data = torch.randn((batch_size, max_sample_size, *additional_shape), device=device)
    sample_sizes = torch.tensor([3, 2, 5], dtype=torch.int64, device=device)
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    # Test applying a simple function (square)
    squared = ragged_batch.apply(lambda x: x**2)

    # Check shapes
    assert squared.tensor.shape == data.shape
    assert squared.mask.shape == ragged_batch.mask.shape
    assert squared.sample_sizes.shape == sample_sizes.shape

    # Check that function was applied correctly
    for i in range(batch_size):
        assert torch.all(squared.tensor[i, : sample_sizes[i]] == data[i, : sample_sizes[i]] ** 2)
        assert torch.all(squared.mask[i] == ragged_batch.mask[i])
        assert squared.sample_sizes[i] == sample_sizes[i]

    # Test with a more complex function
    def combined_func(x):
        return (torch.sin(x), torch.cos(x))

    transformed_sin, transformed_cos = ragged_batch.apply(combined_func)

    # Check shapes
    assert transformed_sin.tensor.shape == data.shape
    assert transformed_cos.tensor.shape == data.shape
    assert transformed_sin.mask.shape == ragged_batch.mask.shape
    assert transformed_cos.mask.shape == ragged_batch.mask.shape
    assert transformed_sin.sample_sizes.shape == sample_sizes.shape
    assert transformed_cos.sample_sizes.shape == sample_sizes.shape

    # Check that function was applied correctly
    for i in range(batch_size):
        expected_sin = torch.sin(data[i, : sample_sizes[i]])
        expected_cos = torch.cos(data[i, : sample_sizes[i]])
        assert torch.allclose(transformed_sin.tensor[i, : sample_sizes[i]], expected_sin)
        assert torch.allclose(transformed_cos.tensor[i, : sample_sizes[i]], expected_cos)
        assert torch.all(transformed_sin.mask[i] == ragged_batch.mask[i])
        assert torch.all(transformed_cos.mask[i] == ragged_batch.mask[i])
        assert transformed_sin.sample_sizes[i] == sample_sizes[i]
        assert transformed_cos.sample_sizes[i] == sample_sizes[i]


def test_ragged_batch_split(capsys):
    batch_shape = (2, 3)
    max_sample_size = 5
    additional_shape = (2,)
    device = "cuda:0"  # Only test on GPU

    # Create ragged batch
    data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device)
    sample_sizes = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    # Split into individual samples
    split_samples = ragged_batch.split()

    # Check number of samples
    assert len(split_samples) == batch_shape[0]
    assert len(split_samples[0]) == batch_shape[1]

    # Check each sample
    for s0 in range(batch_shape[0]):
        for s1 in range(batch_shape[1]):
            # Check shape - should be (sample_size, *additional_shape)
            expected_shape = (sample_sizes[s0, s1], *additional_shape)
            assert split_samples[s0][s1].shape == expected_shape

        # Check content matches original data
        assert torch.allclose(split_samples[s0][s1], data[s0, s1, : sample_sizes[s0, s1]])


def test_ragged_batch_unsqueeze(capsys):
    batch_shape = (2, 3)
    max_sample_size = 5
    additional_shape = (2,)
    device = "cuda:0"  # Only test on GPU

    # Create ragged batch
    data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device)
    sample_sizes = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    # Test unsqueeze at different dimensions
    # Test unsqueeze after non-uniform dim (should not affect non-uniform dim)
    unsqueezed_after = ragged_batch.unsqueeze_data_dim(3)
    assert unsqueezed_after.tensor.shape == (*batch_shape, max_sample_size, 1, *additional_shape)
    assert unsqueezed_after.non_uniform_dim == ragged_batch.non_uniform_dim
    assert torch.allclose(unsqueezed_after.tensor.squeeze(3), ragged_batch.tensor)

    # Test unsqueeze before non-uniform dim (should shift non-uniform dim)
    unsqueezed_before = ragged_batch.unsqueeze_data_dim(2)
    assert unsqueezed_before.tensor.shape == (*batch_shape, 1, max_sample_size, *additional_shape)
    assert unsqueezed_before.non_uniform_dim == ragged_batch.non_uniform_dim + 1
    assert torch.allclose(unsqueezed_before.tensor.squeeze(2), ragged_batch.tensor)

    # Test unsqueeze at end (should not affect non-uniform dim)
    unsqueezed_end = ragged_batch.unsqueeze_data_dim(-1)
    assert unsqueezed_end.tensor.shape == (*batch_shape, max_sample_size, *additional_shape, 1)
    assert unsqueezed_end.non_uniform_dim == ragged_batch.non_uniform_dim
    assert torch.allclose(unsqueezed_end.tensor.squeeze(-1), ragged_batch.tensor)

    # Test that sample sizes and mask are preserved
    assert torch.all(unsqueezed_after.sample_sizes == sample_sizes)
    assert torch.all(unsqueezed_before.sample_sizes == sample_sizes)
    assert torch.all(unsqueezed_end.sample_sizes == sample_sizes)
    assert torch.all(unsqueezed_after.mask == ragged_batch.mask)
    assert torch.all(unsqueezed_before.mask == ragged_batch.mask)
    assert torch.all(unsqueezed_end.mask == ragged_batch.mask)


def test_ragged_batch_repeat_samples_sequence(capsys):
    """Test repeating samples in ragged batch using sequence of ints"""
    batch_shape = (2, 3)
    max_sample_size = 5
    additional_shape = (2,)
    device = "cuda:0"

    data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device)
    sample_sizes = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    # Test repeating samples using sequence of ints
    num_repeats = [2, 3]
    repeated = ragged_batch.repeat_samples(num_repeats)

    # Check shapes
    expected_batch_shape = (batch_shape[0] * num_repeats[0], batch_shape[1] * num_repeats[1])
    assert repeated.tensor.shape[:2] == expected_batch_shape
    assert repeated.mask.shape[:2] == expected_batch_shape
    assert repeated.sample_sizes.shape == expected_batch_shape

    # Check data repetition (test a few key positions)
    assert torch.all(repeated.tensor[0, 0] == data[0, 0])
    assert torch.all(repeated.tensor[2, 0] == data[0, 0])  # Repeated in dim 0
    assert torch.all(repeated.tensor[0, 3] == data[0, 0])  # Repeated in dim 1
    assert repeated.sample_sizes[0, 0] == sample_sizes[0, 0]

    # Test 3D batch shape
    batch_shape_3d = (2, 3, 4)
    data_3d = torch.randn((*batch_shape_3d, max_sample_size, *additional_shape), device=device)
    sample_sizes_3d = torch.randint(1, max_sample_size + 1, batch_shape_3d, dtype=torch.int64, device=device)
    ragged_batch_3d = RaggedBatch(data_3d, sample_sizes=sample_sizes_3d)

    repeated_3d = ragged_batch_3d.repeat_samples([2, 3, 1])
    expected_batch_shape_3d = (4, 9, 4)
    assert repeated_3d.tensor.shape[:3] == expected_batch_shape_3d
    assert repeated_3d.mask.shape[:3] == expected_batch_shape_3d
    assert repeated_3d.sample_sizes.shape == expected_batch_shape_3d


def test_ragged_batch_broadcast_batch_dims(capsys):
    """Test broadcasting batch dimensions of multiple RaggedBatch instances"""
    max_sample_size = 5
    additional_shape = (2,)
    device = "cuda:0"

    # Create batches with broadcastable shapes: (2,3), (4,3), (2,6) -> (4,6)
    data1 = torch.randn((2, 3, max_sample_size, *additional_shape), device=device)
    sample_sizes1 = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)
    ragged_batch1 = RaggedBatch(data1, sample_sizes=sample_sizes1)

    data2 = torch.randn((4, 3, max_sample_size, *additional_shape), device=device)
    sample_sizes2 = torch.randint(1, max_sample_size + 1, (4, 3), dtype=torch.int64, device=device)
    ragged_batch2 = RaggedBatch(data2, sample_sizes=sample_sizes2)

    data3 = torch.randn((2, 6, max_sample_size, *additional_shape), device=device)
    sample_sizes3 = torch.randint(1, max_sample_size + 1, (2, 6), dtype=torch.int64, device=device)
    ragged_batch3 = RaggedBatch(data3, sample_sizes=sample_sizes3)

    # Test broadcasting
    broadcasted = RaggedBatch.broadcast_batch_dims([ragged_batch1, ragged_batch2, ragged_batch3])

    # Check that all batches have the same broadcasted shape
    expected_batch_shape = (4, 6)
    assert len(broadcasted) == 3
    for batch in broadcasted:
        assert batch.tensor.shape[:2] == expected_batch_shape
        assert batch.sample_sizes.shape == expected_batch_shape

    # Verify key data positions are preserved
    assert torch.all(broadcasted[0].tensor[:2, :3] == data1)  # Original data preserved
    assert torch.all(broadcasted[1].tensor[:4, :3] == data2)  # Original data preserved
    assert torch.all(broadcasted[2].tensor[:2, :6] == data3)  # Original data preserved

    # Test simple broadcasting: (1,2) and (2,1) -> (2,2)
    simple_data1 = torch.randn((1, 2, max_sample_size, *additional_shape), device=device)
    simple_batch1 = RaggedBatch(
        simple_data1, sample_sizes=torch.tensor([[3, 2]], dtype=torch.int64, device=device)
    )

    simple_data2 = torch.randn((2, 1, max_sample_size, *additional_shape), device=device)
    simple_batch2 = RaggedBatch(
        simple_data2, sample_sizes=torch.tensor([[3], [2]], dtype=torch.int64, device=device)
    )

    simple_broadcasted = RaggedBatch.broadcast_batch_dims([simple_batch1, simple_batch2])
    assert simple_broadcasted[0].tensor.shape[:2] == (2, 2)
    assert simple_broadcasted[1].tensor.shape[:2] == (2, 2)

    # Test error cases
    single_batch = RaggedBatch(
        torch.randn((3, max_sample_size, *additional_shape), device=device),
        sample_sizes=torch.tensor([2, 3, 1], dtype=torch.int64, device=device),
    )
    with pytest.raises(AssertionError):
        RaggedBatch.broadcast_batch_dims([ragged_batch1, single_batch])

    # Non-broadcastable shapes
    non_broadcastable_data = torch.randn((3, 3, max_sample_size, *additional_shape), device=device)
    non_broadcastable_batch = RaggedBatch(
        non_broadcastable_data,
        sample_sizes=torch.randint(1, max_sample_size + 1, (3, 3), dtype=torch.int64, device=device),
    )
    with pytest.raises(AssertionError):
        RaggedBatch.broadcast_batch_dims([ragged_batch1, non_broadcastable_batch])


def test_ragged_batch_repeat_samples_edge_cases(capsys):
    """Test edge cases for repeat_samples with sequence of ints"""
    batch_shape = (2, 3)
    max_sample_size = 5
    additional_shape = (2,)
    device = "cuda:0"

    data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device)
    sample_sizes = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    # Test with all ones (should not change anything)
    repeated_ones = ragged_batch.repeat_samples([1, 1])
    assert repeated_ones.tensor.shape == ragged_batch.tensor.shape
    assert torch.all(repeated_ones.tensor == ragged_batch.tensor)

    # Test error cases
    with pytest.raises(AssertionError):
        ragged_batch.repeat_samples([2, 3, 4])  # Wrong sequence length

    with pytest.raises(AssertionError):
        ragged_batch.repeat_samples([2, 3], batch_dim=0)  # batch_dim with sequence


def test_ragged_batch_from_oversize_tensor(capsys):
    """Test creating a ragged batch from an oversized tensor using the class method"""
    batch_shape = (2, 3)
    max_sample_size = 10
    additional_shape = (2,)
    device = "cuda:0"

    # Create data with larger max_sample_size than needed
    data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device)
    sample_sizes = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)

    # Test using the class method
    ragged_batch = RaggedBatch.FromOversizeTensor(data, sample_sizes=sample_sizes)

    # Check that tensor is narrowed to max sample size (5)
    expected_max_size = 5
    assert ragged_batch.max_sample_size == expected_max_size
    assert ragged_batch.tensor.shape[ragged_batch.non_uniform_dim] == expected_max_size
    assert ragged_batch.mask.shape[-1] == expected_max_size

    # Check that data is preserved
    assert torch.all(ragged_batch.tensor == data[..., :expected_max_size, :])
    assert torch.all(ragged_batch.sample_sizes == sample_sizes)

    # Test with different non-uniform dimension
    batch_shape_3d = (2, 3, 4)
    data_3d = torch.randn((*batch_shape_3d, max_sample_size, *additional_shape), device=device)
    sample_sizes_3d = torch.randint(1, 6, batch_shape_3d, dtype=torch.int64, device=device)
    ragged_batch_3d = RaggedBatch.FromOversizeTensor(data_3d, sample_sizes=sample_sizes_3d, non_uniform_dim=3)

    expected_max_size_3d = torch.max(sample_sizes_3d).item()
    assert ragged_batch_3d.max_sample_size == expected_max_size_3d
    assert ragged_batch_3d.tensor.shape[3] == expected_max_size_3d
    assert ragged_batch_3d.mask.shape[-1] == expected_max_size_3d


def test_ragged_batch_broadcast_batch_dims_to_shape(capsys):
    """Test broadcasting batch dimensions to a specific shape"""
    max_sample_size = 5
    additional_shape = (2,)
    device = "cuda:0"

    # Test 2D asymmetric broadcasting: (2, 5) -> (8, 15) - using factors 4 and 3
    batch_shape = (2, 5)
    data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device)
    sample_sizes = torch.randint(1, max_sample_size + 1, batch_shape, dtype=torch.int64, device=device)
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    new_batch_shape = (8, 15)
    broadcasted = ragged_batch.broadcast_batch_dims_to_shape(new_batch_shape)

    # Check shapes
    assert broadcasted.tensor.shape[:2] == new_batch_shape
    assert broadcasted.mask.shape[:2] == new_batch_shape
    assert broadcasted.sample_sizes.shape == new_batch_shape

    # Check data repetition (multipliers should be [4, 3])
    assert torch.all(broadcasted.tensor[0, 0] == data[0, 0])
    assert torch.all(broadcasted.tensor[2, 0] == data[0, 0])  # Repeated in dim 0
    assert torch.all(broadcasted.tensor[4, 0] == data[0, 0])  # Repeated in dim 0
    assert torch.all(broadcasted.tensor[6, 0] == data[0, 0])  # Repeated in dim 0
    assert torch.all(broadcasted.tensor[0, 5] == data[0, 0])  # Repeated in dim 1
    assert torch.all(broadcasted.tensor[0, 10] == data[0, 0])  # Repeated in dim 1

    # Test 3D asymmetric broadcasting: (2, 3, 4) -> (6, 9, 16) - using factors 3, 3, 4
    batch_shape_3d = (2, 3, 4)
    data_3d = torch.randn((*batch_shape_3d, max_sample_size, *additional_shape), device=device)
    sample_sizes_3d = torch.randint(1, max_sample_size + 1, batch_shape_3d, dtype=torch.int64, device=device)
    ragged_batch_3d = RaggedBatch(data_3d, sample_sizes=sample_sizes_3d)

    new_batch_shape_3d = (6, 9, 16)
    broadcasted_3d = ragged_batch_3d.broadcast_batch_dims_to_shape(new_batch_shape_3d)

    # Check shapes
    assert broadcasted_3d.tensor.shape[:3] == new_batch_shape_3d
    assert broadcasted_3d.mask.shape[:3] == new_batch_shape_3d
    assert broadcasted_3d.sample_sizes.shape == new_batch_shape_3d

    # Check data repetition (multipliers should be [3, 3, 4])
    assert torch.all(broadcasted_3d.tensor[0, 0, 0] == data_3d[0, 0, 0])
    assert torch.all(broadcasted_3d.tensor[2, 0, 0] == data_3d[0, 0, 0])  # Repeated in dim 0
    assert torch.all(broadcasted_3d.tensor[4, 0, 0] == data_3d[0, 0, 0])  # Repeated in dim 0
    assert torch.all(broadcasted_3d.tensor[0, 3, 0] == data_3d[0, 0, 0])  # Repeated in dim 1
    assert torch.all(broadcasted_3d.tensor[0, 6, 0] == data_3d[0, 0, 0])  # Repeated in dim 1
    assert torch.all(broadcasted_3d.tensor[0, 0, 4] == data_3d[0, 0, 0])  # Repeated in dim 2
    assert torch.all(broadcasted_3d.tensor[0, 0, 8] == data_3d[0, 0, 0])  # Repeated in dim 2
    assert torch.all(broadcasted_3d.tensor[0, 0, 12] == data_3d[0, 0, 0])  # Repeated in dim 2

    # Test error cases
    # Wrong number of dimensions
    with pytest.raises(AssertionError):
        ragged_batch.broadcast_batch_dims_to_shape((8, 15, 20))  # 3D instead of 2D

    # Non-divisible shape
    with pytest.raises(AssertionError):
        ragged_batch.broadcast_batch_dims_to_shape((7, 15))  # 7 is not divisible by 2

    # Smaller shape (should also fail as it's not a valid broadcast)
    with pytest.raises(AssertionError):
        ragged_batch.broadcast_batch_dims_to_shape((1, 1))

    # Test that non-uniform dimension and other properties are preserved
    assert broadcasted.non_uniform_dim == ragged_batch.non_uniform_dim
    assert broadcasted.num_batch_dims == ragged_batch.num_batch_dims
    assert broadcasted.tensor.device == ragged_batch.tensor.device
    assert broadcasted.tensor.dtype == ragged_batch.tensor.dtype


if __name__ == "__main__":
    pytest.main([__file__])
