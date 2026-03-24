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


import numpy as np
import pytest
import torch


def _storage_data_ptr(t: torch.Tensor) -> int:
    # Prefer untyped storage (newer PyTorch); fall back for older versions.
    if hasattr(t, "untyped_storage"):
        return int(t.untyped_storage().data_ptr())
    return int(t.storage().data_ptr())  # type: ignore[attr-defined]


def _round_up(x: int, a: int) -> int:
    if a <= 1:
        return x
    rem = x % a
    return x if rem == 0 else (x + (a - rem))


def test_multi_tensor_copier_nested_structure_and_values():
    import accvlab.multi_tensor_copier as mtc

    device = torch.device("cuda:0")

    # Nested structure with tuples/lists (output preserves tuples).
    data = [
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
        (
            torch.ones((2, 3), dtype=torch.float16) * 7,
            [
                torch.zeros((1,), dtype=torch.int64),
                (torch.randn((5,), dtype=torch.float32),),
            ],
        ),
    ]

    # background-thread path (returns earlier; staging/submission happens off-thread).
    h = mtc.start_copy(data, device, use_pinned_staging=True)
    out = h.get()

    assert isinstance(out, list)
    assert isinstance(out[0], torch.Tensor)
    assert isinstance(out[1], tuple)

    leaves_in = [data[0], data[1][0], data[1][1][0], data[1][1][1][0]]
    leaves_out = [out[0], out[1][0], out[1][1][0], out[1][1][1][0]]

    for a, b in zip(leaves_in, leaves_out):
        assert b.device == device
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        torch.testing.assert_close(a, b.cpu())

    assert h.ready() is True


def test_multi_tensor_copier_numpy_to_cpu_tensors_only():
    # numpy.ndarray leaves should be accepted and converted to CPU torch.Tensors during traversal.
    import accvlab.multi_tensor_copier as mtc

    data = [
        np.arange(12, dtype=np.float32).reshape(3, 4),
        (np.ones((2, 3), dtype=np.float32), [np.zeros((1,), dtype=np.int64)]),
    ]

    h = mtc.start_copy(data, "cpu", pack_cpu_tensors=False)
    out = h.get()

    assert isinstance(out, list)
    assert isinstance(out[0], torch.Tensor)
    assert out[0].device.type == "cpu"
    torch.testing.assert_close(out[0], torch.from_numpy(data[0]))

    assert isinstance(out[1], tuple)
    assert isinstance(out[1][0], torch.Tensor)
    torch.testing.assert_close(out[1][0], torch.from_numpy(data[1][0]))

    assert isinstance(out[1][1], list)
    assert isinstance(out[1][1][0], torch.Tensor)
    torch.testing.assert_close(out[1][1][0], torch.from_numpy(data[1][1][0]))


def test_multi_tensor_copier_dict_and_passthrough_leaves():
    import accvlab.multi_tensor_copier as mtc

    device = torch.device("cuda:0")
    marker = object()

    data = {
        "a": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "b": (np.ones((2,), dtype=np.float32), {"meta": marker}),
    }

    h = mtc.start_copy(data, device, use_pinned_staging=True)
    out = h.get()

    assert isinstance(out, dict)
    assert set(out.keys()) == {"a", "b"}
    assert isinstance(out["b"], tuple)
    assert isinstance(out["b"][1], dict)
    assert out["b"][1]["meta"] is marker  # passthrough identity preserved

    assert isinstance(out["a"], torch.Tensor)
    assert out["a"].device == device
    torch.testing.assert_close(out["a"].cpu(), data["a"])

    assert isinstance(out["b"][0], torch.Tensor)
    assert out["b"][0].device == device
    torch.testing.assert_close(out["b"][0].cpu(), torch.from_numpy(data["b"][0]))


def test_multi_tensor_copier_root_leaf_behavior():
    import accvlab.multi_tensor_copier as mtc

    device = torch.device("cuda:0")

    t = torch.arange(5, dtype=torch.int64)
    h = mtc.start_copy(t, device)
    out = h.get()
    assert isinstance(out, torch.Tensor)
    assert out.device == device
    torch.testing.assert_close(out.cpu(), t)

    marker = object()
    h2 = mtc.start_copy(marker, device)
    out2 = h2.get()
    assert out2 is marker


@pytest.mark.parametrize("use_pinned_staging", [True, False])
def test_multi_tensor_copier_pack_cpu_tensors(use_pinned_staging: bool):
    import accvlab.multi_tensor_copier as mtc

    device = torch.device("cuda:0")

    # Many small CPU tensors of the same dtype -> should trigger packing when enabled.
    data = [
        [torch.arange(64, dtype=torch.float32) + i for i in range(50)],
        [torch.ones((16,), dtype=torch.float32) * 3.0, torch.zeros((8,), dtype=torch.float32)],
    ]

    h = mtc.start_copy(
        data,
        device,
        use_pinned_staging=use_pinned_staging,
        pack_cpu_tensors=True,
    )

    out = h.get()
    assert isinstance(out, list)
    assert isinstance(out[0], list)
    assert isinstance(out[1], list)

    for i in range(50):
        torch.testing.assert_close(out[0][i].cpu(), data[0][i])
        assert out[0][i].device == device

    for i in range(len(data[1])):
        torch.testing.assert_close(out[1][i].cpu(), data[1][i])
        assert out[1][i].device == device


@pytest.mark.parametrize("min_packed_alignment_bytes", [1, 16, 6])
@pytest.mark.parametrize("use_pinned_staging", [True, False])
def test_multi_tensor_copier_pack_cpu_tensors_mixed_dtypes_and_alignment(
    min_packed_alignment_bytes: int, use_pinned_staging: bool
):
    import accvlab.multi_tensor_copier as mtc

    device = torch.device("cuda:0")

    # Mixed dtypes (incl. complex). Include one non-contiguous tensor to ensure it falls back.
    a_f32 = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    a_i64 = torch.arange(17, dtype=torch.int64)
    a_f16 = (torch.arange(11, dtype=torch.float16) + 1).reshape(-1)
    a_c64 = (torch.arange(9, dtype=torch.float32) + 1j * torch.arange(9, dtype=torch.float32)).to(
        torch.complex64
    )
    a_c128 = (torch.arange(5, dtype=torch.float64) + 1j * torch.arange(5, dtype=torch.float64)).to(
        torch.complex128
    )
    a_noncontig = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()  # non-contiguous

    data = [a_f32, [a_i64, (a_f16, [a_c64, a_c128, a_noncontig])]]

    h = mtc.start_copy(
        data,
        device,
        use_pinned_staging=use_pinned_staging,
        pack_cpu_tensors=True,
        min_packed_alignment_bytes=min_packed_alignment_bytes,
    )
    out = h.get()

    leaves_in = [a_f32, a_i64, a_f16, a_c64, a_c128, a_noncontig]
    leaves_out = [out[0], out[1][0], out[1][1][0], out[1][1][1][0], out[1][1][1][1], out[1][1][1][2]]

    for a, b in zip(leaves_in, leaves_out):
        assert b.device == device
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        torch.testing.assert_close(a, b.cpu())

    # Alignment check for the packed buffer views (if packing happened):
    # group outputs by shared storage base pointer and validate offsets in the largest group.
    storage_ptrs = [_storage_data_ptr(t) for t in leaves_out]
    base_ptr = max(set(storage_ptrs), key=storage_ptrs.count)
    packed_out = [t for t in leaves_out if _storage_data_ptr(t) == base_ptr]
    assert len(packed_out) >= 2  # packing should trigger for this case
    for t in packed_out:
        elem_sz = int(t.element_size())
        required_align = max(int(min_packed_alignment_bytes), elem_sz)
        required_align = _round_up(required_align, elem_sz)
        byte_off = int(t.data_ptr()) - int(base_ptr)
        assert byte_off % required_align == 0


@pytest.mark.parametrize("use_pinned_staging", [True, False])
def test_multi_tensor_copier_gpu_to_cpu(use_pinned_staging: bool):
    import accvlab.multi_tensor_copier as mtc

    device = torch.device("cuda:0")

    data_cpu = [
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
        (
            torch.ones((2, 3), dtype=torch.float16) * 7,
            [
                torch.zeros((1,), dtype=torch.int64),
                (torch.randn((5,), dtype=torch.float32),),
            ],
        ),
    ]
    data_gpu = [
        data_cpu[0].to(device),
        (
            data_cpu[1][0].to(device),
            [
                data_cpu[1][1][0].to(device),
                (data_cpu[1][1][1][0].to(device),),
            ],
        ),
    ]

    h = mtc.start_copy(data_gpu, "cpu", use_pinned_staging=use_pinned_staging)
    out = h.get()

    assert isinstance(out, list)
    assert isinstance(out[0], torch.Tensor)
    assert isinstance(out[1], tuple)

    leaves_ref = [data_cpu[0], data_cpu[1][0], data_cpu[1][1][0], data_cpu[1][1][1][0]]
    leaves_out = [out[0], out[1][0], out[1][1][0], out[1][1][1][0]]

    for ref, result in zip(leaves_ref, leaves_out):
        assert result.device.type == "cpu"
        assert ref.shape == result.shape
        assert ref.dtype == result.dtype
        torch.testing.assert_close(ref, result)

    if use_pinned_staging:
        for result in leaves_out:
            assert result.is_pinned()

    assert h.ready() is True


@pytest.mark.parametrize("use_pinned_staging", [True, False])
def test_multi_tensor_copier_gpu_to_cpu_many_small_tensors(use_pinned_staging: bool):
    """D2H with many small tensors."""
    import accvlab.multi_tensor_copier as mtc

    device = torch.device("cuda:0")

    data_gpu = [torch.arange(64, dtype=torch.float32, device=device) + i for i in range(50)]

    h = mtc.start_copy(data_gpu, "cpu", use_pinned_staging=use_pinned_staging)
    out = h.get()

    assert isinstance(out, list)
    assert len(out) == 50

    for i in range(50):
        expected = torch.arange(64, dtype=torch.float32) + i
        assert out[i].device.type == "cpu"
        torch.testing.assert_close(out[i], expected)


@pytest.mark.parametrize("target_device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_pinned_staging", [True, False])
@pytest.mark.parametrize("pack_cpu_tensors", [True, False])
def test_multi_tensor_copier_mixed_devices(
    target_device: str, use_pinned_staging: bool, pack_cpu_tensors: bool
):
    """Mixed CPU + GPU inputs copied to a GPU or CPU target.

    Verifies that CPU and GPU tensors are handled correctly regardless of target device:
    tensors already on the target are reused as-is, others are copied.
    """
    import accvlab.multi_tensor_copier as mtc

    gpu = torch.device("cuda:0")
    target = torch.device(target_device)

    cpu_a = torch.arange(32, dtype=torch.float32)
    cpu_b = torch.ones((4, 3), dtype=torch.float16)
    gpu_c = torch.randn(5, dtype=torch.float32, device=gpu) + 10
    gpu_d = torch.arange(8, dtype=torch.int64, device=gpu)

    data = [cpu_a, {"gpu": gpu_c, "cpu": cpu_b}, (gpu_d,)]

    h = mtc.start_copy(
        data,
        target,
        use_pinned_staging=use_pinned_staging,
        pack_cpu_tensors=pack_cpu_tensors,
    )
    out = h.get()

    assert isinstance(out, list)
    assert isinstance(out[1], dict)
    assert isinstance(out[2], tuple)

    leaves_in = [cpu_a, cpu_b, gpu_c, gpu_d]
    leaves_out = [out[0], out[1]["cpu"], out[1]["gpu"], out[2][0]]

    for ref, result in zip(leaves_in, leaves_out):
        assert result.device == target
        assert result.shape == ref.shape
        assert result.dtype == ref.dtype
        torch.testing.assert_close(result.cpu(), ref.cpu())

    # Tensors already on the target device should be reused (same storage).
    for ref, result in zip(leaves_in, leaves_out):
        if ref.device == target:
            assert _storage_data_ptr(result) == _storage_data_ptr(ref)

    # D2H with pinned staging: GPU-origin outputs land in pinned memory.
    if target.type == "cpu" and use_pinned_staging:
        for ref, result in zip(leaves_in, leaves_out):
            if ref.device.type == "cuda":
                assert result.is_pinned()


@pytest.mark.parametrize("use_pinned_staging", [True, False])
def test_multi_tensor_copier_pack_chunked(use_pinned_staging: bool):
    """Packing with a tiny chunk size forces multiple chunks; verify correctness and distinct storages."""
    import accvlab.multi_tensor_copier as mtc

    device = torch.device("cuda:0")

    data = [torch.arange(64, dtype=torch.float32) + i for i in range(20)]
    total_bytes = sum(t.numel() * t.element_size() for t in data)
    chunk_limit = 512
    assert total_bytes > chunk_limit, "test expects data to exceed the chunk limit"

    h = mtc.start_copy(
        data,
        device,
        use_pinned_staging=use_pinned_staging,
        pack_cpu_tensors=True,
        max_packed_chunk_bytes=chunk_limit,
    )
    out = h.get()

    assert isinstance(out, list)
    assert len(out) == len(data)

    for i, (ref, result) in enumerate(zip(data, out)):
        assert result.device == device, f"output {i} on wrong device"
        assert result.shape == ref.shape
        assert result.dtype == ref.dtype
        torch.testing.assert_close(result.cpu(), ref)

    # Verify that multiple GPU storage bases were used (multiple chunks).
    base_ptrs = {_storage_data_ptr(t) for t in out}
    assert len(base_ptrs) >= 2, (
        f"expected multiple GPU storage chunks but got {len(base_ptrs)}; "
        "chunking may not have been applied"
    )


if __name__ == "__main__":
    pytest.main([__file__])
