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

import os
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch

import accvlab.on_demand_video_decoder as nvc
from utils import get_data_dir

SAMPLE = os.path.join(get_data_dir(), "sample_clip", "moving_shape_circle_h265.mp4")
H264_SAMPLE = os.path.join(get_data_dir(), "pix_fmt_variants", "h264_avc1_yuv420p.mp4")
OPEN_GOP_SAMPLE = os.path.join(get_data_dir(), "open_gop_variant", "moving_shape_open_gop_h265.mp4")
CONTROLLED_GOP_DIR = os.path.join(get_data_dir(), "selective_gop")
CLOSED_HIER_B7_64F = os.path.join(CONTROLLED_GOP_DIR, "closed_hier_b7_64f.mp4")
CLOSED_P_ONLY_32F = os.path.join(CONTROLLED_GOP_DIR, "closed_p_only_32f.mp4")
SINGLE_GOP_250F = os.path.join(CONTROLLED_GOP_DIR, "single_gop_250f.mp4")
_GCSR_HEADER = struct.Struct("<4sHHII")


def _single_bundle_graph_offset(bundle):
    data = memoryview(np.ascontiguousarray(bundle, dtype=np.uint8)).cast("B")
    if len(data) < 12 or struct.unpack_from("<I", data, 0)[0] != 1:
        raise ValueError("expected a one-frame serialized GOP bundle")
    frame_offset = struct.unpack_from("<Q", data, 4)[0]
    if frame_offset + 28 > len(data):
        raise ValueError("serialized GOP metadata is truncated")
    gop_len = struct.unpack_from("<7i", data, frame_offset)[5]
    offset = frame_offset + 28

    packet_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + packet_count * 4
    decode_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + decode_count * 4
    binary_size = struct.unpack_from("<Q", data, offset)[0]
    graph_offset = offset + 8 + binary_size
    if graph_offset > len(data):
        raise ValueError("serialized GOP packet payload is truncated")
    return data, graph_offset, gop_len


def _dependency_graph(bundle):
    data, graph_offset, gop_len = _single_bundle_graph_offset(bundle)
    if graph_offset == len(data):
        return None
    if graph_offset + _GCSR_HEADER.size > len(data):
        raise ValueError("dependency graph trailer is truncated")
    magic, version, flags, node_count, edge_count = _GCSR_HEADER.unpack_from(data, graph_offset)
    if magic != b"GCSR" or version != 1 or flags != 0 or node_count != gop_len:
        raise ValueError("dependency graph trailer is invalid")

    offset = graph_offset + _GCSR_HEADER.size
    row_offsets = struct.unpack_from(f"<{node_count + 1}I", data, offset)
    offset += (node_count + 1) * 4
    parent_nodes = struct.unpack_from(f"<{edge_count}I", data, offset)
    offset += edge_count * 4
    coded_positions = struct.unpack_from(f"<{node_count}I", data, offset)
    offset += node_count * 4
    if offset != len(data):
        raise ValueError("dependency graph trailer size is invalid")

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "row_offsets": row_offsets,
        "parent_nodes": parent_nodes,
        "coded_positions": coded_positions,
    }


def _dependency_graph_node_count(bundle):
    graph = _dependency_graph(bundle)
    return None if graph is None else graph["node_count"]


def _dependency_closure(graph, target_node):
    needed = {target_node}
    pending = [target_node]
    while pending:
        node = pending.pop()
        begin = graph["row_offsets"][node]
        end = graph["row_offsets"][node + 1]
        for parent in graph["parent_nodes"][begin:end]:
            if parent not in needed:
                needed.add(parent)
                pending.append(parent)
    return needed


def _tensor(frame) -> torch.Tensor:
    value = torch.as_tensor(frame).clone()
    torch.cuda.synchronize()
    return value


def _group_tensors(decoded_groups):
    return [[_tensor(frame) for frame in group["frames"]] for group in decoded_groups]


def _without_dependency_graph(bundle):
    data, graph_offset, _ = _single_bundle_graph_offset(bundle)
    return np.frombuffer(data[:graph_offset], dtype=np.uint8).copy()


def _with_unsupported_dependency_graph_version(bundle):
    data, graph_offset, _ = _single_bundle_graph_offset(bundle)
    corrupted = np.frombuffer(data, dtype=np.uint8).copy()
    assert bytes(corrupted[graph_offset : graph_offset + 4]) == b"GCSR"
    corrupted[graph_offset + 4] = 0xFF
    return corrupted


def test_prototype_and_diagnostic_entry_points_are_not_public():
    decoder = nvc.CreateGopDecoder(1, 0, True)

    assert not hasattr(decoder, "DecodeFromGOPListRGBSelective")
    assert not hasattr(decoder, "DecodeFromGOPListRGBPlanned")
    assert not hasattr(decoder, "GetLastSelectiveDecodeStats")
    assert not hasattr(decoder._decoder, "DecodeFromGOPListRGBSelective")
    assert not hasattr(decoder._decoder, "DecodeFromGOPListRGBPlanned")
    assert not hasattr(decoder._decoder, "GetLastSelectiveDecodeStats")


@pytest.mark.parametrize(
    (
        "video",
        "probe_targets",
        "expected_partitions",
        "closure_target",
        "expected_closure_size",
        "expect_reordered",
    ),
    [
        (CLOSED_HIER_B7_64F, [0, 31, 32, 63], {(0, 32), (32, 32)}, 31, 8, True),
        (CLOSED_P_ONLY_32F, [0, 31], {(0, 32)}, 31, 32, False),
        (SINGLE_GOP_250F, [0, 125, 249], {(0, 250)}, 249, 64, True),
    ],
)
def test_controlled_fixtures_have_expected_gop_dependency_topology(
    video,
    probe_targets,
    expected_partitions,
    closure_target,
    expected_closure_size,
    expect_reordered,
):
    """Each compact fixture protects one distinct topology contract."""
    assert 10_000 <= os.path.getsize(video) <= 50_000

    demuxer = nvc.CreateGopDecoder(1, 0, True)
    partitions = set()
    graphs = {}
    for target in probe_targets:
        ((bundle, first_ids, gop_lens),) = demuxer.GetGOPList(
            [video],
            [target],
            enable_gop_dependency_graph_optimization=True,
        )
        partition = (int(first_ids[0]), int(gop_lens[0]))
        partitions.add(partition)
        graphs.setdefault(partition, _dependency_graph(bundle))

    assert partitions == expected_partitions
    for graph in graphs.values():
        assert graph is not None
        assert graph["edge_count"] < graph["node_count"] ** 2 / 5
        assert len(_dependency_closure(graph, closure_target)) == expected_closure_size
        reordered = graph["coded_positions"] != tuple(range(graph["node_count"]))
        assert reordered is expect_reordered


@pytest.mark.parametrize(
    ("video", "target"),
    [
        (CLOSED_HIER_B7_64F, 31),
        (CLOSED_P_ONLY_32F, 31),
        (SINGLE_GOP_250F, 249),
    ],
)
def test_selective_decode_matches_complete_decode_for_controlled_topologies(video, target):
    demuxer = nvc.CreateGopDecoder(1, 0, True)
    ((bundle, _, _),) = demuxer.GetGOPList(
        [video],
        [target],
        enable_gop_dependency_graph_optimization=True,
    )

    complete_decoder = nvc.CreateGopDecoder(1, 0, True)
    (complete,) = complete_decoder.DecodeFromGOPListRGB(
        [bundle],
        [video],
        [target],
        as_bgr=False,
        enable_gop_dependency_graph_optimization=False,
    )
    selective_decoder = nvc.CreateGopDecoder(1, 0, True)
    (selective,) = selective_decoder.DecodeFromGOPListRGB(
        [bundle],
        [video],
        [target],
        as_bgr=False,
        enable_gop_dependency_graph_optimization=True,
    )

    assert torch.equal(_tensor(complete), _tensor(selective))


def test_selective_decode_matches_complete_decode_for_native_yuv_output():
    target = 31
    demuxer = nvc.CreateGopDecoder(1, 0, True)
    ((bundle, _, _),) = demuxer.GetGOPList(
        [CLOSED_HIER_B7_64F],
        [target],
        enable_gop_dependency_graph_optimization=True,
    )

    complete_decoder = nvc.CreateGopDecoder(1, 0, True)
    (complete,) = complete_decoder.DecodeFromGOPList(
        [bundle],
        [CLOSED_HIER_B7_64F],
        [target],
        enable_gop_dependency_graph_optimization=False,
    )
    selective_decoder = nvc.CreateGopDecoder(1, 0, True)
    (selective,) = selective_decoder.DecodeFromGOPList(
        [bundle],
        [CLOSED_HIER_B7_64F],
        [target],
        enable_gop_dependency_graph_optimization=True,
    )

    complete_planes = [torch.as_tensor(plane).clone() for plane in complete.cuda()]
    selective_planes = [torch.as_tensor(plane).clone() for plane in selective.cuda()]
    torch.cuda.synchronize()
    assert len(complete_planes) == len(selective_planes)
    assert all(
        torch.equal(complete_plane, selective_plane)
        for complete_plane, selective_plane in zip(complete_planes, selective_planes)
    )


def test_get_gop_groups_opt_in_attaches_dependency_graph():
    demuxer = nvc.CreateGopDecoder(1, 0, True)
    groups = demuxer.GetGOPGroups(
        [{"filepath": CLOSED_HIER_B7_64F, "frame_ids": [31, 1, 31]}],
        enable_gop_dependency_graph_optimization=True,
    )

    assert len(groups) == 1
    assert groups[0]["frame_ids"] == [1, 31]
    assert _dependency_graph_node_count(groups[0]["gop_data"]) == 32


def test_group_selective_decode_matches_complete_decode_for_multi_target_union():
    requested_targets = [31, 1, 20, 10, 31]
    demuxer = nvc.CreateGopDecoder(1, 0, True)
    groups = demuxer.GetGOPGroups(
        [{"filepath": CLOSED_HIER_B7_64F, "frame_ids": requested_targets}],
        enable_gop_dependency_graph_optimization=True,
    )

    complete_decoder = nvc.CreateGopDecoder(1, 0, True)
    complete = _group_tensors(complete_decoder.DecodeFromGOPGroupsRGB(groups))
    selective_decoder = nvc.CreateGopDecoder(1, 0, True)
    selective = _group_tensors(
        selective_decoder.DecodeFromGOPGroupsRGB(
            groups,
            enable_gop_dependency_graph_optimization=True,
        )
    )

    assert groups[0]["frame_ids"] == [1, 10, 20, 31]
    assert len(complete) == len(selective) == 1
    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(complete[0], selective[0]))


def test_group_selective_decode_handles_continuation_and_replay():
    targets = [2, 3, 1]
    demuxer = nvc.CreateGopDecoder(1, 0, True)
    decoder = nvc.CreateGopDecoder(1, 0, True)
    actual = []
    expected = []

    for target in targets:
        groups = demuxer.GetGOPGroups(
            [{"filepath": CLOSED_HIER_B7_64F, "frame_ids": [target]}],
            enable_gop_dependency_graph_optimization=True,
        )
        decoded = decoder.DecodeFromGOPGroupsRGB(
            groups,
            enable_gop_dependency_graph_optimization=True,
        )
        actual.append(_tensor(decoded[0]["frames"][0]))

        complete_decoder = nvc.CreateGopDecoder(1, 0, True)
        complete = complete_decoder.DecodeFromGOPGroupsRGB(groups)
        expected.append(_tensor(complete[0]["frames"][0]))

    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(actual, expected))


def test_group_selective_decode_falls_back_per_group_for_mixed_inputs():
    target = 10
    demuxer = nvc.CreateGopDecoder(2, 0, True)
    generated_groups = demuxer.GetGOPGroups(
        [
            {"filepath": SAMPLE, "frame_ids": [target]},
            {"filepath": H264_SAMPLE, "frame_ids": [target]},
        ],
        enable_gop_dependency_graph_optimization=True,
    )

    legacy_hevc_group = dict(generated_groups[0])
    legacy_hevc_group["gop_data"] = _without_dependency_graph(legacy_hevc_group["gop_data"])
    legacy_hevc_group["source_index"] = 2
    groups = [generated_groups[0], legacy_hevc_group, generated_groups[1]]

    assert _dependency_graph_node_count(groups[0]["gop_data"]) is not None
    assert _dependency_graph_node_count(groups[1]["gop_data"]) is None
    assert _dependency_graph_node_count(groups[2]["gop_data"]) is None

    complete_decoder = nvc.CreateGopDecoder(3, 0, True)
    complete = _group_tensors(complete_decoder.DecodeFromGOPGroupsRGB(groups))
    selective_decoder = nvc.CreateGopDecoder(3, 0, True)
    selective = _group_tensors(
        selective_decoder.DecodeFromGOPGroupsRGB(
            groups,
            enable_gop_dependency_graph_optimization=True,
        )
    )

    assert len(complete) == len(selective) == 3
    assert all(
        torch.equal(complete_frame, selective_frame)
        for complete_group, selective_group in zip(complete, selective)
        for complete_frame, selective_frame in zip(complete_group, selective_group)
    )


def test_group_decode_validates_dependency_graph_only_when_enabled():
    target = 10
    demuxer = nvc.CreateGopDecoder(1, 0, True)
    groups = demuxer.GetGOPGroups(
        [{"filepath": SAMPLE, "frame_ids": [target]}],
        enable_gop_dependency_graph_optimization=True,
    )
    corrupted_group = dict(groups[0])
    corrupted_group["gop_data"] = _with_unsupported_dependency_graph_version(corrupted_group["gop_data"])

    complete_decoder = nvc.CreateGopDecoder(1, 0, True)
    complete_decoder.DecodeFromGOPGroupsRGB([corrupted_group])

    selective_decoder = nvc.CreateGopDecoder(1, 0, True)
    with pytest.raises(RuntimeError, match="unsupported dependency CSR trailer"):
        selective_decoder.DecodeFromGOPGroupsRGB(
            [corrupted_group],
            enable_gop_dependency_graph_optimization=True,
        )


def test_opt_in_get_gop_runs_without_a_visible_cuda_device():
    script = f"""
import struct
import numpy as np
import accvlab.on_demand_video_decoder as nvc

decoder = nvc.CreateGopDecoder(1, 0, True)
(bundle, _, _), = decoder.GetGOPList(
    [{SAMPLE!r}],
    [10],
    enable_gop_dependency_graph_optimization=True,
)
data = memoryview(np.ascontiguousarray(bundle, dtype=np.uint8)).cast("B")
frame_offset = struct.unpack_from("<Q", data, 4)[0]
offset = frame_offset + 28
packet_count = struct.unpack_from("<I", data, offset)[0]
offset += 4 + packet_count * 4
decode_count = struct.unpack_from("<I", data, offset)[0]
offset += 4 + decode_count * 4
binary_size = struct.unpack_from("<Q", data, offset)[0]
graph_offset = offset + 8 + binary_size
assert bytes(data[graph_offset:graph_offset + 4]) == b"GCSR"
"""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_get_gop_and_regular_decode_default_to_complete_decode():
    target = 10
    demuxer = nvc.CreateGopDecoder(1, 0, True)

    ((legacy_bundle, _, _),) = demuxer.GetGOPList([SAMPLE], [target])
    assert _dependency_graph_node_count(legacy_bundle) is None

    ((graph_bundle, _, _),) = demuxer.GetGOPList(
        [SAMPLE],
        [target],
        enable_gop_dependency_graph_optimization=True,
    )
    assert _dependency_graph_node_count(graph_bundle) is not None

    decoder = nvc.CreateGopDecoder(1, 0, True)
    (frame,) = decoder.DecodeFromGOPListRGB([graph_bundle], [SAMPLE], [target], as_bgr=False)
    reference_decoder = nvc.CreateGopDecoder(1, 0, True)
    (reference,) = reference_decoder.DecodeN12ToRGB([SAMPLE], [target], False)
    assert torch.equal(_tensor(reference), _tensor(frame))


def test_gop_cache_keeps_graph_generation_mode_consistent():
    target = 10
    demuxer = nvc.CreateGopDecoder(1, 0, True)

    ((legacy_bundle, _, _),) = demuxer.GetGOPList([SAMPLE], [target], useGOPCache=True)
    assert _dependency_graph_node_count(legacy_bundle) is None
    assert demuxer.isCacheHit() == [False]

    ((graph_bundle, first_ids, _),) = demuxer.GetGOPList(
        [SAMPLE],
        [target],
        useGOPCache=True,
        enable_gop_dependency_graph_optimization=True,
    )
    assert _dependency_graph_node_count(graph_bundle) is not None
    assert demuxer.isCacheHit() == [False]

    demuxer.GetGOPList(
        [SAMPLE],
        [first_ids[0]],
        useGOPCache=True,
        enable_gop_dependency_graph_optimization=True,
    )
    assert demuxer.isCacheHit() == [True]


def test_sequential_selective_decode_handles_continuation_and_replay():
    targets = [2, 3, 1]
    demuxer = nvc.CreateGopDecoder(1, 0, True)
    ((bundle, _, _),) = demuxer.GetGOPList(
        [CLOSED_HIER_B7_64F],
        [targets[0]],
        enable_gop_dependency_graph_optimization=True,
    )

    decoder = nvc.CreateGopDecoder(1, 0, True)
    actual = []
    for target in targets:
        (frame,) = decoder.DecodeFromGOPListRGB(
            [bundle],
            [CLOSED_HIER_B7_64F],
            [target],
            as_bgr=False,
            enable_gop_dependency_graph_optimization=True,
        )
        actual.append(_tensor(frame))

    expected = []
    for target in targets:
        complete_decoder = nvc.CreateGopDecoder(1, 0, True)
        (frame,) = complete_decoder.DecodeFromGOPListRGB(
            [bundle],
            [CLOSED_HIER_B7_64F],
            [target],
            as_bgr=False,
            enable_gop_dependency_graph_optimization=False,
        )
        expected.append(_tensor(frame))

    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(actual, expected))


def test_batch_decode_plans_one_stable_closure_union_per_gop():
    requested_targets = [2, 1]
    demuxer = nvc.CreateGopDecoder(1, 0, True)
    ((bundle, _, _),) = demuxer.GetGOPList(
        [CLOSED_HIER_B7_64F],
        [requested_targets[0]],
        enable_gop_dependency_graph_optimization=True,
    )
    assert _dependency_graph_node_count(bundle) is not None

    references = []
    for target in requested_targets:
        reference_decoder = nvc.CreateGopDecoder(1, 0, True)
        (frame,) = reference_decoder.DecodeFromGOPListRGB(
            [bundle],
            [CLOSED_HIER_B7_64F],
            [target],
            as_bgr=False,
            enable_gop_dependency_graph_optimization=False,
        )
        references.append(_tensor(frame))

    batch_decoder = nvc.CreateBatchAsyncGopDecoder(
        maxfiles=1, max_frames_per_decode_call=len(requested_targets), iGpu=0
    )
    batch_decoder.DecodeFromGOPListRGB(
        [[bundle]],
        [CLOSED_HIER_B7_64F],
        [requested_targets],
        as_bgr=False,
        enable_gop_dependency_graph_optimization=True,
    )
    (outputs,) = batch_decoder.DecodeFromGOPListRGBGetBuffer([CLOSED_HIER_B7_64F], [requested_targets], False)

    assert all(torch.equal(reference, _tensor(output)) for reference, output in zip(references, outputs))


def test_batch_decode_validates_graph_only_when_optimization_is_enabled():
    target = 10
    demuxer = nvc.CreateGopDecoder(1, 0, True)
    ((bundle, _, _),) = demuxer.GetGOPList(
        [SAMPLE],
        [target],
        enable_gop_dependency_graph_optimization=True,
    )
    corrupted_bundle = _with_unsupported_dependency_graph_version(bundle)

    reference_decoder = nvc.CreateGopDecoder(1, 0, True)
    (reference,) = reference_decoder.DecodeN12ToRGB([SAMPLE], [target], False)
    batch_decoder = nvc.CreateBatchAsyncGopDecoder(maxfiles=1, max_frames_per_decode_call=1, iGpu=0)
    batch_decoder.DecodeFromGOPListRGB(
        [[corrupted_bundle]],
        [SAMPLE],
        [[target]],
        as_bgr=False,
    )
    (outputs,) = batch_decoder.DecodeFromGOPListRGBGetBuffer([SAMPLE], [[target]], False)

    assert torch.equal(_tensor(reference), _tensor(outputs[0]))

    batch_decoder = nvc.CreateBatchAsyncGopDecoder(maxfiles=1, max_frames_per_decode_call=1, iGpu=0)
    batch_decoder.DecodeFromGOPListRGB(
        [[corrupted_bundle]],
        [SAMPLE],
        [[target]],
        as_bgr=False,
        enable_gop_dependency_graph_optimization=True,
    )
    with pytest.raises(RuntimeError, match="unsupported dependency CSR trailer"):
        batch_decoder.DecodeFromGOPListRGBGetBuffer([SAMPLE], [[target]], False)


def test_graphless_legacy_hevc_and_h264_fall_back_to_complete_decode():
    target = 10
    demuxer = nvc.CreateGopDecoder(1, 0, True)
    ((new_bundle, _, _),) = demuxer.GetGOPList(
        [SAMPLE],
        [target],
        enable_gop_dependency_graph_optimization=True,
    )
    legacy_hevc_bundle = _without_dependency_graph(new_bundle)
    ((h264_bundle, _, _),) = demuxer.GetGOPList(
        [H264_SAMPLE],
        [target],
        enable_gop_dependency_graph_optimization=True,
    )

    for video, bundle in (
        (SAMPLE, legacy_hevc_bundle),
        (H264_SAMPLE, h264_bundle),
    ):
        assert _dependency_graph_node_count(bundle) is None
        assert _single_bundle_graph_offset(bundle)[1] == bundle.nbytes

        source_decoder = nvc.CreateGopDecoder(1, 0, True)
        (reference,) = source_decoder.DecodeN12ToRGB([video], [target], False)
        fallback_decoder = nvc.CreateGopDecoder(1, 0, True)
        (fallback,) = fallback_decoder.DecodeFromGOPListRGB(
            [bundle],
            [video],
            [target],
            as_bgr=False,
            enable_gop_dependency_graph_optimization=True,
        )

        assert torch.equal(_tensor(reference), _tensor(fallback))


def test_mixed_graph_and_graphless_inputs_decode_in_regular_and_async_batch_apis():
    target = 10
    demuxer = nvc.CreateGopDecoder(2, 0, True)
    hevc_result, h264_result = demuxer.GetGOPList(
        [SAMPLE, H264_SAMPLE],
        [target, target],
        enable_gop_dependency_graph_optimization=True,
    )
    hevc_bundle = hevc_result[0]
    h264_bundle = h264_result[0]
    assert _dependency_graph_node_count(hevc_bundle) is not None
    assert _dependency_graph_node_count(h264_bundle) is None

    reference_decoder = nvc.CreateGopDecoder(2, 0, True)
    references = reference_decoder._decoder.DecodeFromGOPListRGB(
        [_without_dependency_graph(hevc_bundle), h264_bundle],
        [SAMPLE, H264_SAMPLE],
        [target, target],
        False,
    )
    reference_tensors = [_tensor(reference) for reference in references]

    automatic_decoder = nvc.CreateGopDecoder(2, 0, True)
    regular_outputs = automatic_decoder.DecodeFromGOPListRGB(
        [hevc_bundle, h264_bundle],
        [SAMPLE, H264_SAMPLE],
        [target, target],
        as_bgr=False,
        enable_gop_dependency_graph_optimization=True,
    )

    assert all(
        torch.equal(reference, _tensor(output))
        for reference, output in zip(reference_tensors, regular_outputs)
    )

    async_decoder = nvc.CreateBatchAsyncGopDecoder(
        maxfiles=2,
        max_frames_per_decode_call=1,
        iGpu=0,
    )
    async_decoder.DecodeFromGOPListRGB(
        [[hevc_bundle], [h264_bundle]],
        [SAMPLE, H264_SAMPLE],
        [[target], [target]],
        as_bgr=False,
        enable_gop_dependency_graph_optimization=True,
    )
    async_outputs = async_decoder.DecodeFromGOPListRGBGetBuffer(
        [SAMPLE, H264_SAMPLE],
        [[target], [target]],
        False,
    )

    assert all(
        torch.equal(reference, _tensor(outputs[0]))
        for reference, outputs in zip(reference_tensors, async_outputs)
    )


def test_open_gop_dependency_optimization_matches_complete_decode_at_boundaries():
    """Open GOP is supported when every reference resolves inside its GOP bundle."""
    boundary_targets = [18, 19, 20, 39, 40, 59, 60, 77, 78, 79, 80, 99]

    for target in boundary_targets:
        demuxer = nvc.CreateGopDecoder(1, 0, True)
        ((bundle, _, _),) = demuxer.GetGOPList(
            [OPEN_GOP_SAMPLE],
            [target],
            enable_gop_dependency_graph_optimization=True,
        )
        assert _dependency_graph_node_count(bundle) == 20

        complete_decoder = nvc.CreateGopDecoder(1, 0, True)
        (complete,) = complete_decoder.DecodeFromGOPListRGB(
            [bundle],
            [OPEN_GOP_SAMPLE],
            [target],
            as_bgr=False,
            enable_gop_dependency_graph_optimization=False,
        )
        selective_decoder = nvc.CreateGopDecoder(1, 0, True)
        (selective,) = selective_decoder.DecodeFromGOPListRGB(
            [bundle],
            [OPEN_GOP_SAMPLE],
            [target],
            as_bgr=False,
            enable_gop_dependency_graph_optimization=True,
        )
        assert torch.equal(_tensor(complete), _tensor(selective))


def test_complete_decode_after_optimized_call_flushes_callback_state():
    first_target, second_target = 0, 10
    demuxer = nvc.CreateGopDecoder(1, 0, True)
    ((bundle, _, _),) = demuxer.GetGOPList(
        [SAMPLE],
        [first_target],
        enable_gop_dependency_graph_optimization=True,
    )

    decoder = nvc.CreateGopDecoder(1, 0, True)
    decoder.DecodeFromGOPListRGB(
        [bundle],
        [SAMPLE],
        [first_target],
        as_bgr=False,
        enable_gop_dependency_graph_optimization=True,
    )
    (legacy_after_selective,) = decoder.DecodeFromGOPListRGB(
        [_without_dependency_graph(bundle)], [SAMPLE], [second_target], False
    )

    reference_decoder = nvc.CreateGopDecoder(1, 0, True)
    (reference,) = reference_decoder.DecodeFromGOPListRGB(
        [_without_dependency_graph(bundle)], [SAMPLE], [second_target], False
    )
    assert torch.equal(_tensor(reference), _tensor(legacy_after_selective))
