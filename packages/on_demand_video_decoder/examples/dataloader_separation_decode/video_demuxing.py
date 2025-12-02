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
from dataclasses import dataclass
from typing import List, Optional
import torch.cuda.nvtx as nvtx
import numpy as np

import accvlab.on_demand_video_decoder as nvc_ondemand
import video_transforms


@dataclass
class SerializedPacketBundle:
    data: np.ndarray
    first_frame_ids: List[int]
    gop_lens: List[int]
    filepaths: List[str]


class IndexingDemuxerOndemand:
    def __init__(self, batch_size: int, num_cameras: int, use_cache: bool = False) -> None:
        """
        Initialize the IndexingDemuxer.

        Args:
            video_file_path: Path to the video file to demux.
            batch_size: Number of clips.
            num_cameras: Number of cameras.
            use_cache: Whether to use cache.
        """
        print(f"IndexingDemuxerOndemand init !")
        self._batch_size = batch_size
        self._num_cameras = num_cameras
        self._use_cache = use_cache
        self._video_file_paths = [""]
        self._nv_gop_dec = nvc_ondemand.CreateGopDecoder(
            maxfiles=num_cameras,
            iGpu=0,  # use default value, demuxer need no gpu resource
        )
        self._packet_buffers: List[Optional[SerializedPacketBundle]] = [None] * batch_size

    def check_use_cache(self, frame_idx_list: List[int], sample_idx: int) -> bool:
        """
        Check if the cache can be used.

        Args:
            frame_idx_list: List of frame indices to check.
            sample_idx: Index of the clip.
        """
        use_cache = True

        if self._packet_buffers[sample_idx] is None:
            use_cache = False
        else:
            for i in range(len(self._packet_buffers[sample_idx].filepaths)):
                if self._packet_buffers[sample_idx].filepaths[i] != self._video_file_paths[i]:
                    use_cache = False
                    break
                if (
                    frame_idx_list[i] < self._packet_buffers[sample_idx].first_frame_ids[i]
                    or frame_idx_list[i]
                    >= self._packet_buffers[sample_idx].first_frame_ids[i]
                    + self._packet_buffers[sample_idx].gop_lens[i]
                ):
                    use_cache = False
                    break
        return use_cache

    def packet_buffers_for_frame_idx_list(
        self, frame_idx_list: List[int], sample_idx: int = 0
    ) -> video_transforms.PacketOndemandBuffers:
        """
        Fetch packets and dependencies for the given frame indices.

        Args:
            frame_idx: List of frame indices to fetch packets for.
            sample_idx: Index in the batch.

        Returns:
            PacketOndemandBuffers object containing the target frames, packet frames, and packets.
        """
        nvtx.range_push("packet_buffers_for_frame_idx_list")

        if sample_idx >= self._batch_size:
            raise ValueError(
                f"Clip index {sample_idx} is out of range. The number of clips is {self._batch_size}"
            )

        gop_packets = None

        use_cache = False
        if self._use_cache:
            use_cache = self.check_use_cache(frame_idx_list, sample_idx)

        if use_cache == False:
            try:
                nvtx.range_push("get_packets")
                gop_packets, first_frame_ids, gop_lens = self._nv_gop_dec.GetGOP(
                    self._video_file_paths, frame_idx_list
                )
                nvtx.range_pop()  # get_packets

                nvtx.range_push("cache data")
                self._packet_buffers[sample_idx] = SerializedPacketBundle(
                    data=gop_packets,
                    first_frame_ids=first_frame_ids,
                    gop_lens=gop_lens,
                    filepaths=self._video_file_paths,
                )
                nvtx.range_pop()  # get_binary_datas
            except Exception as e:
                print(
                    f"Error fetching packets for frame {frame_idx_list} with video_file_paths: {self._video_file_paths}. Error: {e}"
                )
                exit(1)

        nvtx.range_push("create_PacketOndemandBuffers")
        result = video_transforms.PacketOndemandBuffers(
            gop_packets=gop_packets,
            target_frame_list=frame_idx_list,
            target_file_list=self._video_file_paths,
            use_cache=use_cache,
            sample_idx=sample_idx,
        )
        nvtx.range_pop()  # create_PacketOndemandBuffers
        nvtx.range_pop()  # packet_buffers_for_frame_idx_list
        return result

    def update_path(self, video_file_paths: List[str]):
        """
        Update the video file paths.

        Args:
            video_file_paths: List of video file paths.
        """
        for video_file_path in video_file_paths:
            if not os.path.exists(video_file_path):
                raise FileNotFoundError(f"Video file {video_file_path} does not exist")
        self._video_file_paths = video_file_paths
