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

"""Batch NVDEC video decoding with one pending prefetch and GPU preprocessing."""

from pathlib import Path

import torch

from .batch_utils import build_img_metas, build_sample, collate_samples, update_metadata_for_transform
from .data_index import DataIndex
from .gpu_image_transforms import GPUImageTransforms
from .multi_video_sampler import MultiVideoSampler


class VideoDataProvider:
    """Load calibrated nuScenes video samples without a CPU image round trip.

    Decoder buffers are borrowed. Float conversion and its stream synchronization
    complete before the next decode can overwrite them. Only one prefetch may be
    outstanding; close() drains it before releasing the decoder.
    """

    def __init__(
        self,
        pkl_path,
        data_root,
        img_norm_cfg,
        ida_aug_conf,
        batch_size=1,
        device='cuda:0',
        video_base_path=None,
        num_cameras=6,
    ):
        if batch_size < 1 or num_cameras < 1:
            raise ValueError('batch_size and num_cameras must be positive')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('Video decoding requires a CUDA device')
        self.device = torch.device(
            'cuda', device.index if device.index is not None else torch.cuda.current_device()
        )
        self.batch_size = batch_size
        self.num_cameras = num_cameras
        self.video_base_path = Path(video_base_path or data_root)
        self.data_index = DataIndex(pkl_path, data_root)
        self.img_norm_cfg = img_norm_cfg
        self.ida_aug_conf = ida_aug_conf
        self.transforms = GPUImageTransforms(img_norm_cfg, ida_aug_conf, str(self.device))
        self._decoder = None
        self._pending = None
        self._sampler = None

    def __len__(self):
        return len(self.data_index)

    def _get_decoder(self):
        if self._decoder is None:
            import accvlab.on_demand_video_decoder as nvc

            self._decoder = nvc.CreateSampleReader(
                num_of_set=1, num_of_file=self.batch_size * self.num_cameras, iGpu=self.device.index
            )
        return self._decoder

    def _collect_batch_info(self, indices):
        metadata, paths, frames = [], [], []
        for index in indices:
            info = self.data_index.get_data_info(index)
            video_paths, frame_ids = info['video_paths'], info['video_frame_ids']
            if (
                video_paths is None
                or frame_ids is None
                or len(video_paths) != self.num_cameras
                or len(frame_ids) != self.num_cameras
                or len(info['img_filename']) != self.num_cameras
            ):
                raise ValueError(f'Sample {info["sample_idx"]} must provide all {self.num_cameras} cameras')
            if any(frame < 0 for frame in frame_ids):
                raise ValueError(f'Sample {info["sample_idx"]} has a negative video frame ID')
            metadata.append(info)
            paths.extend(
                str(path if path.is_absolute() else self.video_base_path / path)
                for path in map(Path, video_paths)
            )
            frames.extend(frame_ids)
        return metadata, paths, frames

    def get_batch(self, indices):
        """Return one fixed-size temporal batch; read/decode failures propagate."""
        if len(indices) != self.batch_size:
            raise ValueError('Sample indices must match the configured batch size')
        if self._pending is not None:
            pending_indices, metadata, paths, frames = self._pending
            if pending_indices != tuple(indices):
                raise ValueError('Requested batch does not match the pending prefetch')
            # A failed get must not be attempted again by close(), masking its error.
            self._pending = None
            decoded = self._get_decoder().DecodeN12ToRGBAsyncGetBuffer(paths, frames, True)
        else:
            metadata, paths, frames = self._collect_batch_info(indices)
            decoded = self._get_decoder().DecodeN12ToRGB(paths, frames, True)
        if len(decoded) != len(paths):
            raise RuntimeError('Decoder returned an incomplete batch')
        images = [torch.as_tensor(frame, device=self.device).to(torch.float32) for frame in decoded]
        torch.cuda.current_stream(self.device).synchronize()
        images = self.transforms(images)
        samples = []
        for slot, info in enumerate(metadata):
            info = update_metadata_for_transform(
                info,
                self.transforms.resize,
                self.transforms.crop_h,
                self.transforms.crop_w,
                self.transforms.final_dim,
            )
            metas = build_img_metas(
                info, self.transforms.get_pad_shape(), self.img_norm_cfg, self.ida_aug_conf, self.num_cameras
            )
            start = slot * self.num_cameras
            samples.append(build_sample(images[start : start + self.num_cameras], info, metas))
        return collate_samples(samples)

    def create_multi_video_sampler(self, batch_size):
        if batch_size != self.batch_size:
            raise ValueError('Sampler and decoder batch sizes must match')
        self._sampler = MultiVideoSampler(self.data_index.video_indices, batch_size)
        return self._sampler

    def get_multi_video_batch(self):
        if self._sampler is None:
            raise RuntimeError('Create the video sampler before requesting batches')
        indices, first = self._sampler.get_batch_indices()
        return self.get_batch(indices), first

    def advance_multi_video_frame(self):
        return self._sampler.advance_frame()

    def prefetch_next_batch(self):
        indices = self._sampler.peek_next_indices()
        if indices is None:
            return
        if self._pending is not None:
            raise RuntimeError('A prefetch is already pending')
        metadata, paths, frames = self._collect_batch_info(indices)
        self._get_decoder().DecodeN12ToRGBAsync(paths, frames, True)
        self._pending = (tuple(indices), metadata, paths, frames)

    def close(self):
        pending, self._pending = self._pending, None
        try:
            if pending is not None:
                _, _, paths, frames = pending
                self._get_decoder().DecodeN12ToRGBAsyncGetBuffer(paths, frames, True)
        finally:
            self._decoder = None
