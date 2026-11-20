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

"""Keep one scene per batch slot, padding finished slots without repeating labels."""


class MultiVideoSampler:
    def __init__(self, video_indices, batch_size):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if not video_indices or any(not indices for indices in video_indices.values()):
            raise ValueError('Each video must contain at least one frame')
        self.batch_size = batch_size
        self.videos = list(video_indices.items())
        self.num_videos = len(self.videos)
        self.total_rounds = (self.num_videos + batch_size - 1) // batch_size
        self.set_round(0)

    def set_round(self, round_idx):
        if not 0 <= round_idx < self.total_rounds:
            raise ValueError('round_idx is outside the available rounds')
        start = round_idx * self.batch_size
        count = min(self.batch_size, self.num_videos - start)
        self.batch_assignments = [start + slot % count for slot in range(self.batch_size)]
        self.active_mask = [slot < count for slot in range(self.batch_size)]
        self.frame_idx = 0

    def get_video_length(self, batch_pos):
        return len(self.videos[self.batch_assignments[batch_pos]][1])

    def get_max_frames_in_round(self):
        return max(self.get_video_length(slot) for slot, active in enumerate(self.active_mask) if active)

    def _indices_at(self, frame_idx):
        return [
            self.videos[video][1][min(frame_idx, len(self.videos[video][1]) - 1)]
            for video in self.batch_assignments
        ]

    def get_batch_indices(self):
        return self._indices_at(self.frame_idx), [self.frame_idx == 0] * self.batch_size

    def advance_frame(self):
        self.frame_idx += 1
        return self.frame_idx < self.get_max_frames_in_round()

    def peek_next_indices(self):
        if self.frame_idx + 1 >= self.get_max_frames_in_round():
            return None
        return self._indices_at(self.frame_idx + 1)
