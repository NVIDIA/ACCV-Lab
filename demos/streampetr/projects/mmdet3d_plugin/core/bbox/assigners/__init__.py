# From StreamPETR (https://github.com/exiawsh/StreamPETR).
# No copyright holder is named upstream; this does not imply the absence of
# copyright.
#
# Changes by NVIDIA: Added batched assigner imports
#                    (HungarianAssigner2DBatched,
#                    HungarianAssigner3DBatched).
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

from .hungarian_assigner_3d import HungarianAssigner3D
from .hungarian_assigner_2d import HungarianAssigner2D
from .hungarian_assigner_2d_batched import HungarianAssigner2DBatched
from .hungarian_assigner_3d_batched import HungarianAssigner3DBatched
__all__ = ['HungarianAssigner3D', 'HungarianAssigner2D', 'HungarianAssigner2DBatched', 'HungarianAssigner3DBatched']