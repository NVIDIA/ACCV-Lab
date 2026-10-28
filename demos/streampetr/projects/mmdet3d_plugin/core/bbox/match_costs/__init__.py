# From DETR3D (https://github.com/WangYueFt/detr3d).
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Unchanged copy in StreamPETR (https://github.com/exiawsh/StreamPETR).
# ------------------------------------------------------------------------
# Changes by NVIDIA: Added batched cost imports (BBoxL1CostBatched,
#                    FocalLossCostBatched, IoUCostBatched).
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from mmdet.core.bbox.match_costs import build_match_cost
from .match_cost import BBox3DL1Cost
from .match_cost_batched import BBoxL1CostBatched, FocalLossCostBatched, IoUCostBatched

__all__ = ['build_match_cost', 'BBox3DL1Cost', 'BBoxL1CostBatched', 'FocalLossCostBatched', 'IoUCostBatched']