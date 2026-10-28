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

import logging
import time
import torch
import psutil
from typing import Optional, Callable, Union, Dict, List, Tuple
from mmcv.runner import get_dist_info
from mmcv.runner.iter_based_runner import IterLoader, IterBasedRunner
from mmcv.runner.builder import RUNNERS


@RUNNERS.register_module()
class DataPipelineRuntimeEvalRunner(IterBasedRunner):
    
    def __init__(self,
                 model: torch.nn.Module,
                 batch_processor: Optional[Callable] = None,
                 optimizer: Union[Dict, torch.optim.Optimizer, None] = None,
                 work_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 meta: Optional[Dict] = None,
                 max_iters: Optional[int] = None,
                 max_epochs: Optional[int] = None,
                 num_warmup_iters = None
                 ) -> None:

        super().__init__(model,
                 batch_processor,
                 optimizer,
                 work_dir,
                 logger,
                 meta,
                 max_iters,
                 max_epochs)
        
        assert num_warmup_iters is not None, "Parameters `num_warmup_iters` must be set"
        self._num_warmup_iters = num_warmup_iters

    def run(self,
        data_loaders,
        workflow: List[Tuple[str, int]],
        max_iters: Optional[int] = None,
        **kwargs) -> None:

        assert len(data_loaders) == 1, f"Only a single data loader is supported, i.e. `len(data_loaders) == 1`; Got `len(data_loaders) == {len(data_loaders)}`."
        assert self._num_warmup_iters < self._max_iters, f"The number of warmup iterations (got: {self._num_warmup_iters}) must be smaller than the total number of iterations (got: {self._max_iters})"

        iter_loader = IterLoader(data_loaders[0])

        start_time = None
        end_time = None
        print("Starting data pipeline runtime evaluation run")
        for i in range(self._max_iters):
            if i == self._num_warmup_iters:
                print("Warmup iterations finished")
                start_time = time.time()
                self._start_first_cpu_usage_measurement()
            data = next(iter_loader)
        end_time = time.time()
        cpu_usage = self._get_cpu_usage_since_last()

        num_non_warmup_iters = self._max_iters - self._num_warmup_iters
        runtime = end_time - start_time
        mean_runtime = runtime / num_non_warmup_iters
        print(f"Runtime for {num_non_warmup_iters} non-warmup iterations: {runtime}")
        print(f"Average runtime: {mean_runtime}")
        print(f"Average CPU usage: {cpu_usage}")
        exit()

    @staticmethod
    def _start_first_cpu_usage_measurement(per_cpu=False):
        psutil.cpu_percent(percpu=per_cpu)

    @staticmethod
    def _get_cpu_usage_since_last(per_cpu=False):
        res = psutil.cpu_percent(percpu=per_cpu)
        return res



