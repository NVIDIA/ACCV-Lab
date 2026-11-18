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

import pytest

from accvlab.optim_test_tools import Stopwatch
from accvlab.optim_test_tools.singleton_base import SingletonBase


@pytest.fixture
def fresh_stopwatch():
    # The stopwatch is a singleton which can be enabled only once. Use a new instance for each test.
    SingletonBase._instances.pop(Stopwatch, None)
    yield
    SingletonBase._instances.pop(Stopwatch, None)


def test_print_with_measurement_only_seen_during_warmup(fresh_stopwatch, capsys):
    stopwatch = Stopwatch(num_warmup_iters=2, print_every_n_iters=2, do_cuda_sync=False)
    for i in range(6):
        stopwatch.start_meas("step")
        stopwatch.end_meas("step")
        # Performed only in some iterations; the first time during the warm-up phase.
        if i % 5 == 0:
            stopwatch.start_meas("periodic")
            stopwatch.end_meas("periodic")
        stopwatch.finish_iter()
    out = capsys.readouterr().out
    assert "`periodic`: not measured yet" in out
    assert stopwatch.get_num_nonwarmup_iters_measured() == 4


def test_print_with_cpu_usage_measurement_not_performed_yet(fresh_stopwatch, capsys):
    stopwatch = Stopwatch(num_warmup_iters=0, print_every_n_iters=None, do_cuda_sync=False)
    stopwatch.set_cpu_usage_meas_name("loop")
    stopwatch.start_meas("step")
    stopwatch.end_meas("step")
    stopwatch.finish_iter()
    stopwatch.print_eval_times()
    out = capsys.readouterr().out
    assert "`loop`: not measured yet" in out
    assert "Mean CPU usage during `loop`: not measured yet" in out
