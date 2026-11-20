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

import time
from typing import Optional

if __name__ != "__main__":
    from .singleton_base import SingletonBase
else:
    from singleton_base import SingletonBase


class Stopwatch(SingletonBase):
    """Stopwatch for performing runtime evaluations.

    This is a singleton class for performing runtime measurements and obtaining the total as well as average
    run-time for the measurements

      1. per measurement.
      2. per (training) iteration, where a single measurement can be performed multiple times, or only in some
         iterations.

    A warm-up phase can be defined when configuring the stopwatch. During the warm-up phase, no measurements
    are performed.

    Multiple measurements can be performed, and are distinguished by name. The end of a (training) iteration is
    indicated explicitly (by calling the method :meth:`finish_iter`). This is done to automatically start
    measurements after the warm-up phase is finished, to average the measurements over the iterations, and
    to print the measurements in certain intervals.

    The CPU usage can be measured for one "type" of measurement (i.e. one measurement name). This is done by
    calling the :meth:`set_cpu_usage_meas_name` before the first measurement with the corresponding name is
    started. The CPU usage is then measured whenever the measurement is running and the average CPU usage is
    printed together with the other measurements.

    One-time measurements can be performed at any point in the code (see :meth:`start_one_time_measurement`,
    :meth:`end_one_time_measurement`). They are not affected by the warm-up phase and are reported as such
    (i.e. in own section and without averages etc.). Each one-time measurement (i.e. measurement with a given
    name) can be performed only once.

    Warning:

        The CPU usage is measured using :func:`psutil.cpu_percent`. To ensure that the interval for which the
        CPU usage is measured is correct, the function :func:`psutil.cpu_percent` must not be called outside
        of the stopwatch during the measurement.

    The stopwatch must be first enabled before any measurements are performed. If not enabled, calls to any
    methods have minimal overhead. To ensure this, the methods are empty in disabled state and replaced with
    methods implementing the functionality when enabled. This means that the runtime overhead for using the
    stopwatch is negligible when it is not enabled. The enabling can be done from any part of the code (as
    this is a singleton).

    """

    class _TimeAccumulator:
        def __init__(self):
            self._start_time = 0
            self._accum_time = 0
            self._num_meas = 0
            self._running = False

        def start(self, now):
            assert not self._running, "Trying to start a time emasurement which is already running"
            self._start_time = now
            self._running = True

        def end(self, now):
            assert self._running, "Trying to end a time measurement which was not started"
            self._accum_time += now - self._start_time
            self._num_meas += 1
            self._running = False

        def get_accum_time(self):
            assert not self._running, "Calling `get_accum_time()` not supported while measurement is running"
            return self._accum_time

        def get_num_meas(self):
            assert not self._running, "Calling `get_num_meas()` not supported while measurement is running"
            return self._num_meas

        def is_running(self):
            return self._running

    class _TimeAndCPUUsageAccumulator(_TimeAccumulator):
        def __init__(self):
            super().__init__()
            from psutil import cpu_percent

            self._cpu_percent = cpu_percent
            self._cpu_usage_times_time_accum = 0
            self._cpu_start_time = None

        def start(self, now):
            super().start(now)
            self._cpu_start_time = now

        def end(self, now):
            super().end(now)
            cpu_interval = now - self._cpu_start_time
            cpu_usage = self._cpu_percent()
            self._cpu_usage_times_time_accum += cpu_interval * cpu_usage

        def get_mean_cpu_usage(self):
            res = self._cpu_usage_times_time_accum / self.get_accum_time()
            return res

    def __init__(self, *args, **kwargs):
        """

        Note:
            When obtaining an object using (``Stopwatch()``) the singleton is returned if already
            created.

            If parameters are provided when calling ``Stopwatch()``, this will enable the stopwatch
            (equivalent to calling :meth:`enable`). Note that enabling can only be done once, and will
            lead to an error if attempted a second time.

        Args:
            num_warmup_iters: The number of warmup iterations to be performed before the runtime
                measurement is started.
            print_every_n_iters: Once in how many iterations to print the measured runtime. If
                ``None``, the runtime is not printed automatically (but can still be printed manually
                by calling :meth:`print_eval_times`).
            do_cuda_sync: Whether to synchronize the CUDA device every time a measurement is started
                or stopped.
        """
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._enabled = False
            self._SET_DOCSTRINGS_OF_ENABLED_METHOD_VARIANTS()
        if len(args) > 0 or len(kwargs) > 0:
            self.enable(*args, **kwargs)

    def enable(self, num_warmup_iters: int, print_every_n_iters: Optional[int], do_cuda_sync: bool):
        """Enable the stopwatch

        This method can be called only once and enables the Stopwatch singleton.
        Any measurements started or performed before calling this method are ignored.

        Args:
            num_warmup_iters: The number of warmup iterations to be performed before the runtime measurement is started
            print_every_n_iters: Once in how many iterations to print the measured runtime. If ``None``, the runtime is not
                printed automatically (but can be still printed manually by calling :meth:`print_eval_times`).
            do_cuda_sync: Whether to synchronize CUDA device every time a measurement is started or stopped.
        """
        assert not self._enabled, (
            "Stopwatch singleton can be enabled only once (either by passing arguments when it "
            "is obtained, or by calling :meth:`enable` directly)"
        )
        self._enabled = True
        self._num_warmup_iters = num_warmup_iters
        self._print_every_n_iters = print_every_n_iters
        self._times = dict()
        self._do_cuda_sync = do_cuda_sync
        self._num_iters_finished = 0
        self._one_time_measurements = dict()
        # Warmup is only finished already if there is no warmup (i.e. 0 warmup iterations)
        self._warmup_finished = num_warmup_iters == 0
        if self._do_cuda_sync:
            import torch

            self._torch = torch

        # Name of the measurement for which CPU usage is measured.
        # If None, no CPU usage measurements are performed.
        # This is set by the method `set_cpu_usage_meas_name`.
        self._cpu_usage_meas_name = None

        # Set the methods
        self.set_cpu_usage_meas_name = self._set_cpu_usage_meas_name_enabled
        self.start_meas = self._start_meas_enabled
        self.end_meas = self._end_meas_enabled
        self.start_one_time_measurement = self._start_one_time_measurement_enabled
        self.end_one_time_measurement = self._end_one_time_measurement_enabled
        self.print_eval_times = self._print_eval_times_enabled
        self.finish_iter = self._finish_iter_enabled
        self.get_num_nonwarmup_iters_measured = self._get_num_nonwarmup_iters_measured_enabled

    @property
    def is_enabled(self) -> bool:
        '''Whether the stopwatch is enabled'''
        return self._enabled

    def print_eval_times(self):
        """Print the evaluation times"""
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def set_cpu_usage_meas_name(self, name: str):
        """Set the name of the CPU usage measurement

        This method must be called before the first measurement with the corresponding name is started.

        Important:
            If the CPU usage measurement is already set, it cannot be changed. Calling this method with
            a different name will raise an error.

            Calling this method with the same name as the current CPU usage measurement name is allowed
            and will have no effect. This is useful to set the CPU usage name right before starting
            the first measurement with the corresponding name, even if the corresponding code
            region is called iteratively.

        Warning:
            The CPU usage is measured using :func:`psutil.cpu_percent`. To ensure that the interval for which the CPU usage is measured
            is correct, the function :func:`psutil.cpu_percent` must not be called outside of the stopwatch during the measurement.


        Args:
            name: Name of the measurement
        """
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def start_meas(self, name: str):
        """Start a measurement with given name.

        Args:
            name: Name for the measurement
        """
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def end_meas(self, name: str):
        """End a measurement with given name.

        Args:
            name: Name of the measurements

        """
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def start_one_time_measurement(self, name: str):
        """Start a one-time measurement with given name.

        Args:
            name: Name of the measurement
        """
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def end_one_time_measurement(self, name: str):
        """End a one-time measurement with given name.

        Args:
            name: Name of the measurement
        """
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def finish_iter(self):
        """Finish the current iteration."""
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def get_num_nonwarmup_iters_measured(self) -> int:
        """Get the number of non-warmup iterations performed.

        Returns:
            Number of measured non-warmup iterations
        """
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        return 0

    @classmethod
    def _SET_DOCSTRINGS_OF_ENABLED_METHOD_VARIANTS(cls):
        """Set the docstrings of the enabled method variants.

        This is done to ensure that the correct docstring is present in the methods
        once the Stopwatch is enabled, and the original (disabled) methods are replaced by the
        enabled variants.
        """
        cls._print_eval_times_enabled.__doc__ = cls.print_eval_times.__doc__
        cls._set_cpu_usage_meas_name_enabled.__doc__ = cls.set_cpu_usage_meas_name.__doc__
        cls._start_meas_enabled.__doc__ = cls.start_meas.__doc__
        cls._end_meas_enabled.__doc__ = cls.end_meas.__doc__
        cls._start_one_time_measurement_enabled.__doc__ = cls.start_one_time_measurement.__doc__
        cls._end_one_time_measurement_enabled.__doc__ = cls.end_one_time_measurement.__doc__
        cls._finish_iter_enabled.__doc__ = cls.finish_iter.__doc__
        cls._get_num_nonwarmup_iters_measured_enabled.__doc__ = cls.get_num_nonwarmup_iters_measured.__doc__

    def _print_eval_times_enabled(self):
        """TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        """
        print("#" * 25 + " Stopwatch " + "#" * 25)
        if not self._warmup_finished:
            print("~" * 50)
            print(
                f"warmup not finished ({self._num_iters_finished} out of {self._num_warmup_iters} warmup iterations finished)"
            )
        elif self._num_iters_finished == self._num_warmup_iters:
            print("~" * 50)
            print(
                f"warmup finished ({self._num_warmup_iters} iterations), but no iterations were performed after that"
            )
        else:
            num_measured_iters = self._num_iters_finished - self._num_warmup_iters
            print("~" * 50)
            print(f"Num. measured iterations: {num_measured_iters}")
            # Mean time per iteration
            print(". " * 25)
            print(f"Average runtime per iteration:")
            for time_name, time in self._times.items():
                print(f"  `{time_name}`: {time.get_accum_time() / num_measured_iters}")
            # Mean time per measured interval
            print(". " * 25)
            print(f"Average runtime per measured interval:")
            for time_name, time in self._times.items():
                # A measurement may be registered (e.g. first seen during warm-up) without having been
                # performed since the warm-up phase finished.
                num_meas = time.get_num_meas()
                print(
                    f"  `{time_name}`: {time.get_accum_time() / num_meas if num_meas > 0 else 'not measured yet'}"
                )
            # Total runtime
            print(". " * 25)
            print(f"Total runtime:")
            for time_name, time in self._times.items():
                print(f"  `{time_name}`: {time.get_accum_time()}")
            # Mean CPU usage
            print(". " * 25)
            if self._cpu_usage_meas_name is not None:
                cpu_usage_meas = self._times[self._cpu_usage_meas_name]
                print(
                    f"Mean CPU usage during `{self._cpu_usage_meas_name}`: "
                    f"{cpu_usage_meas.get_mean_cpu_usage() if cpu_usage_meas.get_accum_time() > 0 else 'not measured yet'}"
                )
        if len(self._one_time_measurements) > 0:
            print("~" * 50)
            print(f"One-time measurements:")
            for time_name, time in self._one_time_measurements.items():
                print(f"  `{time_name}`: {time.get_accum_time() if not time.is_running() else 'running'}")
        print("~" * 50)
        print("#" * 61)

    def _set_cpu_usage_meas_name_enabled(self, name: str):
        """TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        """
        if self._cpu_usage_meas_name is not None:
            if name == self._cpu_usage_meas_name:
                return
            else:
                raise ValueError(
                    f"CPU usage measurement name `{self._cpu_usage_meas_name}` already set. Cannot set it to `{name}`."
                )
        if name in self._times:
            raise ValueError(
                f"Measurement name `{name}` already present. A measurement must be selected for CPU usage measurements before being performed for the first time."
            )
        self._cpu_usage_meas_name = name
        self._times[name] = self._TimeAndCPUUsageAccumulator()

    def _start_meas_enabled(self, name: str):
        """TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        """
        if not name in self._times:
            self._times[name] = self._TimeAccumulator()
        if not self._warmup_finished:
            return
        if self._do_cuda_sync:
            self._torch.cuda.synchronize()
        self._times[name].start(time.time())

    def _end_meas_enabled(self, name: str):
        """TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        """
        assert name in self._times, f"Entry `{name}` not present, cannot end measurment"
        if not self._warmup_finished:
            return
        if self._do_cuda_sync:
            self._torch.cuda.synchronize()
        self._times[name].end(time.time())

    def _start_one_time_measurement_enabled(self, name: str):
        """TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        """
        assert (
            name not in self._one_time_measurements
        ), f"One-time measurement `{name}` already present, cannot start it again"
        self._one_time_measurements[name] = self._TimeAccumulator()
        self._one_time_measurements[name].start(time.time())

    def _end_one_time_measurement_enabled(self, name: str):
        """TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        """
        assert (
            name in self._one_time_measurements
        ), f"One-time measurement `{name}` not present, cannot end it"
        self._one_time_measurements[name].end(time.time())

    def _finish_iter_enabled(self):
        """TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        """
        for k, v in self._times.items():
            assert (
                not v.is_running()
            ), f"At the end of an iteration, all started measurements need to be ended. Measurement \
                for '{k}' is started but not ended."
        self._num_iters_finished += 1
        if not self._warmup_finished:
            self._warmup_finished = self._num_iters_finished == self._num_warmup_iters
        else:
            num_nonwarmup_iters = self._num_iters_finished - self._num_warmup_iters
            if num_nonwarmup_iters > 0 and (
                self._print_every_n_iters is not None and num_nonwarmup_iters % self._print_every_n_iters == 0
            ):
                self.print_eval_times()

    def _get_num_nonwarmup_iters_measured_enabled(self):
        """TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        """
        if self._warmup_finished:
            return self._num_iters_finished - self._num_warmup_iters
        else:
            return 0


if __name__ == "__main__":

    # --------------------------- Main Script ---------------------------
    stopwatch = Stopwatch()
    # To activate the stop watch, it needs to be enabled.
    # If the following step is omitted, calls to method of the stopwatch will have no effect
    # (and `get_num_nonwarmup_iters_measured()`) will always return `0``.
    # Try commenting out the following line:
    stopwatch.enable(num_warmup_iters=5, print_every_n_iters=2, do_cuda_sync=False)
    # Note that if the stopwatch is not enabled, calling its methods has minimal overhead
    # (call to an empty method).
    # -------------------------------------------------------------------

    # ---------------------------- Code Part I ----------------------------

    stopwatch.set_cpu_usage_meas_name("meas1")

    # Note that if a code part (see below) is used in isolation, the stopwatch will be disabled and
    # no error will occur. The overhead for the stopwatch is minimal in this case (call to an empty function).

    stopwatch = Stopwatch()
    stopwatch.start_one_time_measurement("complete_run")

    stopwatch.start_one_time_measurement("preparation")
    time.sleep(0.1)
    stopwatch.end_one_time_measurement("preparation")

    num_iters = 16
    for i in range(num_iters):
        # -------------------------- Code Part II --------------------------
        # This will not create a new instance, but re-use the instance created above.
        # No configuration should be done (passing parameters will raise error).
        stopwatch = Stopwatch()
        stopwatch.start_meas("meas1")
        # time.sleep(0.16)
        for i in range(1000000):
            i = float(i) ** 0.3 + 1.0 / (float(i) + 1.0)
        # ... continue and at some point call code part II ...
        # ------------------------------------------------------------------

        # -------------------------- Code Part II --------------------------
        stopwatch = Stopwatch()
        stopwatch.start_meas("meas2")
        time.sleep(0.05)
        stopwatch.end_meas("meas2")
        # ------------------------------------------------------------------

        # ----------------------- Code Part II (back) ----------------------
        # ... back from the call to code part II ...
        stopwatch.end_meas("meas1")
        # ------------------------------------------------------------------

        # -------------------------- Code Part IV --------------------------
        stopwatch = Stopwatch()
        # Measurements can be performed in only some of the iterations
        if i % 3 == 2:
            stopwatch.start_meas("meas3")
            time.sleep(0.01)
            stopwatch.end_meas("meas3")
        # We can perform a measurement for the same entry multiple times in a
        # single iteration. This can be in the same or different code parts.
        stopwatch.start_meas("meas2")
        time.sleep(0.01)
        stopwatch.end_meas("meas2")
        # ------------------------------------------------------------------

        # -------------------------- Code Part V ---------------------------
        # Finish the iteration
        stopwatch = Stopwatch()
        stopwatch.finish_iter()
        # ------------------------------------------------------------------

    # ---------------------------- Code Part VI ----------------------------
    # End the one-time measurement
    stopwatch.end_one_time_measurement("complete_run")
    # Print the final result
    print("Final measurement:")
    stopwatch.print_eval_times()
    # ----------------------------------------------------------------------
