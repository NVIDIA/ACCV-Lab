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
import signal
import sys
import unittest
from unittest.mock import MagicMock, call, patch

import pytest

from accvlab.on_demand_video_decoder._mps.manager import MpsManager


@pytest.fixture(autouse=True)
def reset_manager():
    MpsManager._started = False
    MpsManager._pipe_dir = ""
    yield
    MpsManager._started = False
    MpsManager._pipe_dir = ""


class TestMpsManager:
    def test_start_idempotent(self):
        with (
            patch.object(MpsManager, "_is_running", return_value=True),
            patch("subprocess.Popen") as mock_popen,
            patch("atexit.register"),
            patch("os.makedirs"),
        ):
            MpsManager.start()
            MpsManager.start()

        mock_popen.assert_not_called()

    def test_start_raises_on_daemon_timeout(self):
        with (
            patch.object(MpsManager, "_is_running", return_value=False),
            patch("subprocess.Popen"),
            patch("os.makedirs"),
            patch("time.sleep"),
            patch("time.monotonic", side_effect=[0.0, 0.1, 5.1]),
        ):
            with pytest.raises(RuntimeError, match="failed to start within 5s"):
                MpsManager.start()

    def test_stop_clears_started_before_subprocess(self):
        # If subprocess.run raises, _started must already be False so atexit
        # doesn't call stop() a second time and send a duplicate quit.
        MpsManager._started = True
        call_order = []

        def record_started(*_args, **_kwargs):
            call_order.append(("subprocess", MpsManager._started))
            raise RuntimeError("simulated subprocess failure")

        with patch("subprocess.run", side_effect=record_started):
            with pytest.raises(RuntimeError):
                MpsManager.stop()

        assert call_order == [("subprocess", False)]

    @pytest.mark.parametrize(
        "job_id, expected_suffix",
        [
            ("42", "-42"),
            ("", ""),
        ],
    )
    def test_pipe_dir_with_and_without_slurm_job_id(self, job_id, expected_suffix, monkeypatch):
        if job_id:
            monkeypatch.setenv("SLURM_JOB_ID", job_id)
        else:
            monkeypatch.delenv("SLURM_JOB_ID", raising=False)

        pipe_dir = MpsManager._pipe_directory()
        uid = os.getuid()

        assert pipe_dir == f"/tmp/accvlab-mps-{uid}{expected_suffix}"


class TestMpsLauncher:
    def test_no_args_exits_with_usage(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["accvlab-mps"])

        from accvlab.on_demand_video_decoder._mps.launcher import mps_launch

        with pytest.raises(SystemExit) as exc_info:
            mps_launch()

        assert exc_info.value.code == 1
        assert "Usage:" in capsys.readouterr().err

    def test_child_execvpe_oserror_exits_127(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["accvlab-mps", "nonexistent-command"])
        monkeypatch.setattr(os, "fork", lambda: 0)  # simulate child
        monkeypatch.setattr(os, "execvpe", MagicMock(side_effect=OSError("No such file or directory")))
        exit_called_with = []

        def fake_os_exit(code):
            exit_called_with.append(code)
            raise SystemExit(code)

        monkeypatch.setattr(os, "_exit", fake_os_exit)

        with patch.object(MpsManager, "start"):
            from accvlab.on_demand_video_decoder._mps import launcher

            with pytest.raises(SystemExit):
                launcher.mps_launch()

        assert exit_called_with == [127]
        assert "accvlab-mps:" in capsys.readouterr().err

    def test_sigint_sigquit_set_to_ign(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["accvlab-mps", "true"])
        monkeypatch.setattr(os, "fork", lambda: 99)  # simulate parent
        monkeypatch.setattr(os, "waitpid", lambda _pid, _flags: (99, 0))

        recorded_handlers = {}

        original_signal = signal.signal

        def capture_signal(sig, handler):
            recorded_handlers[sig] = handler
            return original_signal(sig, handler)

        monkeypatch.setattr(signal, "signal", capture_signal)

        with (
            patch.object(MpsManager, "start"),
            patch.object(MpsManager, "stop"),
            patch("sys.exit"),
        ):
            from accvlab.on_demand_video_decoder._mps import launcher

            launcher.mps_launch()

        assert recorded_handlers.get(signal.SIGINT) is signal.SIG_IGN
        assert recorded_handlers.get(signal.SIGQUIT) is signal.SIG_IGN

    def test_exit_code_propagation(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["accvlab-mps", "true"])
        monkeypatch.setattr(os, "fork", lambda: 99)

        # os.waitstatus_to_exitcode(exit_status) where exit_status encodes exit code 3:
        # a normal exit is encoded as (exit_code << 8).
        exit_code = 3
        wait_status = exit_code << 8
        monkeypatch.setattr(os, "waitpid", lambda _pid, _flags: (99, wait_status))

        captured_exit = []
        monkeypatch.setattr(sys, "exit", lambda code: captured_exit.append(code))

        with (
            patch.object(MpsManager, "start"),
            patch.object(MpsManager, "stop"),
        ):
            from accvlab.on_demand_video_decoder._mps import launcher

            launcher.mps_launch()

        assert captured_exit == [exit_code]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
