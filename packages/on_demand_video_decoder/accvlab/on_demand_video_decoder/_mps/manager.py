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

import atexit
import os
import subprocess
import time


class MpsManager:
    _started: bool = False
    _pipe_dir: str = ""

    @classmethod
    def _pipe_directory(cls) -> str:
        if not cls._pipe_dir:
            uid = os.getuid()
            job_id = os.environ.get("SLURM_JOB_ID", "")
            suffix = f"-{job_id}" if job_id else ""
            cls._pipe_dir = f"/tmp/accvlab-mps-{uid}{suffix}"
        return cls._pipe_dir

    @classmethod
    def start(cls) -> None:
        if cls._started:
            return
        pipe_dir = cls._pipe_directory()
        log_dir = pipe_dir.replace("accvlab-mps-", "accvlab-log-", 1)
        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = log_dir
        os.environ["ACCVLAB_MPS_LEVEL"] = "task"
        if not cls._is_running():
            subprocess.Popen(["nvidia-cuda-mps-control", "-d"])
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if cls._is_running():
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("nvidia-cuda-mps-control daemon failed to start within 5s")
        atexit.register(cls.stop)
        cls._started = True

    @classmethod
    def stop(cls) -> None:
        if not cls._started:
            return
        cls._started = False
        subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=b"quit\n",
            capture_output=True,
            env=os.environ.copy(),
            timeout=5,
            check=False,
        )

    @classmethod
    def _is_running(cls) -> bool:
        r = subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=b"get_server_list\n",
            capture_output=True,
            env=os.environ.copy(),
            timeout=2,
            check=False,
        )
        return r.returncode == 0
