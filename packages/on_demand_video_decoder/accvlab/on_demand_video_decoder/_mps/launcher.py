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


def mps_launch() -> None:
    if len(sys.argv) < 2:
        print("Usage: accvlab-mps <command> [args...]", file=sys.stderr)
        sys.exit(1)

    from .manager import MpsManager

    MpsManager.start()

    pid = os.fork()

    if pid == 0:
        # Child: replace this process with the user's command.
        # execvpe wipes all Python state (including atexit), so the child starts clean.
        try:
            os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
        except OSError as e:
            print(f"accvlab-mps: {e}", file=sys.stderr)
        os._exit(127)

    # Parent: sole job is to forward signals and stop the daemon when the child exits.
    def _forward(sig: int, _frame) -> None:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass

    # SIGHUP/SIGTERM are sent to this PID directly — forward to child.
    signal.signal(signal.SIGHUP, _forward)
    signal.signal(signal.SIGTERM, _forward)
    # SIGINT/SIGQUIT come from the terminal to the whole process group,
    # so the child already receives them; parent ignores to avoid racing.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)

    _, status = os.waitpid(pid, 0)

    MpsManager.stop()

    sys.exit(os.waitstatus_to_exitcode(status))
