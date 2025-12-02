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

"""
CUDA Context Safety Tests

This module tests CUDA context management in multi-threaded scenarios.
It is designed to catch CUDA_ERROR_INVALID_CONTEXT errors that occur when:
- Object is created in thread A (cuCtxPushCurrent to thread A's stack)
- Object is destroyed in thread B (cuCtxPopCurrent fails because thread B's stack is empty)

Root Cause:
    PyNvSampleReader constructor calls cuCtxPushCurrent() which pushes the context
    to the current thread's context stack. The destructor calls cuCtxPopCurrent()
    which tries to pop from the current thread's stack. If the destructor runs
    on a different thread (e.g., Python GC thread), the pop will fail because
    that thread's context stack is empty.

Fix:
    Use temporary push/pop pattern instead of long-lived push:
    - Constructor: push -> create stream -> pop (immediately)
    - Destructor: push -> destroy stream -> pop (in same thread)
    - This ensures each operation is self-contained and thread-safe.
"""

import pytest
import sys
import threading
import gc
import time
import queue
import os

import utils
import accvlab.on_demand_video_decoder as nvc


def get_test_files():
    """Get test video files and verify they exist."""
    path_base = utils.get_data_dir()
    print(f"Data directory: {path_base}")
    print(f"Data directory exists: {os.path.exists(path_base)}")

    files = utils.select_random_clip(path_base)

    return files


class TestCudaContextMultiThread:
    """
    Test CUDA context management in multi-threaded scenarios.

    These tests are designed to reproduce CUDA_ERROR_INVALID_CONTEXT errors
    that occur when:
    - Object is created in thread A (cuCtxPushCurrent to thread A's stack)
    - Object is destroyed in thread B (cuCtxPopCurrent fails because thread B's stack is empty)

    The fix should ensure that:
    - Context is pushed/popped within the same operation scope
    - Destructor works correctly regardless of which thread runs it
    """

    def test_create_main_destroy_worker_thread(self):
        """
        Test: Create object in main thread, destroy in worker thread.

        This reproduces CUDA_ERROR_INVALID_CONTEXT when:
        1. Main thread creates PyNvSampleReader (pushes context to main thread's stack)
        2. Worker thread destroys the object (tries to pop from worker thread's empty stack)

        Expected error (before fix):
            [FATAL] CUDA driver API error CUDA_ERROR_INVALID_CONTEXT at line 106
            in file PyNvSampleReader.cpp
        """
        print("\n=== Test: Create in main thread, destroy in worker thread ===")

        # Get test files first to determine num_of_file
        files = get_test_files()
        num_files = len(files) if files else 4

        # Create object in main thread
        reader = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=num_files,  # Must be >= number of files to decode
            iGpu=0,
        )
        print(f"Created reader in main thread: {threading.current_thread().name}")

        # Do some decoding to ensure the object is fully initialized
        frames = [0] * len(files)
        try:
            _ = reader.DecodeN12ToRGB(files, frames, True)
            print("Decoding completed in main thread")
        except Exception as e:
            print(f"Decoding failed with error: {e}")
            raise

        # Queue to pass object to worker thread
        obj_queue = queue.Queue()
        error_queue = queue.Queue()

        def worker_destroy():
            """Worker thread that destroys the object."""
            try:
                print(f"Worker thread started: {threading.current_thread().name}")
                obj = obj_queue.get(timeout=5)
                print(f"Worker thread received object, about to delete...")

                # Explicitly delete the object in this thread
                del obj
                gc.collect()  # Force garbage collection

                print(f"Worker thread: Object deleted successfully")
                error_queue.put(None)  # No error
            except Exception as e:
                print(f"Worker thread error: {e}")
                error_queue.put(e)

        # Start worker thread
        worker = threading.Thread(target=worker_destroy, name="DestroyerThread")
        worker.start()

        # Pass object to worker thread and release main thread's reference
        obj_queue.put(reader)
        del reader  # Remove main thread's reference

        # Wait for worker to finish
        worker.join(timeout=10)

        # Check if there was an error
        try:
            error = error_queue.get_nowait()
            if error is not None:
                pytest.fail(f"Worker thread encountered error: {error}")
        except queue.Empty:
            pytest.fail("Worker thread did not complete")

        print("Test passed: No CUDA_ERROR_INVALID_CONTEXT")

    def test_create_worker_destroy_main_thread(self):
        """
        Test: Create object in worker thread, destroy in main thread.

        This is another scenario that can trigger CUDA_ERROR_INVALID_CONTEXT:
        1. Worker thread creates PyNvSampleReader (pushes context to worker thread's stack)
        2. Main thread destroys the object (tries to pop from main thread's stack)

        Expected error (before fix):
            [FATAL] CUDA driver API error CUDA_ERROR_INVALID_CONTEXT at line 106
            in file PyNvSampleReader.cpp
        """
        print("\n=== Test: Create in worker thread, destroy in main thread ===")

        obj_queue = queue.Queue()
        error_queue = queue.Queue()

        def worker_create():
            """Worker thread that creates the object."""
            try:
                print(f"Worker thread started: {threading.current_thread().name}")

                # Get test files first to determine num_of_file
                files = get_test_files()
                num_files = len(files) if files else 4

                reader = nvc.CreateSampleReader(
                    num_of_set=2,
                    num_of_file=num_files,  # Must be >= number of files to decode
                    iGpu=0,
                )
                print(f"Created reader in worker thread")

                # Do some decoding
                frames = [0] * len(files)
                _ = reader.DecodeN12ToRGB(files, frames, True)
                print("Decoding completed in worker thread")

                # Pass object to main thread
                obj_queue.put(reader)
                print("Worker thread: Object passed to queue")
                error_queue.put(None)
            except Exception as e:
                print(f"Worker thread error: {e}")
                error_queue.put(e)
                obj_queue.put(None)

        # Start worker thread
        worker = threading.Thread(target=worker_create, name="CreatorThread")
        worker.start()
        worker.join(timeout=10)

        # Check worker thread error
        try:
            error = error_queue.get_nowait()
            if error is not None:
                pytest.fail(f"Worker thread encountered error: {error}")
        except queue.Empty:
            pytest.fail("Worker thread did not complete")

        # Get object from queue
        reader = obj_queue.get(timeout=5)
        assert reader is not None, "Failed to get reader from worker thread"

        print(f"Main thread received object: {threading.current_thread().name}")

        # Delete object in main thread - this should trigger CUDA_ERROR_INVALID_CONTEXT
        # if the context management is incorrect
        print("Main thread: About to delete object...")
        del reader
        gc.collect()

        print("Test passed: No CUDA_ERROR_INVALID_CONTEXT")

    def test_multiple_create_destroy_cycles_different_threads(self):
        """
        Test: Multiple create/destroy cycles across different threads.

        This stress tests the context management by repeatedly creating
        and destroying objects in different threads.
        """
        print("\n=== Test: Multiple create/destroy cycles across threads ===")

        num_cycles = 5
        errors = []

        def cycle_test(cycle_id: int):
            """One cycle of create and destroy."""
            try:
                thread_name = threading.current_thread().name
                print(f"Cycle {cycle_id}: Creating in {thread_name}")

                # Get test files first
                files = get_test_files()
                files_to_use = files[:2] if files else []
                num_files = max(len(files_to_use), 2)

                reader = nvc.CreateSampleReader(
                    num_of_set=2,
                    num_of_file=num_files,  # Must be >= number of files to decode
                    iGpu=0,
                )

                # Do a small decode operation
                if files_to_use:
                    frames = [0] * len(files_to_use)
                    _ = reader.DecodeN12ToRGB(files_to_use, frames, True)

                # Explicitly delete
                del reader
                gc.collect()

                print(f"Cycle {cycle_id}: Completed in {thread_name}")
                return None
            except Exception as e:
                return f"Cycle {cycle_id} error: {e}"

        # Run cycles in different threads
        threads = []
        results = queue.Queue()

        for i in range(num_cycles):

            def run_cycle(cycle_id):
                result = cycle_test(cycle_id)
                results.put(result)

            t = threading.Thread(target=run_cycle, args=(i,), name=f"CycleThread-{i}")
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads
        for t in threads:
            t.join(timeout=30)

        # Collect results
        while not results.empty():
            result = results.get()
            if result is not None:
                errors.append(result)

        if errors:
            pytest.fail(f"Errors during multi-threaded cycles: {errors}")

        print("Test passed: All cycles completed without CUDA_ERROR_INVALID_CONTEXT")

    def test_gc_triggered_destruction(self):
        """
        Test: Let Python GC destroy the object.

        Python's garbage collector may run on a different thread than
        where the object was created, potentially triggering the
        CUDA_ERROR_INVALID_CONTEXT error.
        """
        print("\n=== Test: GC-triggered destruction ===")

        def create_and_abandon():
            """Create objects and let them go out of scope."""
            # Get test files once
            files = get_test_files()
            files_to_use = files[:2] if files else []
            num_files = max(len(files_to_use), 2)

            readers = []
            for i in range(3):
                reader = nvc.CreateSampleReader(
                    num_of_set=2,
                    num_of_file=num_files,  # Must be >= number of files to decode
                    iGpu=0,
                )
                readers.append(reader)

                # Do some work
                if files_to_use:
                    frames = [0] * len(files_to_use)
                    _ = reader.DecodeN12ToRGB(files_to_use, frames, True)

            # Return without explicitly deleting - objects go out of scope
            print(f"Created {len(readers)} readers, letting them go out of scope")

        # Create objects
        create_and_abandon()

        # Force multiple GC cycles
        for i in range(3):
            print(f"GC cycle {i+1}")
            gc.collect()
            time.sleep(0.1)

        print("Test passed: GC destruction completed without error")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
