/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PyNvSampleReader.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sstream>
#include <exception>

#include <cuda.h>
#include <cuda_runtime.h>

#include "nvtx3/nvtx3.hpp"

PyNvSampleReader::PyNvSampleReader(int num_of_set, int num_of_file, int iGpu,
                                   bool bSuppressNoColorRangeWarning)
    : num_of_set(num_of_set),
      num_of_file(num_of_file),
      gpu_id(iGpu),
      suppress_no_color_range_given_warning(bSuppressNoColorRangeWarning),
      decode_result_queue(1),  // Buffer size = 1
      has_pending_task(false) {
#ifdef IS_DEBUG_BUILD
    std::cout << "New PyNvSampleReader object" << std::endl;
#endif
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu) {
        std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]"
                  << std::endl;
    }
    this->destroy_context = false;

    // To do, we can reuse current context, we can check its func
    // ck(cuCtxGetCurrent(&cuContext));
    this->cu_context = nullptr;
    if (!this->cu_context) {
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, this->gpu_id));
        ck(cuDevicePrimaryCtxRetain(&this->cu_context, cuDevice));
        this->destroy_context = true;
    }
    if (!this->cu_context) {
        throw std::domain_error(
            "[ERROR] Failed to create a cuda context. Create a "
            "cudacontext and pass it as "
            "named argument 'cudacontext = app_ctx'");
    }
    // Temporarily push context for stream creation, then immediately pop.
    // This ensures the destructor can run on any thread without issues.
    ck(cuCtxPushCurrent(this->cu_context));
    ck(cuStreamCreate(&this->cu_stream, CU_STREAM_DEFAULT));
    ck(cuCtxPopCurrent(NULL));
    VideoReaderMap.reserve(this->num_of_file);
    for (int i = 0; i < this->num_of_file; i++) {
        VideoReaderMap.emplace_back(this->num_of_set);
    }
}

PyNvSampleReader::~PyNvSampleReader() {
#ifdef IS_DEBUG_BUILD
    std::cout << "Delete PyNvSampleReader object" << std::endl;
#endif

    // Wait for any pending async decode tasks to complete
    bool need_join = false;
    {
        std::lock_guard<std::mutex> lock(async_mutex);
        need_join = has_pending_task;
    }

    // Join outside of lock to avoid deadlock
    // (worker thread needs to acquire async_mutex to set has_pending_task = false)
    if (need_join) {
        decode_worker.join();
    }

    // Clear the decode result queue
    decode_result_queue.clear();

    this->clearAllReaders();

    if (this->cu_stream) {
        // Temporarily push context for stream destruction.
        // This ensures the destructor works correctly on any thread.
        ck(cuCtxPushCurrent(this->cu_context));
        ck(cuStreamDestroy(this->cu_stream));
        ck(cuCtxPopCurrent(NULL));
    }
    if (this->destroy_context) {
        // Only release the primary context reference.
        // No need to pop - we use temporary push/pop pattern instead.
        ck(cuDevicePrimaryCtxRelease(this->gpu_id));
    }
}

// Clear all video readers before destroying context
void PyNvSampleReader::clearAllReaders() {
    // Wait for any pending async task to complete before clearing readers
    waitForPendingAsyncTask();

    for (auto& reader_map : VideoReaderMap) {
        reader_map.clearAllReaders();
    }
}

// Helper function to process video frames in parallel
template <typename T, typename Func>
std::vector<T> process_frames_in_parallel(const std::vector<std::string>& filepaths,
                                          const std::vector<int>& frame_ids,
                                          const std::vector<PyNvVideoReader*>& video_readers,
                                          Func process_frame) {
    nvtxRangePushA("Process Frames in Parallel");
    std::vector<T> res(filepaths.size());
    std::exception_ptr eptr = nullptr;
    std::mutex mutex;

    std::vector<std::thread> threads;
    threads.reserve(filepaths.size());

    for (int i = 0; i < filepaths.size(); i++) {
        threads.emplace_back([&, i]() {
            try {
                res[i] = process_frame(video_readers[i], frame_ids[i]);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(mutex);
                eptr = std::current_exception();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (eptr) {
        nvtxRangePop();
        std::rethrow_exception(eptr);
    }
    nvtxRangePop();
    return res;
}

std::vector<RGBFrame> PyNvSampleReader::run_rgb_out(const std::vector<std::string>& filepaths,
                                                    const std::vector<int> frame_ids, bool as_bgr) {
    // NOTE: Do NOT call waitForPendingAsyncTask() here!
    // This function is called by async worker thread, which would cause deadlock.
    // The wait is done at the Python API entry point (DecodeN12ToRGB binding).

    // Validate input sizes
    if (filepaths.size() != frame_ids.size()) {
        throw std::invalid_argument("filepaths.size() (" + std::to_string(filepaths.size()) +
                                    ") must equal frame_ids.size() (" + std::to_string(frame_ids.size()) +
                                    ")");
    }

    // Check that number of files doesn't exceed num_of_file
    if (filepaths.size() > static_cast<size_t>(this->num_of_file)) {
        throw std::invalid_argument(
            "Number of files to decode (" + std::to_string(filepaths.size()) + ") exceeds num_of_file (" +
            std::to_string(this->num_of_file) +
            ") specified in CreateSampleReader. Please create a new reader with larger num_of_file.");
    }

    std::vector<PyNvVideoReader*> video_readers(filepaths.size());

    nvtxRangePushA("Get Video Readers");
    for (int i = 0; i < filepaths.size(); i++) {
        FixedSizeVideoReaderMap& reader_map = this->VideoReaderMap[i];
        PyNvVideoReader* video_reader = nullptr;

        // Only create new reader if cache is not full AND key doesn't exist
        // This prevents creating readers that would be discarded (memory leak)
        if (reader_map.notFull() && !reader_map.contains(filepaths[i])) {
            video_reader = new PyNvVideoReader(filepaths[i], this->gpu_id, this->cu_context, this->cu_stream);
        }

        auto cur_video_reader = reader_map.find(filepaths[i], video_reader);
        video_readers[i] = cur_video_reader;
    }
    nvtxRangePop();

    return process_frames_in_parallel<RGBFrame>(filepaths, frame_ids, video_readers,
                                                [as_bgr](PyNvVideoReader* reader, int frame_id) {
                                                    return reader->run_single_rgb_out(frame_id, as_bgr);
                                                });
}

std::vector<DecodedFrameExt> PyNvSampleReader::run(const std::vector<std::string>& filepaths,
                                                   const std::vector<int> frame_ids) {
    // NOTE: Do NOT call waitForPendingAsyncTask() here!
    // The wait is done at the Python API entry point (Decode binding).

    // Validate input sizes
    if (filepaths.size() != frame_ids.size()) {
        throw std::invalid_argument("filepaths.size() (" + std::to_string(filepaths.size()) +
                                    ") must equal frame_ids.size() (" + std::to_string(frame_ids.size()) +
                                    ")");
    }

    // Check that number of files doesn't exceed num_of_file
    if (filepaths.size() > static_cast<size_t>(this->num_of_file)) {
        throw std::invalid_argument(
            "Number of files to decode (" + std::to_string(filepaths.size()) + ") exceeds num_of_file (" +
            std::to_string(this->num_of_file) +
            ") specified in CreateSampleReader. Please create a new reader with larger num_of_file.");
    }

    std::vector<PyNvVideoReader*> video_readers(filepaths.size());

    for (int i = 0; i < filepaths.size(); i++) {
        FixedSizeVideoReaderMap& reader_map = this->VideoReaderMap[i];
        PyNvVideoReader* video_reader = nullptr;

        // Only create new reader if cache is not full AND key doesn't exist
        // This prevents creating readers that would be discarded (memory leak)
        if (reader_map.notFull() && !reader_map.contains(filepaths[i])) {
            video_reader = new PyNvVideoReader(filepaths[i], this->gpu_id, this->cu_context, this->cu_stream);
        }

        auto cur_video_reader = reader_map.find(filepaths[i], video_reader);
        video_readers[i] = cur_video_reader;
    }

    return process_frames_in_parallel<DecodedFrameExt>(
        filepaths, frame_ids, video_readers,
        [](PyNvVideoReader* reader, int frame_id) { return reader->run_single(frame_id); });
}

void Init_PyNvSampleReader(py::module& m) {
    // Create a factory function for convenient instantiation
    m.def(
        "CreateSampleReader",
        [](int num_of_set, int num_of_file, int iGpu, bool suppressNoColorRangeWarning) {
            return std::make_shared<PyNvSampleReader>(num_of_set, num_of_file, iGpu,
                                                      suppressNoColorRangeWarning);
        },
        py::arg("num_of_set"), py::arg("num_of_file"), py::arg("iGpu") = 0,
        py::arg("suppressNoColorRangeWarning") = false,
        R"pbdoc(
            Initialize sample reader with multiple video readers.
            
            This factory function creates a PyNvSampleReader instance with the specified
            configuration for high-throughput multi-file video processing. It's the
            recommended way to create sample reader instances.
            
            Args:
                num_of_set: Number of video readers per file for parallel processing
                num_of_file: Number of files to handle simultaneously
                iGpu: GPU device ID to use for decoding (0 for primary GPU)
                suppressNoColorRangeWarning: Suppress warning when no color range can be extracted from video files (limited/MPEG range is assumed)
            
            Returns:
                PyNvSampleReader instance configured with the specified parameters
            
            Raises:
                RuntimeError: If GPU initialization fails or parameters are invalid
            
            Example:
                >>> reader = CreateSampleReader(num_of_set=2, num_of_file=3, iGpu=0)
                >>> frames = reader.Decode(['v0.mp4', 'v1.mp4'], [0, 10])
            
            Note:
                The parameter `num_of_set` in `nvc.CreateSampleReader` controls the decoding cycle:
                - For a specific decoder instance, if you are decoding clipA, after calling `DecodeN12ToRGB` `num_of_set` times, the input returns to clipA again
                - If you are continuously decoding the same clip, then `num_of_set` can be set to 1
            )pbdoc");

    // Define the PyNvSampleReader class and its methods
    py::class_<PyNvSampleReader, shared_ptr<PyNvSampleReader>>(m, "PyNvSampleReader", py::module_local(),
                                                               R"pbdoc(
        NVIDIA GPU-accelerated sample reader for multi-file video processing.
        
        This class provides high-performance video reading capabilities using NVIDIA
        hardware acceleration for multiple video files with multiple readers per file.
        It's designed for scenarios requiring high-throughput processing of multiple
        video streams simultaneously.
        
        Key Features:
        - GPU-accelerated decoding using NVIDIA hardware
        - Multiple video readers per file for parallel processing
        - Multi-file support with configurable reader pools
        - RGB and YUV output formats
        - Resource management with explicit cleanup
        - Optimized for high-throughput batch processing
        )pbdoc")
        .def(py::init<int, int, int, bool>(), py::arg("num_of_set"), py::arg("num_of_file"),
             py::arg("iGpu") = 0, py::arg("suppressNoColorRangeWarning") = false,
             R"pbdoc(
            Initialize sample reader with set of particular parameters.
            
            Args:
                num_of_set: Number of video readers per file for parallel processing
                num_of_file: Number of files to handle simultaneously
                iGpu: GPU device ID to use for decoding (0 for primary GPU)
                suppressNoColorRangeWarning: Suppress warning when no color range can be extracted from video files (limited/MPEG range is assumed)
            
            Raises:
                RuntimeError: If GPU initialization fails or parameters are invalid

            Note:
                The parameter `num_of_set` in `nvc.CreateSampleReader` controls the decoding cycle:
                - For a specific decoder instance, if you are decoding clipA, after calling `DecodeN12ToRGB` `num_of_set` times, the input returns to clipA again
                - If you are continuously decoding the same clip, then `num_of_set` can be set to 1
            )pbdoc")
        .def(
            "Decode",
            [](std::shared_ptr<PyNvSampleReader>& reader, const std::vector<std::string>& filepaths,
               const std::vector<int> frame_ids) {
                try {
                    // Wait for any pending async task before sync decode
                    reader->waitForPendingAsyncTask();
                    return reader->run(filepaths, frame_ids);
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Decodes video frames into uncompressed YUV data.
            
            This method performs GPU-accelerated decoding of specific frames from multiple
            video files using the configured reader pools. It returns frames in YUV format
            with metadata.
            
            Args:
                filepaths: List of video file paths to decode from
                frame_ids: List of frame IDs to decode from the video files
            
            Returns:
                List of DecodedFrameExt objects containing the decoded frame data.
                Each frame includes YUV pixel data, metadata, and timing information.
            
            Raises:
                RuntimeError: If video files cannot be decoded or frame IDs are invalid
                ValueError: If frame_ids contain invalid indices or filepaths is empty
            
            Example:
                >>> reader = PyNvSampleReader(num_of_set=2, num_of_file=3)
                >>> frames = reader.Decode(['video1.mp4', 'video2.mp4'], [0, 10])
                >>> print(f"Decoded {len(frames)} frames")
            )pbdoc")
        .def(
            "DecodeN12ToRGB",
            [](std::shared_ptr<PyNvSampleReader>& reader, const std::vector<std::string>& filepaths,
               const std::vector<int> frame_ids, bool as_bgr) {
                try {
                    // Wait for any pending async task before sync decode
                    reader->waitForPendingAsyncTask();

                    // Invalidate any async result by clearing the buffer
                    // This prevents users from accidentally retrieving stale async results
                    reader->clearDecodeResultBuffer();

                    return reader->run_rgb_out(filepaths, frame_ids, as_bgr);
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"), py::arg("as_bgr") = false,
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Decodes video frames and converts them to RGB/BGR format.
            
            This method performs GPU-accelerated decoding and color space conversion
            from YUV to RGB/BGR format for multiple video files. It's optimized for
            machine learning applications that require RGB input data.
            
            Args:
                filepaths: List of video file paths to decode from
                frame_ids: List of frame IDs to decode from the video files
                as_bgr: Whether to output in BGR format (True) or RGB format (False). BGR is commonly used in OpenCV applications.
            
            Returns:
                List of RGBFrame objects containing the decoded and color-converted frame data.
                Each frame includes RGB/BGR pixel data and metadata.
            
            Raises:
                RuntimeError: If video files cannot be decoded or frame IDs are invalid
                ValueError: If frame_ids contain invalid indices or filepaths is empty
            
            Example:

                Ref to Sample: `samples/SampleStreamAccess.py`
                
                >>> reader = PyNvSampleReader(num_of_set=2, num_of_file=3)
                >>> rgb_frames = reader.DecodeN12ToRGB(['video1.mp4', 'video2.mp4'], [0, 10], as_bgr=True)
                >>> print(f"Decoded {len(rgb_frames)} RGB frames")
            )pbdoc")
        .def(
            "clearAllReaders", [](std::shared_ptr<PyNvSampleReader>& reader) { reader->clearAllReaders(); },
            R"pbdoc(
            Clear all video readers and release associated resources.
            
            This method releases all video reader instances and their associated
            GPU resources. It should be called when the reader is no longer needed
            to free up GPU memory and other system resources.
            
            Example:
                >>> reader = PyNvSampleReader(num_of_set=2, num_of_file=3)
                >>> frames = reader.Decode(['video1.mp4'], [0, 10, 20])
                >>> reader.clearAllReaders()  # Clean up resources
            )pbdoc")
        .def(
            "release_device_memory",
            [](std::shared_ptr<PyNvSampleReader>& reader) { reader->ReleaseMemPools(); },
            R"pbdoc(
            Release GPU device memory pool to free up GPU memory.
            
            This method releases the GPU memory pool and resets the pool state.
            This is useful for temporarily freeing excessive GPU memory usage.
            
            Note: After calling this method, the memory pool will need to be
            re-allocated on the next decode operation.
            
            Example:
                >>> reader = PyNvSampleReader(num_of_set=2, num_of_file=3)
                >>> frames = reader.Decode(['video1.mp4'], [0, 10, 20])
                >>> reader.release_device_memory()  # Free GPU memory pool
            )pbdoc")
        .def(
            "release_decoder", [](std::shared_ptr<PyNvSampleReader>& reader) { reader->ReleaseDecoder(); },
            R"pbdoc(
            Release all video decoder instances to free up GPU memory.
            
            This method clears all video readers, which releases:
            - NvDecoder instances and their GPU frame buffers
            - Each video reader's GPUMemoryPool instances
            
            This is useful for freeing GPU memory occupied by decoder instances.
            
            Note: After calling this method, video readers will need to be
            re-created on the next decode operation.
            
            Example:
                >>> reader = PyNvSampleReader(num_of_set=2, num_of_file=3)
                >>> frames = reader.Decode(['video1.mp4'], [0, 10, 20])
                >>> reader.release_decoder()  # Free decoder instances
            )pbdoc")
        .def(
            "DecodeN12ToRGBAsync",
            [](std::shared_ptr<PyNvSampleReader>& reader, const std::vector<std::string>& filepaths,
               const std::vector<int> frame_ids, bool as_bgr) {
                try {
                    reader->DecodeN12ToRGBAsync(filepaths, frame_ids, as_bgr);
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"), py::arg("as_bgr") = false,
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Asynchronously decode video frames and convert them to RGB/BGR format.
            
            This method submits a decode task to a background thread and returns immediately.
            The decoded frames will be stored in an internal buffer and can be retrieved
            using DecodeN12ToRGBAsyncGetBuffer.
            
            .. IMPORTANT::
                **Buffer Clearing Behavior**: Calling this method will clear any pending 
                results from the internal buffer. You MUST ensure that you have already 
                retrieved all buffered results using DecodeN12ToRGBAsyncGetBuffer before 
                calling this method again. Otherwise, pending decoded frames will be 
                discarded and cannot be recovered.
                
                **Deep Copy Requirement**: After retrieving frames via DecodeN12ToRGBAsyncGetBuffer, 
                you MUST ensure that the frames have been deep-copied (e.g., using PyTorch's 
                ``clone()``, or other deep-copy operations) or have been fully 
                consumed by post-processing operations (e.g., resize) before calling 
                DecodeN12ToRGBAsync again. This is because RGBFrame objects use zero-copy 
                semantics and reference GPU memory from the internal memory pool. The memory 
                pool may reuse the same GPU memory allocation for new decode operations, 
                which could corrupt data if the previous frames are still being referenced.
            
            .. WARNING::
                **GPU Memory Management**: The GPU memory used by decoded frames is managed 
                by an internal GPU memory pool (GPUMemoryPool), not by the RGBFrame objects 
                themselves. This has important implications:
                
                1. **Zero-Copy Memory Access**: RGBFrame objects use zero-copy semantics 
                   through the ``__cuda_array_interface__`` protocol. When you convert them 
                   to PyTorch tensors using ``torch.as_tensor(frame)``, the tensor directly 
                   references the GPU memory from the memory pool without copying.
                
                2. **No Explicit Memory Release**: You cannot explicitly release the GPU 
                   memory of individual RGBFrame objects. The memory is only released when:
                   - The PyNvVideoReader instance that owns the memory pool is destroyed
                   - The memory pool is explicitly released via ReleaseMemPools()
                
                3. **Memory Lifetime**: Even after calling DecodeN12ToRGBAsyncGetBuffer and 
                   getting the frames, the GPU memory remains allocated in the memory pool 
                   until the reader is destroyed or the pool is released. PyTorch tensors 
                   created from RGBFrame objects will continue to reference this memory.
            
            If a previous async decode task is still running, this method will wait for
            it to complete before starting the new task, and print a warning.
            
            Args:
                filepaths: List of video file paths to decode from
                frame_ids: List of frame IDs to decode from the video files
                as_bgr: Whether to output in BGR format (True) or RGB format (False). 
                        BGR is commonly used in OpenCV applications.
            
            Note:
                Only one async decode task can be pending at a time. If you call this
                method while a previous task is still running, it will wait for the
                previous task to complete and print a warning.
            
            Example:
                >>> reader = PyNvSampleReader(num_of_set=2, num_of_file=3)
                >>> reader.DecodeN12ToRGBAsync(['video1.mp4', 'video2.mp4'], [0, 10], as_bgr=False)
                >>> # Do other work...
                >>> frames = reader.DecodeN12ToRGBAsyncGetBuffer(['video1.mp4', 'video2.mp4'], [0, 10], False)
                >>> # Process frames (memory is zero-copy referenced by PyTorch tensors)
                >>> tensor_list = [torch.as_tensor(frame, device='cuda').clone() for frame in frames]
                >>> # Note: GPU memory is still allocated in the memory pool
                >>> # Memory will be released when reader is destroyed or ReleaseMemPools() is called
            )pbdoc")
        .def(
            "DecodeN12ToRGBAsyncGetBuffer",
            [](std::shared_ptr<PyNvSampleReader>& reader, const std::vector<std::string>& filepaths,
               const std::vector<int> frame_ids, bool as_bgr) {
                try {
                    return reader->DecodeN12ToRGBAsyncGetBuffer(filepaths, frame_ids, as_bgr);
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"), py::arg("as_bgr") = false,
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Get decoded frames from the async decode buffer.
            
            This method retrieves decoded frames from the internal buffer that were
            previously submitted via DecodeN12ToRGBAsync. It validates that the
            requested filepaths and frame_ids match the buffered result.
            
            .. WARNING::
                **Zero-Copy GPU Memory Access**: The returned RGBFrame objects use zero-copy 
                semantics through the ``__cuda_array_interface__`` protocol. This means:
                
                1. **Memory Ownership**: The GPU memory is NOT owned by the RGBFrame objects. 
                   It is managed by an internal GPU memory pool (GPUMemoryPool) within the 
                   PyNvVideoReader instance.
                
                2. **PyTorch Tensor Conversion**: When you convert RGBFrame to PyTorch tensors 
                   using ``torch.as_tensor(frame, device='cuda')``, PyTorch will create a 
                   zero-copy tensor that directly references the GPU memory from the memory 
                   pool. No data is copied.
                
                3. **Memory Lifetime**: The GPU memory remains allocated in the memory pool 
                   even after this method returns. It will NOT be released when:
                   - RGBFrame objects go out of scope
                   - PyTorch tensors are deleted
                   - Python garbage collection runs
                
                4. **Memory Release**: The GPU memory is only released when:
                   - The PyNvSampleReader instance is destroyed
                   - ReleaseMemPools() is explicitly called on the reader
                
                5. **Memory Pool Behavior**: The memory pool reuses the same GPU memory 
                   allocation across multiple decode operations. Calling this method does 
                   NOT free the GPU memory - it only removes the frame data from the 
                   internal buffer queue.
            
            Args:
                filepaths: List of video file paths (must match the async request)
                frame_ids: List of frame IDs (must match the async request)
                as_bgr: BGR format flag (must match the async request)
            
            Returns:
                List of RGBFrame objects containing the decoded and color-converted frame data.
                Each frame includes RGB/BGR pixel data and metadata. The GPU memory is 
                managed by the internal memory pool and uses zero-copy semantics.
            
            Raises:
                RuntimeError: If no matching result is found in buffer, validation fails, or decoding failed
            
            Example:
                >>> reader = PyNvSampleReader(num_of_set=2, num_of_file=3)
                >>> reader.DecodeN12ToRGBAsync(['video1.mp4', 'video2.mp4'], [0, 10], as_bgr=False)
                >>> # Do other work...
                >>> frames = reader.DecodeN12ToRGBAsyncGetBuffer(['video1.mp4', 'video2.mp4'], [0, 10], False)
                >>> # Convert to PyTorch tensors (zero-copy, no memory allocation)
                >>> tensor_list = [torch.as_tensor(frame, device='cuda').clone() for frame in frames]
                >>> # Note: GPU memory is still in the memory pool, referenced by tensors
                >>> # Memory will persist until reader is destroyed or ReleaseMemPools() is called
            )pbdoc");
}

void PyNvSampleReader::waitForPendingAsyncTask() {
    bool need_join = false;
    {
        std::lock_guard<std::mutex> lock(async_mutex);
        need_join = has_pending_task;
    }
    // Join outside of lock to avoid deadlock
    if (need_join) {
        decode_worker.join();
    }
}

void PyNvSampleReader::ReleaseMemPools() {
    // Wait for any pending async task to complete before releasing memory
    waitForPendingAsyncTask();

    // Release GPU memory pools from all video readers
    // Each PyNvVideoReader has its own GPUMemoryPool that stores decoded frames
    for (auto& reader_map : VideoReaderMap) {
        reader_map.releaseAllMemPools();
    }
}

void PyNvSampleReader::ReleaseDecoder() {
    // Wait for any pending async task to complete before clearing readers
    waitForPendingAsyncTask();
    // Clear all video readers to release their GPU memory
    // (NvDecoder instances and their GPUMemoryPool instances)
    clearAllReaders();
}

std::string PyNvSampleReader::generate_request_key(const std::vector<std::string>& filepaths,
                                                   const std::vector<int>& frame_ids, bool as_bgr) {
    std::ostringstream oss;
    oss << as_bgr << ":";
    for (size_t i = 0; i < filepaths.size(); ++i) {
        oss << filepaths[i] << ":" << frame_ids[i];
        if (i < filepaths.size() - 1) {
            oss << ",";
        }
    }
    return oss.str();
}

bool PyNvSampleReader::validate_request(const DecodeResult& result, const std::vector<std::string>& filepaths,
                                        const std::vector<int>& frame_ids, bool as_bgr) {
    if (result.file_path_list.size() != filepaths.size() || result.frame_id_list.size() != frame_ids.size()) {
        return false;
    }

    if (result.as_bgr != as_bgr) {
        return false;
    }

    for (size_t i = 0; i < filepaths.size(); ++i) {
        if (result.file_path_list[i] != filepaths[i] || result.frame_id_list[i] != frame_ids[i]) {
            return false;
        }
    }

    return true;
}

void PyNvSampleReader::DecodeN12ToRGBAsync(const std::vector<std::string>& filepaths,
                                           const std::vector<int>& frame_ids, bool as_bgr) {
    std::unique_lock<std::mutex> lock(async_mutex);

    // If there's a pending task, wait for it to complete
    if (has_pending_task) {
        std::cerr << "[WARNING] DecodeN12ToRGBAsync: A previous async decode task is still running. "
                     "Waiting for it to complete before starting the new task."
                  << std::endl;

        // Release lock before join to avoid deadlock
        // (worker thread needs to acquire async_mutex to set has_pending_task = false)
        lock.unlock();
        decode_worker.join();
        lock.lock();  // Re-acquire lock after join

        has_pending_task = false;

        // Clear the old result from queue (it will be replaced by the new task)
        // Since queue size is 1, we can safely pop if not empty
        while (!decode_result_queue.empty()) {
            decode_result_queue.pop_front();
        }
    }

    // Capture parameters for async task
    std::vector<std::string> filepaths_copy = filepaths;
    std::vector<int> frame_ids_copy = frame_ids;
    bool as_bgr_copy = as_bgr;

    // Start async decode task
    has_pending_task = true;
    decode_worker.start([this, filepaths_copy, frame_ids_copy, as_bgr_copy]() {
        DecodeResult result;
        result.file_path_list = filepaths_copy;
        result.frame_id_list = frame_ids_copy;
        result.as_bgr = as_bgr_copy;
        result.is_ready = false;

        try {
            // Perform the actual decoding
            std::vector<RGBFrame> frames = this->run_rgb_out(filepaths_copy, frame_ids_copy, as_bgr_copy);

            // Store the result - mark as ready before pushing to queue
            result.decoded_frames = std::move(frames);
            result.is_ready = true;

            // Push result to queue (this will block if queue is full, which should not happen
            // since we cleared it)
            decode_result_queue.push_back(result);

            // Mark task as completed
            {
                std::lock_guard<std::mutex> lock(async_mutex);
                has_pending_task = false;
            }
        } catch (...) {
            // Store exception - mark as ready before pushing to queue
            result.exception = std::current_exception();
            result.is_ready = true;

            // Push result to queue even if there's an exception
            decode_result_queue.push_back(result);

            // Mark task as completed
            {
                std::lock_guard<std::mutex> lock(async_mutex);
                has_pending_task = false;
            }
        }
    });
}

void PyNvSampleReader::clearDecodeResultBuffer() {
    // Clear any existing async result from the queue
    // This invalidates stale async results when sync API is called
    while (!decode_result_queue.empty()) {
        decode_result_queue.pop_front();
    }
}

std::vector<RGBFrame> PyNvSampleReader::DecodeN12ToRGBAsyncGetBuffer(
    const std::vector<std::string>& filepaths, const std::vector<int>& frame_ids, bool as_bgr) {
    // Check if there's a pending task or result available
    {
        std::lock_guard<std::mutex> lock(async_mutex);
        if (!has_pending_task && decode_result_queue.empty()) {
            throw std::runtime_error(
                "DecodeN12ToRGBAsyncGetBuffer: No pending decode task and buffer is empty. "
                "Call DecodeN12ToRGBAsync first before calling GetBuffer, "
                "or ensure you haven't already retrieved the result.");
        }
    }

    // Wait for result to be available in queue (this will block until result is pushed)
    DecodeResult result = decode_result_queue.pop_front();

    // Result should always be ready when popped from queue (since we only push after completion)
    // But check for safety
    if (!result.is_ready) {
        throw std::runtime_error(
            "DecodeN12ToRGBAsyncGetBuffer: Internal error - result not ready when popped from queue");
    }

    // Check if there was an exception during decoding
    if (result.exception) {
        std::rethrow_exception(result.exception);
    }

    // Validate that the result matches the request
    if (!validate_request(result, filepaths, frame_ids, as_bgr)) {
        std::ostringstream oss;
        oss << "DecodeN12ToRGBAsyncGetBuffer: Request parameters do not match buffered result. "
            << "Expected: " << generate_request_key(filepaths, frame_ids, as_bgr)
            << ", Got: " << generate_request_key(result.file_path_list, result.frame_id_list, result.as_bgr);
        throw std::runtime_error(oss.str());
    }

    return result.decoded_frames;
}