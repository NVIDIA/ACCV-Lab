/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <torch/extension.h>

#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

enum class NodeKind {
    List,
    Tuple,
    Dict,
    TensorLeaf,
    Passthrough,
};

// Minimal PyTree representation of the input structure.
//
// - Lists/tuples store children in `seq` (and preserve tuple/list kind).
// - Dicts store ordered key/value pairs in `items` (keys preserved as Python objects).
// - Tensor leaves store `tensor_idx` (index into CopyState.inputs/outputs).
// - Passthrough leaves store the original Python object and are returned unchanged in output.
struct Node {
    NodeKind kind{NodeKind::Passthrough};

    std::vector<Node> seq;
    std::vector<std::pair<py::object, Node>> items;

    size_t tensor_idx{0};
    py::object obj;
};

// Minimal RAII wrapper around a CUDA event.
//
// Events are used to track completion of all enqueued copies on a CUDA stream,
// so `ready()` can poll and `get()` / destructor can wait before releasing
// staging buffers.
struct CudaEvent {
    cudaEvent_t ev{nullptr};
    int device_index{-1};

    CudaEvent() = default;
    CudaEvent(cudaEvent_t e, int dev) : ev(e), device_index(dev) {}

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept {
        ev = other.ev;
        device_index = other.device_index;
        other.ev = nullptr;
        other.device_index = -1;
    }
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            cleanup_no_throw();
            ev = other.ev;
            device_index = other.device_index;
            other.ev = nullptr;
            other.device_index = -1;
        }
        return *this;
    }

    ~CudaEvent() { cleanup_no_throw(); }

    void cleanup_no_throw() noexcept {
        if (ev != nullptr) {
            // cudaEventDestroy does not require the Python GIL.
            cudaEventDestroy(ev);
            ev = nullptr;
            device_index = -1;
        }
    }
};

struct PyConversionCtx {
    py::object numpy_ndarray_type;
    py::object numpy_ascontiguousarray;
    py::object torch_from_numpy;
};

static PyConversionCtx make_py_conversion_ctx_from_cache(const py::dict& cache) {
    // Called with GIL held. `cache` is a Python-owned dict stored on the module object.
    PyConversionCtx ctx;
    ctx.numpy_ndarray_type = cache["numpy_ndarray_type"];
    ctx.numpy_ascontiguousarray = cache["numpy_ascontiguousarray"];
    ctx.torch_from_numpy = cache["torch_from_numpy"];
    return ctx;
}

static Node make_tensor_leaf_node(const std::vector<at::Tensor>& inputs) {
    Node out;
    out.kind = NodeKind::TensorLeaf;
    out.tensor_idx = inputs.size() - 1;
    return out;
}

static Node traverse_build_tree_impl(const py::handle& obj, const PyConversionCtx& ctx,
                                     std::vector<at::Tensor>& inputs);

static Node traverse_sequence(NodeKind kind, ssize_t n, const std::function<py::handle(ssize_t)>& get_item,
                              const PyConversionCtx& ctx, std::vector<at::Tensor>& inputs) {
    Node out;
    out.kind = kind;
    out.seq.reserve(static_cast<size_t>(n));
    for (ssize_t i = 0; i < n; ++i) {
        py::handle item = get_item(i);
        out.seq.push_back(traverse_build_tree_impl(item, ctx, inputs));
    }
    return out;
}

// Traverse an input Python object and build:
// - a PyTree describing container structure (lists/tuples/dicts),
// - a flat list of tensor leaves (`inputs`) (numpy arrays are converted to tensors),
// - passthrough leaves stored as Python objects and returned unchanged.
//
// Invariant: TensorLeaf nodes store indices aligned between `inputs[i]` and `outputs[i]`.
//
// NOTE: Must be called with the GIL held (touches Python objects).
static Node traverse_build_tree_impl(const py::handle& obj, const PyConversionCtx& ctx,
                                     std::vector<at::Tensor>& inputs) {
    // Containers
    if (py::isinstance<py::list>(obj)) {
        auto in_list = py::reinterpret_borrow<py::list>(obj);
        const ssize_t n = static_cast<ssize_t>(in_list.size());
        const Node out = traverse_sequence(
            NodeKind::List, n, [&](ssize_t i) { return in_list[i]; }, ctx, inputs);
        return out;
    }

    if (py::isinstance<py::tuple>(obj)) {
        auto in_tup = py::reinterpret_borrow<py::tuple>(obj);
        const ssize_t n = static_cast<ssize_t>(in_tup.size());
        const Node out = traverse_sequence(
            NodeKind::Tuple, n, [&](ssize_t i) { return in_tup[i]; }, ctx, inputs);
        return out;
    }

    if (py::isinstance<py::dict>(obj)) {
        auto in_dict = py::reinterpret_borrow<py::dict>(obj);
        Node out;
        out.kind = NodeKind::Dict;
        out.items.reserve(static_cast<size_t>(in_dict.size()));
        for (auto kv : in_dict) {
            py::object k = py::reinterpret_borrow<py::object>(kv.first);
            py::handle v = kv.second;
            out.items.emplace_back(std::move(k), traverse_build_tree_impl(v, ctx, inputs));
        }
        return out;
    }

    // Tensor leaves
    if (py::isinstance(obj, ctx.numpy_ndarray_type)) {
        py::object tensor_obj;
        try {
            tensor_obj = ctx.torch_from_numpy(obj);  // usually zero-copy
        } catch (const py::error_already_set&) {
            // If direct conversion fails (e.g., unusual strides), make a contiguous copy and retry.
            py::object contiguous = ctx.numpy_ascontiguousarray(obj);
            tensor_obj = ctx.torch_from_numpy(contiguous);
        }
        inputs.push_back(py::cast<at::Tensor>(tensor_obj));
        const Node out = make_tensor_leaf_node(inputs);
        return out;
    }

    try {
        inputs.push_back(py::cast<at::Tensor>(obj));
        const Node out = make_tensor_leaf_node(inputs);
        return out;
    } catch (const py::cast_error&) {
        // Passthrough: return the original object unchanged.
        Node out;
        out.kind = NodeKind::Passthrough;
        out.obj = py::reinterpret_borrow<py::object>(obj);
        return out;
    }
}

// Parse a device string (e.g. "cpu", "cuda:0") into a c10::Device and convert errors
// into a user-friendly runtime_error for Python.
static c10::Device parse_device(const std::string& device_str) {
    try {
        c10::Device dev(device_str);
        return dev;
    } catch (const c10::Error& /*e*/) {
        throw std::runtime_error(std::string("Invalid device string: '") + device_str + "'");
    }
}

}  // namespace

namespace {

struct CopyState {
    // Flattened tensor leaves extracted during input traversal (GIL-held).
    std::vector<at::Tensor> inputs;
    // Outputs aligned with `inputs` (same indexing).
    std::vector<at::Tensor> outputs;
    // Optional per-tensor pinned staging buffers (only used when pinning is enabled
    // and the tensor is not part of the packed buffer).
    std::vector<at::Tensor> pinned_buffers;
    // Per-chunk packed CPU staging.  Each chunk has an over-sized allocation (full) and an
    // aligned narrow view (the actual buffer used for memcpy / H2D).
    std::vector<at::Tensor> packed_cpu_chunks_full;
    std::vector<at::Tensor> packed_cpu_chunks;
    // Completion events for CUDA submission.
    std::vector<CudaEvent> events;

    c10::Device target_device{c10::kCPU};
    bool use_pinned_staging{true};
    bool pack_cpu_tensors{true};
    // Minimum alignment (bytes) for each tensor start within the packed buffer.
    // Effective per-tensor alignment is max(min_packed_alignment_bytes, element_size()).
    int64_t min_packed_alignment_bytes{16};
    // Maximum bytes per packed chunk.  When the total packed data exceeds this, multiple
    // chunks are allocated, each transferred with its own H2D copy.
    int64_t max_packed_chunk_bytes{32 * 1024 * 1024};

    // CUDA streams captured at call time (on the user's thread) so that work enqueued by
    // the copier is correctly ordered with respect to the user's preceding GPU operations.
    // synchronize_source_streams() uses these to establish all necessary cross-stream
    // dependencies before any per-tensor copies are enqueued.
    std::optional<at::cuda::CUDAStream> target_stream;  // target CUDA device (H2D / D2D)
    bool target_stream_used{false};
    std::unordered_map<int, at::cuda::CUDAStream> src_streams;  // per source CUDA device
    // Events recorded at capture time on each source stream. For D2D,
    // synchronize_source_streams makes the target stream wait on these, pinning the
    // sync point to the moment start_copy was called (not whenever the background thread
    // happens to run).  Cleaned up automatically via CudaEvent RAII.
    std::unordered_map<int, CudaEvent> src_capture_events;
    // Source device indices whose streams received copy work (populated by
    // synchronize_source_streams; used for completion-event recording).
    std::vector<int> src_streams_used;

    // If scheduling fails in the background task, the exception is captured here
    // and rethrown in `ready()` / `get()`.
    std::exception_ptr exc;

    // Completion signal for background submission.
    // `done` becomes ready once staging + copy submission has finished (or failed).
    std::shared_future<void> done;
};

class CopyThreadPool {
   public:
    static CopyThreadPool& instance() {
        static CopyThreadPool pool;
        return pool;
    }

    // Enqueue a background task. Tasks MUST NOT call Python APIs (no GIL); only ATen/CUDA code is allowed.
    void enqueue(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            tasks_.push_back(std::move(fn));
        }
        cond_var_.notify_one();
    }

   private:
    // Create a small, process-wide worker pool to overlap CPU staging/CUDA submission with Python work.
    CopyThreadPool() {
        const unsigned hardware_conc = std::thread::hardware_concurrency();
        const unsigned n_workers = std::max(1u, std::min(4u, (hardware_conc == 0u ? 4u : hardware_conc)));
        workers_.reserve(n_workers);
        for (unsigned i = 0; i < n_workers; ++i) {
            workers_.emplace_back([this]() { this->worker_loop(); });
        }
    }

    ~CopyThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            stop_ = true;
        }
        cond_var_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    void worker_loop() {
        while (true) {
            std::function<void()> fn;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_var_.wait(lock, [&]() { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) {
                    return;
                }
                fn = std::move(tasks_.front());
                tasks_.pop_front();
            }
            fn();
        }
    }

    std::mutex mutex_;
    std::condition_variable cond_var_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stop_{false};
};

// A precomputed plan for the optional "pack many small CPU tensors into one staging buffer" fast path.
// When enabled, multiple small contiguous CPU tensors (mix of different dtypes allowed) are copied into a
// single packed *byte* buffer (pinned or pageable), transferred with a single H2D, and then reconstructed as
// per-tensor views sharing the packed GPU storage.
//
// For each input i: byte_offset_by_input[i] is the starting *byte* offset inside the packed buffer,
// or -1 if this input is not packed.
struct PackPlan {
    // For each input leaf i: starting byte offset inside its chunk, or -1 if not packed.
    // IMPORTANT: This is checked for all tensors, not only the packed ones, and not only if packing is
    //            enabled. Therefore, it has to be initialized with -1 for all non-packed inputs.
    std::vector<int64_t> byte_offset_by_input;  // -1 => not packed
    // For each input leaf i: which chunk it belongs to, or -1 if not packed.
    std::vector<int64_t> chunk_index_by_input;  // -1 => not packed
    // Byte size of each chunk (one entry per chunk).
    std::vector<int64_t> chunk_sizes;
    // Whether packing is enabled for this call (if false, treat everything as "not packed").
    bool enabled{false};
};

static inline int64_t round_up_i64(int64_t x, int64_t a) {
    if (a <= 1) {
        return x;
    }
    const int64_t rem = x % a;
    const int64_t res = rem == 0 ? x : (x + (a - rem));
    return res;
}

static inline int64_t next_pow2_i64(int64_t x) {
    if (x <= 1) {
        return 1;
    }
    // Round up to the next power of two (clamped to int64 range).
    uint64_t v = static_cast<uint64_t>(x - 1);
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v += 1;
    if (v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return std::numeric_limits<int64_t>::max();
    }
    return static_cast<int64_t>(v);
}

static inline int64_t packed_buffer_alignment_bytes(const CopyState& cs) {
    // Ensure the packed buffer itself is aligned to at least this many bytes.
    // Also round up to a power-of-two so we can use bit-masking for pointer alignment.
    const int64_t requested = std::max<int64_t>(16, cs.min_packed_alignment_bytes);
    return next_pow2_i64(requested);
}

static inline int64_t aligned_slice_offset_bytes(const void* base_ptr, int64_t alignment_pow2) {
    if (alignment_pow2 <= 1) {
        return 0;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(base_ptr);
    const uintptr_t a = static_cast<uintptr_t>(alignment_pow2);
    const uintptr_t aligned = (base + (a - 1)) & ~(a - 1);
    int64_t off = static_cast<int64_t>(aligned - base);
    return off;
}

// Bucket ordering key: clamp a required alignment to {16,8,4,2,1} by rounding down to the nearest bucket <= 16.
// IMPORTANT: This is only used for ordering; the actual per-tensor alignment is preserved separately.
static inline int64_t pack_bucket_key(int64_t required_align) {
    if (required_align >= 16) {
        return 16;
    }
    if (required_align >= 8) {
        return 8;
    }
    if (required_align >= 4) {
        return 4;
    }
    if (required_align >= 2) {
        return 2;
    }
    return 1;
}

// Packing candidate: an input tensor i that can be packed into the CPU staging byte buffer.
struct PackCandidate {
    size_t idx;
    int64_t bytes;
    int64_t required_align;
};

// Buckets of packing candidates in descending alignment order.
// Note: complex128 element_size() is 16, complex64 is 8.
struct PackBuckets {
    std::vector<PackCandidate> a16;
    std::vector<PackCandidate> a8;
    std::vector<PackCandidate> a4;
    std::vector<PackCandidate> a2;
    std::vector<PackCandidate> a1;

    void add(PackCandidate c) {
        switch (pack_bucket_key(c.required_align)) {
            case 16:
                a16.push_back(c);
                break;
            case 8:
                a8.push_back(c);
                break;
            case 4:
                a4.push_back(c);
                break;
            case 2:
                a2.push_back(c);
                break;
            default:
                a1.push_back(c);
                break;
        }
    }

    template <typename F>
    void for_each_bucket_desc(F&& f) const {
        f(16, a16);
        f(8, a8);
        f(4, a4);
        f(2, a2);
        f(1, a1);
    }
};

static std::optional<PackCandidate> make_pack_candidate(const CopyState& copy_state, size_t i,
                                                        int64_t min_align) {
    // Heuristic thresholds: only pack "small" tensors.
    constexpr int64_t kPackMaxBytesPerTensor = 256 * 1024;  // 256KB

    const auto& in = copy_state.inputs[i];
    // Only consider CPU tensors that will be transferred to CUDA.
    if (!in.device().is_cpu() || in.device() == copy_state.target_device) {
        return std::nullopt;
    }
    // Packing requires a flat contiguous view.
    if (!in.is_contiguous()) {
        return std::nullopt;
    }
    const int64_t bytes = in.numel() * in.element_size();
    // Skip tensors that are too big; packing targets "many tiny tensors" overhead.
    if (bytes == 0 || bytes > kPackMaxBytesPerTensor) {
        return std::nullopt;
    }
    const int64_t elem_sz = static_cast<int64_t>(in.element_size());
    // Effective alignment must be >= requested minimum AND must guarantee element alignment.
    // If min_align is not a multiple of elem_sz, round up to the next multiple to preserve
    // the invariant that byte_offset % elem_sz == 0.
    int64_t required_align = std::max<int64_t>(min_align, elem_sz);
    required_align = round_up_i64(required_align, elem_sz);
    return PackCandidate{i, bytes, required_align};
}

// Assign byte offsets within chunked packed buffers for each candidate tensor, processing
// alignment buckets in descending order to minimise inter-tensor padding.  When a tensor
// would exceed `max_chunk_bytes` in the current chunk, a new chunk is started.  Populates
// `pack_plan.byte_offset_by_input`, `chunk_index_by_input`, and `chunk_sizes`.
static void layout_packed_offsets(const PackBuckets& buckets, PackPlan& pack_plan, int64_t& packed_count,
                                  int64_t max_chunk_bytes) {
    int64_t cursor = 0;
    int64_t chunk_idx = 0;
    packed_count = 0;

    auto finalize_chunk = [&]() {
        if (cursor > 0) {
            pack_plan.chunk_sizes.push_back(cursor);
            cursor = 0;
            ++chunk_idx;
        }
    };

    auto pack_bucket = [&](int64_t bucket_align, const std::vector<PackCandidate>& bucket) {
        if (bucket.empty()) {
            return;
        }
        for (const auto& c : bucket) {
            int64_t aligned_cursor = round_up_i64(cursor, c.required_align);
            if (aligned_cursor + c.bytes > max_chunk_bytes && cursor > 0) {
                finalize_chunk();
                aligned_cursor = round_up_i64(cursor, c.required_align);
            }
            cursor = aligned_cursor;
            pack_plan.byte_offset_by_input[c.idx] = cursor;
            pack_plan.chunk_index_by_input[c.idx] = chunk_idx;
            cursor += c.bytes;
            packed_count += 1;
        }
    };

    buckets.for_each_bucket_desc(pack_bucket);
    if (cursor > 0) {
        pack_plan.chunk_sizes.push_back(cursor);
    }
}

// Decide whether to enable the packed-CPU-tensors fast path and, if enabled, compute
// per-tensor chunk assignments and byte offsets within each chunk.
static PackPlan compute_pack_plan(const CopyState& copy_state) {
    PackPlan pack_plan;
    const size_t n = copy_state.inputs.size();
    pack_plan.byte_offset_by_input.assign(n, -1);
    pack_plan.chunk_index_by_input.assign(n, -1);

    if (!copy_state.pack_cpu_tensors || !copy_state.target_device.is_cuda()) {
        return pack_plan;
    }

    // We only pack tensors that:
    // - are on CPU (and not already on the target device),
    // - are contiguous (so we can treat them as a flat buffer),
    // - are "small enough" individually,
    //
    // Mixed-dtype packing: we pack raw bytes and reconstruct typed tensors as views
    // sharing the packed GPU storage.
    const int64_t min_align = std::max<int64_t>(1, copy_state.min_packed_alignment_bytes);
    PackBuckets buckets;
    for (size_t i = 0; i < n; ++i) {
        if (auto cand = make_pack_candidate(copy_state, i, min_align)) {
            buckets.add(*cand);
        }
    }

    int64_t packed_count = 0;
    layout_packed_offsets(buckets, pack_plan, packed_count, copy_state.max_packed_chunk_bytes);

    if (packed_count >= 2 && !pack_plan.chunk_sizes.empty()) {
        pack_plan.enabled = true;
    } else {
        pack_plan.enabled = false;
        pack_plan.chunk_sizes.clear();
        std::fill(pack_plan.byte_offset_by_input.begin(), pack_plan.byte_offset_by_input.end(), -1);
        std::fill(pack_plan.chunk_index_by_input.begin(), pack_plan.chunk_index_by_input.end(), -1);
    }
    return pack_plan;
}

// Allocate staging buffers required by the chosen plans:
// - packed CPU buffer (pinned or pageable) if packing is enabled
// - per-tensor pinned buffers for remaining CPU tensors if pinning is enabled
//
// This function only allocates; it does not perform data copies.
static void allocate_staging_buffers(CopyState& copy_state, const PackPlan& pack_plan) {
    // Per-chunk packed CPU buffers (pinned or pageable).
    if (pack_plan.enabled) {
        auto opts = at::TensorOptions().dtype(at::kByte).device(c10::kCPU).pinned_memory(
            copy_state.use_pinned_staging);
        const int64_t alignment = packed_buffer_alignment_bytes(copy_state);
        copy_state.packed_cpu_chunks_full.reserve(pack_plan.chunk_sizes.size());
        copy_state.packed_cpu_chunks.reserve(pack_plan.chunk_sizes.size());
        for (int64_t chunk_bytes : pack_plan.chunk_sizes) {
            const int64_t total_alloc = chunk_bytes + alignment - 1;
            auto full = at::empty({total_alloc}, opts);
            const int64_t off = aligned_slice_offset_bytes(full.data_ptr(), alignment);
            copy_state.packed_cpu_chunks.push_back(full.narrow(0, off, chunk_bytes));
            copy_state.packed_cpu_chunks_full.push_back(std::move(full));
        }
    }

    // Per-tensor pinned buffers for CPU tensors not covered by packing (CPU->CUDA).
    if (copy_state.use_pinned_staging && copy_state.target_device.is_cuda()) {
        for (size_t i = 0; i < copy_state.inputs.size(); ++i) {
            const auto& in = copy_state.inputs[i];
            if (!in.device().is_cpu() || in.device() == copy_state.target_device) {
                continue;
            }
            if (pack_plan.byte_offset_by_input[i] >= 0) {
                continue;
            }
            auto opts = in.options().device(c10::kCPU).pinned_memory(true);
            copy_state.pinned_buffers[i] = at::empty(in.sizes(), opts);
        }
    }

    // Per-tensor pinned buffers for CUDA tensors when target is CPU (CUDA->CPU / D2H).
    // Pinned host memory enables non_blocking D2H copies on a CUDA stream.
    if (copy_state.use_pinned_staging && !copy_state.target_device.is_cuda()) {
        for (size_t i = 0; i < copy_state.inputs.size(); ++i) {
            const auto& in = copy_state.inputs[i];
            if (!in.device().is_cuda()) {
                continue;
            }
            auto opts = in.options().device(c10::kCPU).pinned_memory(true);
            copy_state.pinned_buffers[i] = at::empty(in.sizes(), opts);
        }
    }
}

// Fill CPU staging buffers (packed buffer slices and/or per-tensor pinned buffers).
//
// This is the CPU-heavy part and is parallelized via at::parallel_for.
// It does NOT enqueue CUDA transfers; it only prepares source buffers.
static void fill_cpu_staging_buffers(CopyState& copy_state, const PackPlan& pack_plan) {
    if (!copy_state.target_device.is_cuda()) {
        return;
    }

    auto stage_one = [&](size_t i) -> void {
        const auto& in = copy_state.inputs[i];
        if (!in.device().is_cpu() || in.device() == copy_state.target_device) {
            return;
        }
        const int64_t off = pack_plan.byte_offset_by_input[i];
        if (off >= 0) {
            const int64_t chunk_idx = pack_plan.chunk_index_by_input[i];
            const int64_t bytes = in.numel() * in.element_size();
            auto* dst = static_cast<uint8_t*>(
                            copy_state.packed_cpu_chunks[static_cast<size_t>(chunk_idx)].data_ptr()) +
                        off;
            const auto* src = static_cast<const uint8_t*>(in.data_ptr());
            std::memcpy(dst, src, static_cast<size_t>(bytes));
            return;
        }
        auto& pinned = copy_state.pinned_buffers[i];
        if (pinned.defined()) {
            pinned.copy_(in, /*non_blocking=*/false);
        }
    };

    at::parallel_for(0, static_cast<int64_t>(copy_state.inputs.size()), 1, [&](int64_t begin, int64_t end) {
        for (int64_t ii = begin; ii < end; ++ii) {
            stage_one(static_cast<size_t>(ii));
        }
    });
}

// Enqueue packed CPU->CUDA transfers (one H2D per chunk) and populate copy_state.outputs[i]
// with GPU views/slices into the corresponding chunk's GPU buffer.
static void enqueue_packed_transfer(CopyState& copy_state, const PackPlan& pack_plan) {
    if (!copy_state.target_device.is_cuda() || copy_state.packed_cpu_chunks.empty() ||
        !copy_state.target_stream.has_value()) {
        return;
    }
    const auto stream = *copy_state.target_stream;
    c10::cuda::CUDAGuard guard(stream.device_index());
    at::cuda::CUDAStreamGuard stream_guard(stream);
    copy_state.target_stream_used = true;

    auto gpu_opts = at::TensorOptions().dtype(at::kByte).device(copy_state.target_device);
    const int64_t alignment = packed_buffer_alignment_bytes(copy_state);

    // Allocate GPU buffers and enqueue H2D copies for each chunk.
    const size_t num_chunks = pack_plan.chunk_sizes.size();
    std::vector<at::Tensor> gpu_chunks(num_chunks);
    std::vector<int64_t> gpu_base_offsets(num_chunks);
    for (size_t c = 0; c < num_chunks; ++c) {
        const int64_t chunk_bytes = pack_plan.chunk_sizes[c];
        const int64_t total_alloc = chunk_bytes + alignment - 1;
        auto gpu_full = at::empty({total_alloc}, gpu_opts);
        const int64_t base_off = aligned_slice_offset_bytes(gpu_full.data_ptr(), alignment);
        gpu_chunks[c] = gpu_full.narrow(0, base_off, chunk_bytes);
        gpu_base_offsets[c] = base_off;
        gpu_chunks[c].copy_(copy_state.packed_cpu_chunks[c],
                            /*non_blocking=*/copy_state.use_pinned_staging);
    }

    // Create per-tensor output views referencing the correct chunk's GPU storage.
    for (size_t i = 0; i < copy_state.inputs.size(); ++i) {
        const int64_t off = pack_plan.byte_offset_by_input[i];
        if (off < 0) {
            continue;
        }
        const auto chunk_idx = static_cast<size_t>(pack_plan.chunk_index_by_input[i]);
        const int64_t base_off = gpu_base_offsets[chunk_idx];
        const auto& in = copy_state.inputs[i];
        const int64_t elem_sz = static_cast<int64_t>(in.element_size());
        if (elem_sz <= 0 || ((base_off + off) % elem_sz) != 0) {
            throw std::runtime_error(
                "Packed buffer alignment invariant violated (byte offset not divisible by element size).");
        }
        const int64_t storage_off_elems = (base_off + off) / elem_sz;
        auto out = at::empty({0}, in.options().device(copy_state.target_device));
        out.set_(gpu_chunks[chunk_idx].storage(), storage_off_elems, in.sizes(), in.strides());
        copy_state.outputs[i] = out;
    }
}

// Single synchronization point for all captured source CUDA streams, regardless of copy
// direction.  Called once before any per-tensor copies are enqueued.
//
// - D2H (CPU target, CUDA sources): copies will run on the source streams, so ordering with
//   prior GPU work is implicit.  We mark these streams as used for completion-event tracking.
// - D2D (CUDA target, source on a different device): copies run on the target stream, so we
//   make it wait on the capture-time event recorded in start_copy_impl.  This pins the sync
//   point to the moment start_copy was called, regardless of background-thread timing.
// - H2D / CPU→CPU: no source CUDA streams; nothing to do.
static void synchronize_source_streams(CopyState& copy_state) {
    for (const auto& [src_dev_idx, src_stream] : copy_state.src_streams) {
        if (!copy_state.target_device.is_cuda()) {
            // D2H: work runs on the source stream — implicit ordering with prior GPU work.
            copy_state.src_streams_used.push_back(src_dev_idx);
            continue;
        }
        if (src_dev_idx == static_cast<int>(copy_state.target_stream->device_index())) {
            continue;  // same device as target — tensors reused as-is
        }
        // D2D: target stream waits on the capture-time event from the source stream.
        auto ev_it = copy_state.src_capture_events.find(src_dev_idx);
        if (ev_it == copy_state.src_capture_events.end() || ev_it->second.ev == nullptr) {
            throw std::runtime_error("Internal error: no capture event for source CUDA device " +
                                     std::to_string(src_dev_idx));
        }
        auto st = cudaStreamWaitEvent(copy_state.target_stream->stream(), ev_it->second.ev, 0);
        if (st != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamWaitEvent failed: ") + cudaGetErrorString(st));
        }
    }
}

// Enqueue per-tensor transfers for all leaves not handled by packing.
// synchronize_source_streams() must be called before this function.
//
// Behavior summary:
// - tensors already on target device are reused
// - CUDA->CPU (D2H): copies run on the captured source-device stream.  With pinned staging,
//   an async D2H into the pinned buffer is used; otherwise a synchronous `.to()`.
// - CPU->CPU: synchronous `.to()`
// - *->CUDA: copies run on the captured target stream.  D2D is safe because
//   synchronize_source_streams inserted the necessary cross-device event waits.
//   Pinned staging enables non_blocking H2D.
static void enqueue_per_tensor_transfers(CopyState& copy_state, const PackPlan& pack_plan) {
    for (size_t i = 0; i < copy_state.inputs.size(); ++i) {
        const auto& in = copy_state.inputs[i];

        if (pack_plan.byte_offset_by_input[i] >= 0) {
            continue;  // handled by packed transfer
        }

        if (in.device() == copy_state.target_device) {
            copy_state.outputs[i] = in;
            continue;
        }

        // --- non-CUDA target (D2H / CPU→CPU) ---
        if (!copy_state.target_device.is_cuda()) {
            if (in.device().is_cuda()) {
                const auto src_dev_idx = static_cast<int>(in.device().index());
                const auto& stream = copy_state.src_streams.at(src_dev_idx);
                c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(src_dev_idx));
                at::cuda::CUDAStreamGuard stream_guard(stream);
                if (copy_state.use_pinned_staging && copy_state.pinned_buffers[i].defined()) {
                    copy_state.pinned_buffers[i].copy_(in, /*non_blocking=*/true);
                    copy_state.outputs[i] = copy_state.pinned_buffers[i];
                } else {
                    copy_state.outputs[i] = in.to(copy_state.target_device);
                }
            } else {
                copy_state.outputs[i] = in.to(copy_state.target_device);
            }
            continue;
        }

        // --- CUDA target (H2D / D2D) ---
        const auto target_stream = *copy_state.target_stream;
        c10::cuda::CUDAGuard guard(target_stream.device_index());
        at::cuda::CUDAStreamGuard stream_guard(target_stream);
        copy_state.target_stream_used = true;

        copy_state.outputs[i] = at::empty(in.sizes(), in.options().device(copy_state.target_device));
        if (copy_state.use_pinned_staging && in.device().is_cpu() && copy_state.pinned_buffers[i].defined()) {
            copy_state.outputs[i].copy_(copy_state.pinned_buffers[i], /*non_blocking=*/true);
        } else {
            copy_state.outputs[i].copy_(in, /*non_blocking=*/in.device().is_cuda());
        }
    }
}

static void record_event_on_stream(CopyState& copy_state, at::cuda::CUDAStream stream, int dev_idx) {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev_idx));
    cudaEvent_t ev = nullptr;
    const auto st_create = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    if (st_create != cudaSuccess) {
        throw std::runtime_error(std::string("cudaEventCreateWithFlags failed: ") +
                                 cudaGetErrorString(st_create));
    }
    const auto st_rec = cudaEventRecord(ev, stream.stream());
    if (st_rec != cudaSuccess) {
        cudaEventDestroy(ev);
        throw std::runtime_error(std::string("cudaEventRecord failed: ") + cudaGetErrorString(st_rec));
    }
    copy_state.events.emplace_back(ev, dev_idx);
}

// Record completion events on all streams that received work, so that
// `ready()`/`get()`/destructor can wait for all enqueued copies to complete.
// src_streams_used is populated by synchronize_source_streams (unique keys).
static void record_completion_events(CopyState& copy_state) {
    if (copy_state.target_stream_used && copy_state.target_stream.has_value()) {
        const auto stream = *copy_state.target_stream;
        record_event_on_stream(copy_state, stream, static_cast<int>(stream.device_index()));
    }
    for (int dev_idx : copy_state.src_streams_used) {
        record_event_on_stream(copy_state, copy_state.src_streams.at(dev_idx), dev_idx);
    }
}

// Orchestrate the full copy scheduling:
// - compute packing plan
// - allocate + fill CPU staging buffers
// - enqueue packed transfer (optional)
// - synchronize all source CUDA streams (single sync point for D2H / D2D)
// - enqueue remaining per-tensor transfers
// - record completion events on streams that received work
//
// Uses the CUDA streams captured at call time (stored in CopyState) to ensure correct
// ordering with respect to the user's preceding GPU operations.
//
// Called without the GIL (either on a background pool thread or under gil_scoped_release).
static void schedule_copies(CopyState& copy_state) {
    copy_state.outputs.assign(copy_state.inputs.size(), at::Tensor());
    copy_state.pinned_buffers.assign(copy_state.inputs.size(), at::Tensor());
    copy_state.packed_cpu_chunks_full.clear();
    copy_state.packed_cpu_chunks.clear();
    copy_state.events.clear();
    copy_state.target_stream_used = false;
    copy_state.src_streams_used.clear();

    PackPlan pack_plan = compute_pack_plan(copy_state);

    allocate_staging_buffers(copy_state, pack_plan);
    fill_cpu_staging_buffers(copy_state, pack_plan);

    if (pack_plan.enabled) {
        enqueue_packed_transfer(copy_state, pack_plan);
    }
    synchronize_source_streams(copy_state);
    enqueue_per_tensor_transfers(copy_state, pack_plan);
    record_completion_events(copy_state);
}

template <typename F>
static void for_each_cuda_event(const std::vector<CudaEvent>& events, F&& action) {
    for (const auto& e : events) {
        if (e.ev == nullptr) {
            continue;
        }
        c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(e.device_index));
        action(e.ev);
    }
}

template <typename F>
static bool for_each_cuda_event_while_action_succeeds(const std::vector<CudaEvent>& events, F&& action) {
    for (const auto& e : events) {
        if (e.ev == nullptr) {
            continue;
        }
        c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(e.device_index));
        if (!action(e.ev)) {
            return false;
        }
    }
    return true;
}

}  // namespace

// Python-facing handle representing an in-flight copy.
//
// Owns:
// - a PyTree describing the output structure,
// - a shared CopyState containing inputs/staging/outputs/events and a completion future.
//
// Semantics:
// - `ready()` is non-blocking (polls completion future + CUDA events)
// - `get()` blocks until submission finishes and CUDA work completes, then reconstructs the output
// - destructor is conservative: waits for completion and best-effort syncs CUDA to keep lifetime safe
class AsyncCopyHandle {
   public:
    AsyncCopyHandle(Node root, std::shared_ptr<CopyState> copy_state)
        : root_(std::move(root)), copy_state_(std::move(copy_state)) {}

    AsyncCopyHandle(const AsyncCopyHandle&) = delete;
    AsyncCopyHandle& operator=(const AsyncCopyHandle&) = delete;
    AsyncCopyHandle(AsyncCopyHandle&&) = default;
    AsyncCopyHandle& operator=(AsyncCopyHandle&&) = default;

    ~AsyncCopyHandle() {
        // Best-effort cleanup; must not throw.
        cleanup_no_throw();

        // Ensure Python refcounts are decremented with the GIL held.
        if (Py_IsInitialized()) {
            py::gil_scoped_acquire gil;
            // Drop any Python references held by the PyTree.
            root_ = Node();
        }
        // tensors + cuda events clean up without requiring GIL.
    }

    bool ready() const {
        if (!copy_state_ || !copy_state_->done.valid()) {
            return false;
        }
        if (copy_state_->done.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            return false;
        }
        rethrow_if_async_failed(*copy_state_);
        const bool out = cuda_events_ready(copy_state_->events);
        return out;
    }

    py::object get() {
        {
            py::gil_scoped_release nogil;
            wait_for_submission(*copy_state_);
            rethrow_if_async_failed(*copy_state_);
            cuda_events_sync_or_throw(copy_state_->events);
        }

        return build_output(root_, copy_state_->outputs);
    }

   private:
    template <typename PySeq>
    static py::object build_output_sequence(const Node& n, const std::vector<at::Tensor>& outputs) {
        PySeq out(static_cast<ssize_t>(n.seq.size()));
        for (size_t i = 0; i < n.seq.size(); ++i) {
            out[static_cast<ssize_t>(i)] = build_output(n.seq[i], outputs);
        }
        return out;
    }

    static py::object build_output(const Node& n, const std::vector<at::Tensor>& outputs) {
        switch (n.kind) {
            case NodeKind::List: {
                py::object out = build_output_sequence<py::list>(n, outputs);
                return out;
            }
            case NodeKind::Tuple: {
                py::object out = build_output_sequence<py::tuple>(n, outputs);
                return out;
            }
            case NodeKind::Dict: {
                py::dict out;
                for (const auto& kv : n.items) {
                    out[kv.first] = build_output(kv.second, outputs);
                }
                return out;
            }
            case NodeKind::TensorLeaf: {
                if (n.tensor_idx >= outputs.size()) {
                    throw std::runtime_error("Internal error: TensorLeaf index out of range.");
                }
                return py::cast(outputs[n.tensor_idx]);
            }
            case NodeKind::Passthrough:
            default:
                return n.obj;
        }
    }

    static void wait_for_submission(const CopyState& cs) {
        if (cs.done.valid()) {
            cs.done.wait();
        }
    }

    static void rethrow_if_async_failed(const CopyState& cs) {
        if (cs.exc) {
            std::rethrow_exception(cs.exc);
        }
    }

    static bool cuda_events_ready(const std::vector<CudaEvent>& events) {
        const bool out = for_each_cuda_event_while_action_succeeds(events, [](cudaEvent_t ev) {
            const auto st = cudaEventQuery(ev);
            if (st == cudaErrorNotReady) {
                cudaGetLastError();  // clear sticky error
                return false;
            }
            if (st != cudaSuccess) {
                throw std::runtime_error(std::string("cudaEventQuery failed: ") + cudaGetErrorString(st));
            }
            return true;
        });
        return out;
    }

    static void cuda_events_sync_or_throw(const std::vector<CudaEvent>& events) {
        for_each_cuda_event(events, [](cudaEvent_t ev) {
            const auto st = cudaEventSynchronize(ev);
            if (st != cudaSuccess) {
                throw std::runtime_error(std::string("cudaEventSynchronize failed: ") +
                                         cudaGetErrorString(st));
            }
        });
    }

    void cleanup_no_throw() noexcept {
        if (!copy_state_) {
            return;
        }
        try {
            wait_for_submission(*copy_state_);
        } catch (...) {
            // swallow
        }
        try {
            for_each_cuda_event(copy_state_->events, [](cudaEvent_t ev) {
                (void)cudaEventSynchronize(ev);
                (void)cudaGetLastError();
            });
        } catch (...) {
            // swallow
        }
    }

    Node root_;
    std::shared_ptr<CopyState> copy_state_;
};

// Core implementation behind the pybind wrapper.
//
// Responsibilities:
// - traverse Python structure under the GIL using the provided conversion context (numpy/torch cache)
// - create CopyState and store user flags
// - either schedule immediately without the GIL, or enqueue scheduling to the worker pool
// - return an AsyncCopyHandle that can be waited via ready()/get()
static AsyncCopyHandle start_copy_impl(py::object data, const std::string& device, bool use_pinned_staging,
                                       bool use_background_thread, bool pack_cpu_tensors,
                                       int64_t min_packed_alignment_bytes, int64_t max_packed_chunk_bytes,
                                       const PyConversionCtx& ctx) {
    auto copy_state = std::make_shared<CopyState>();
    Node root = traverse_build_tree_impl(data, ctx, copy_state->inputs);
    copy_state->target_device = parse_device(device);
    copy_state->use_pinned_staging = use_pinned_staging;
    copy_state->pack_cpu_tensors = pack_cpu_tensors;
    copy_state->min_packed_alignment_bytes = std::max<int64_t>(1, min_packed_alignment_bytes);
    copy_state->max_packed_chunk_bytes = std::max<int64_t>(1, max_packed_chunk_bytes);

    // Capture the user's current CUDA streams while still on the caller's thread.
    // These are used later (potentially on a background thread) so that copy work is
    // correctly ordered with the user's preceding GPU operations.
    if (copy_state->target_device.is_cuda()) {
        const auto dev_idx = (copy_state->target_device.index() >= 0)
                                 ? copy_state->target_device.index()
                                 : static_cast<c10::DeviceIndex>(at::cuda::current_device());
        copy_state->target_stream = at::cuda::getCurrentCUDAStream(dev_idx);
    }
    // For each unique source CUDA device, capture the current stream.  When the target is
    // also CUDA (D2D), record an event to pin the cross-device sync point to this moment.
    const bool need_capture_events = copy_state->target_device.is_cuda();
    for (const auto& t : copy_state->inputs) {
        if (t.device().is_cuda()) {
            const int dev = static_cast<int>(t.device().index());
            if (copy_state->src_streams.find(dev) == copy_state->src_streams.end()) {
                const auto stream = at::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(dev));
                copy_state->src_streams.emplace(dev, stream);

                if (need_capture_events) {
                    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
                    cudaEvent_t ev = nullptr;
                    auto st = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
                    if (st != cudaSuccess) {
                        throw std::runtime_error(std::string("cudaEventCreateWithFlags failed: ") +
                                                 cudaGetErrorString(st));
                    }
                    st = cudaEventRecord(ev, stream.stream());
                    if (st != cudaSuccess) {
                        cudaEventDestroy(ev);
                        throw std::runtime_error(std::string("cudaEventRecord failed: ") +
                                                 cudaGetErrorString(st));
                    }
                    copy_state->src_capture_events.try_emplace(dev, ev, dev);
                }
            }
        }
    }

    if (use_background_thread) {
        // Submit staging + copy enqueue work to the shared pool (no Python API usage).
        auto promise = std::make_shared<std::promise<void>>();
        copy_state->done = promise->get_future().share();
        CopyThreadPool::instance().enqueue([copy_state, promise]() {
            try {
                schedule_copies(*copy_state);
            } catch (...) {
                copy_state->exc = std::current_exception();
            }
            try {
                promise->set_value();
            } catch (...) {
                // swallow
            }
        });
    } else {
        py::gil_scoped_release nogil;
        try {
            schedule_copies(*copy_state);
        } catch (...) {
            copy_state->exc = std::current_exception();
        }
        auto promise = std::make_shared<std::promise<void>>();
        copy_state->done = promise->get_future().share();
        promise->set_value();
        if (copy_state->exc) {
            std::rethrow_exception(copy_state->exc);
        }
    }

    AsyncCopyHandle out(std::move(root), std::move(copy_state));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Python-owned cache of expensive-to-look-up objects used during traversal.
    // Storing this on the module keeps lifetime tied to Python interpreter shutdown (no C++ static py::object issues).
    py::dict py_cache;
    py::object numpy = py::module_::import("numpy");
    py::object torch = py::module_::import("torch");
    py_cache["numpy_ndarray_type"] = numpy.attr("ndarray");
    py_cache["numpy_ascontiguousarray"] = numpy.attr("ascontiguousarray");
    py_cache["torch_from_numpy"] = torch.attr("from_numpy");
    m.attr("_py_cache") = py_cache;

    py::class_<AsyncCopyHandle>(m, "AsyncCopyHandle")
        .def("get", &AsyncCopyHandle::get, "Wait for transfers (if any) and return the copied structure.")
        .def("ready", &AsyncCopyHandle::ready, "Return True if all enqueued async copies have completed.");

    // Pybind entrypoint wrapper.
    //
    // Responsibilities:
    // - traverse Python structure under the GIL (build skeleton + collect inputs + setters),
    //   converting numpy.ndarray leaves to tensors using the module-owned cache
    // - create CopyState and store user flags
    // - either schedule immediately without the GIL, or enqueue scheduling to the worker pool
    // - return an AsyncCopyHandle that can be waited via ready()/get()
    m.def(
        "start_copy",
        [py_cache](py::object data, const std::string& device, bool use_pinned_staging,
                   bool use_background_thread, bool pack_cpu_tensors, int64_t min_packed_alignment_bytes,
                   int64_t max_packed_chunk_bytes) {
            const PyConversionCtx ctx = make_py_conversion_ctx_from_cache(py_cache);
            return start_copy_impl(std::move(data), device, use_pinned_staging, use_background_thread,
                                   pack_cpu_tensors, min_packed_alignment_bytes, max_packed_chunk_bytes, ctx);
        },
        "Start an async copy of a nested list/tuple/dict of tensors to the given device (string).",
        py::arg("data"), py::arg("device"), py::arg("use_pinned_staging") = true,
        py::arg("use_background_thread") = true, py::arg("pack_cpu_tensors") = true,
        py::arg("min_packed_alignment_bytes") = 16, py::arg("max_packed_chunk_bytes") = 32 * 1024 * 1024);
}
