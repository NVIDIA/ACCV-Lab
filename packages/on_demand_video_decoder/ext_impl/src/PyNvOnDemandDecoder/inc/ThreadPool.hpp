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

#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

class ThreadRunner {
   public:
    ThreadRunner() : stopFlag(false), busy(false), hasException(false), exceptionPtr(nullptr) {
        worker = std::thread([this]() { this->threadLoop(); });
    }

    ~ThreadRunner() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            stopFlag = true;
            cv.notify_all();
        }
        if (worker.joinable()) {
            worker.join();
        }
    }

    ThreadRunner(const ThreadRunner&) = delete;
    ThreadRunner& operator=(const ThreadRunner&) = delete;

    ThreadRunner(ThreadRunner&& other) noexcept {
        std::unique_lock<std::mutex> lock(other.mtx);
        worker = std::move(other.worker);
        stopFlag = other.stopFlag;
        busy = other.busy;
        hasException = other.hasException;
        exceptionPtr = other.exceptionPtr;
        tasks = std::move(other.tasks);
    }

    ThreadRunner& operator=(ThreadRunner&& other) noexcept {
        if (this != &other) {
            {
                std::unique_lock<std::mutex> lock(mtx);
                stopFlag = true;
            }
            cv.notify_one();
            if (worker.joinable()) {
                worker.join();
            }
            std::unique_lock<std::mutex> lock(other.mtx);
            worker = std::move(other.worker);
            stopFlag = other.stopFlag;
            busy = other.busy;
            hasException = other.hasException;
            exceptionPtr = other.exceptionPtr;
            tasks = std::move(other.tasks);
        }
        return *this;
    }

    template <typename F, typename... Args>
    void start(F&& f, Args&&... args) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            tasks.emplace(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        }
        cv.notify_one();
    }

    /**
     * @brief Wait for all tasks to complete and rethrow any captured exception.
     * 
     * If a task threw an exception, this method will rethrow the original exception
     * with its full type and message information.
     */
    void join() {
        std::unique_lock<std::mutex> lock(mtx);
        cvFinished.wait(lock, [this]() { return tasks.empty() && !busy; });
        if (hasException) {
            hasException = false;  // Reset for next use
            std::exception_ptr ptr = exceptionPtr;
            exceptionPtr = nullptr;  // Reset for next use
            if (ptr) {
                std::rethrow_exception(ptr);  // Rethrow original exception with full info
            }
            throw std::runtime_error("Thread task failed with unknown exception");
        }
    }

    /**
     * @brief Force join by clearing pending tasks and waiting for current task.
     * 
     * This clears the task queue and waits for any running task to complete,
     * then resets the exception state.
     */
    void force_join() {
        std::unique_lock<std::mutex> lock(mtx);
        // Clear all pending tasks
        while (!tasks.empty()) {
            tasks.pop();
        }
        // Wait for current task to finish (if any)
        cvFinished.wait(lock, [this]() { return !busy; });
        // Reset exception state
        hasException = false;
        exceptionPtr = nullptr;
    }

   private:
    void threadLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this]() { return stopFlag || !tasks.empty(); });
                if (stopFlag && tasks.empty()) return;
                task = std::move(tasks.front());
                tasks.pop();
                busy = true;
            }

            try {
                task();  // execute task
            } catch (const std::exception& e) {
                // Capture exception with full information for later rethrow
                std::cerr << "[ThreadRunner] Exception caught: " << e.what() << std::endl;
                exceptionPtr = std::current_exception();
                hasException = true;
            } catch (...) {
                // Capture unknown exception
                std::cerr << "[ThreadRunner] Unknown exception caught" << std::endl;
                exceptionPtr = std::current_exception();
                hasException = true;
            }

            {
                std::unique_lock<std::mutex> lock(mtx);
                busy = false;  // reset status
            }
            cvFinished.notify_all();
        }
    }

    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    std::condition_variable cvFinished;
    std::queue<std::function<void()>> tasks;
    bool stopFlag;
    bool busy;
    bool hasException;
    std::exception_ptr exceptionPtr;  // Store original exception for rethrow
};

class ThreadPool {
   public:
    ThreadPool() : ThreadPool(available_cpu_count()) {}
    explicit ThreadPool(size_t max_worker_count) : max_worker_count(std::max<size_t>(1, max_worker_count)) {}

    static size_t available_cpu_count() {
#if defined(__linux__)
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set) == 0) {
            const size_t cpu_count = CPU_COUNT(&cpu_set);
            if (cpu_count > 0) {
                return cpu_count;
            }
        }
#endif
        const size_t cpu_count = std::thread::hardware_concurrency();
        return cpu_count == 0 ? 1 : cpu_count;
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    template <typename Func>
    void submit_indexed(size_t task_count, Func&& task) {
        wait_all();
        if (task_count == 0) {
            return;
        }

        const size_t worker_count = std::min(task_count, max_worker_count);
        while (workers.size() < worker_count) {
            workers.emplace_back(std::make_unique<ThreadRunner>());
        }

        using Task = typename std::decay<Func>::type;
        auto task_ptr = std::make_shared<Task>(std::forward<Func>(task));
        auto next_index = std::make_shared<std::atomic<size_t>>(0);
        active_state = std::make_shared<TaskState>(task_count);
        active_worker_count = 0;

        try {
            for (; active_worker_count < worker_count; ++active_worker_count) {
                workers[active_worker_count]->start(
                    [task_ptr, next_index, state = active_state, task_count]() {
                        while (true) {
                            const size_t index = next_index->fetch_add(1);
                            if (index >= task_count) {
                                return;
                            }
                            try {
                                (*task_ptr)(index);
                            } catch (...) {
                                state->exceptions[index] = std::current_exception();
                            }
                        }
                    });
            }
        } catch (...) {
            active_state->submission_exception = std::current_exception();
            wait_all();
        }
    }

    template <typename Func>
    void run_indexed(size_t task_count, Func&& task) {
        submit_indexed(task_count, std::forward<Func>(task));
        wait_all();
    }

    void wait_all() {
        std::exception_ptr first_exception;
        for (size_t index = 0; index < active_worker_count; ++index) {
            try {
                workers[index]->join();
            } catch (...) {
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        }

        auto state = std::move(active_state);
        active_worker_count = 0;
        if (state) {
            if (!first_exception) {
                first_exception = state->submission_exception;
            }
            for (const auto& exception : state->exceptions) {
                if (!first_exception && exception) {
                    first_exception = exception;
                }
            }
        }

        if (first_exception) {
            std::rethrow_exception(first_exception);
        }
    }

    void force_join() {
        for (auto& worker : workers) {
            worker->force_join();
        }
        active_worker_count = 0;
        active_state.reset();
    }

   private:
    struct TaskState {
        explicit TaskState(size_t task_count) : exceptions(task_count) {}

        std::vector<std::exception_ptr> exceptions;
        std::exception_ptr submission_exception;
    };

    const size_t max_worker_count;
    std::vector<std::unique_ptr<ThreadRunner>> workers;
    size_t active_worker_count = 0;
    std::shared_ptr<TaskState> active_state;
};

inline void wait_all(ThreadPool& first, ThreadPool& second) {
    std::exception_ptr first_exception;
    try {
        first.wait_all();
    } catch (...) {
        first_exception = std::current_exception();
    }
    try {
        second.wait_all();
    } catch (...) {
        if (!first_exception) {
            first_exception = std::current_exception();
        }
    }
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
}
