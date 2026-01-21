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

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <exception>
#include <iostream>

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
