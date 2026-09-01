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

#include "ThreadPool.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace {

class ThreadBarrier {
   public:
    explicit ThreadBarrier(size_t participant_count) : participant_count_(participant_count) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++arrived_count_;
        if (arrived_count_ == participant_count_) {
            condition_.notify_all();
            return;
        }
        if (!condition_.wait_for(lock, std::chrono::seconds(5),
                                 [this]() { return arrived_count_ == participant_count_; })) {
            throw std::runtime_error("Timed out waiting for thread-pool workers");
        }
    }

   private:
    const size_t participant_count_;
    size_t arrived_count_ = 0;
    std::mutex mutex_;
    std::condition_variable condition_;
};

void update_max(std::atomic<size_t>& maximum, size_t value) {
    size_t current = maximum.load();
    while (current < value && !maximum.compare_exchange_weak(current, value)) {
    }
}

TEST(ThreadPoolTest, ZeroTasksDoNotInvokeCallable) {
    ThreadPool pool(4);
    std::atomic<size_t> call_count{0};

    pool.run_indexed(0, [&call_count](size_t) { ++call_count; });

    EXPECT_EQ(call_count.load(), 0U);
}

TEST(ThreadPoolTest, RunsEveryIndexExactlyOnce) {
    ThreadPool pool(4);
    std::vector<std::atomic<size_t>> visit_counts(257);
    for (auto& visit_count : visit_counts) {
        visit_count.store(0);
    }

    pool.run_indexed(visit_counts.size(), [&visit_counts](size_t index) { ++visit_counts[index]; });

    for (size_t index = 0; index < visit_counts.size(); ++index) {
        EXPECT_EQ(visit_counts[index].load(), 1U) << "index " << index;
    }
}

TEST(ThreadPoolTest, DoesNotExceedWorkerLimit) {
    constexpr size_t worker_count = 3;
    ThreadPool pool(worker_count);
    ThreadBarrier barrier(worker_count);
    std::atomic<size_t> active_count{0};
    std::atomic<size_t> maximum_active_count{0};

    pool.run_indexed(12, [&](size_t index) {
        const size_t active = ++active_count;
        update_max(maximum_active_count, active);
        if (index < worker_count) {
            barrier.arrive_and_wait();
        }
        --active_count;
    });

    EXPECT_EQ(maximum_active_count.load(), worker_count);
}

TEST(ThreadPoolTest, ExpandsForLargerBatches) {
    constexpr size_t worker_count = 4;
    ThreadPool pool(worker_count);
    std::atomic<size_t> first_batch_count{0};

    pool.run_indexed(1, [&first_batch_count](size_t) { ++first_batch_count; });

    ThreadBarrier barrier(worker_count);
    std::atomic<size_t> active_count{0};
    std::atomic<size_t> maximum_active_count{0};
    pool.run_indexed(worker_count, [&](size_t) {
        const size_t active = ++active_count;
        update_max(maximum_active_count, active);
        barrier.arrive_and_wait();
        --active_count;
    });

    EXPECT_EQ(first_batch_count.load(), 1U);
    EXPECT_EQ(maximum_active_count.load(), worker_count);
}

TEST(ThreadPoolTest, SubmitIndexedReturnsBeforeBlockedTasksFinish) {
    ThreadPool pool(3);
    std::promise<void> release_promise;
    std::shared_future<void> release = release_promise.get_future().share();
    std::atomic<size_t> completed_count{0};

    pool.submit_indexed(15, [&release, &completed_count](size_t) {
        release.wait();
        ++completed_count;
    });

    EXPECT_EQ(completed_count.load(), 0U);
    release_promise.set_value();
    pool.wait_all();
    EXPECT_EQ(completed_count.load(), 15U);
}

TEST(ThreadPoolTest, WaitsForAllTasksBeforeRethrowingEarliestIndexedException) {
    ThreadPool pool(4);
    std::vector<std::atomic<size_t>> visit_counts(32);
    for (auto& visit_count : visit_counts) {
        visit_count.store(0);
    }

    try {
        pool.run_indexed(visit_counts.size(), [&visit_counts](size_t index) {
            ++visit_counts[index];
            if (index == 2) {
                throw std::logic_error("exception at index 2");
            }
            if (index == 7) {
                throw std::runtime_error("exception at index 7");
            }
        });
        FAIL() << "run_indexed did not rethrow a task exception";
    } catch (const std::logic_error& error) {
        EXPECT_EQ(std::string(error.what()), "exception at index 2");
    } catch (...) {
        FAIL() << "run_indexed did not preserve the earliest exception type";
    }

    for (size_t index = 0; index < visit_counts.size(); ++index) {
        EXPECT_EQ(visit_counts[index].load(), 1U) << "index " << index;
    }
}

TEST(ThreadPoolTest, CanBeReusedAfterTaskException) {
    ThreadPool pool(2);
    EXPECT_THROW(pool.run_indexed(1, [](size_t) { throw std::runtime_error("expected failure"); }),
                 std::runtime_error);

    std::atomic<size_t> completed_count{0};
    EXPECT_NO_THROW(pool.run_indexed(11, [&completed_count](size_t) { ++completed_count; }));
    EXPECT_EQ(completed_count.load(), 11U);
}

TEST(ThreadPoolTest, SupportsMoveOnlyCallable) {
    ThreadPool pool(2);
    std::atomic<size_t> completed_count{0};
    auto marker = std::make_unique<int>(17);

    pool.run_indexed(9, [marker = std::move(marker), &completed_count](size_t) {
        if (*marker != 17) {
            throw std::runtime_error("move-only callable state was not preserved");
        }
        ++completed_count;
    });

    EXPECT_EQ(completed_count.load(), 9U);
}

TEST(ThreadPoolTest, DualPoolWaitCompletesBothPoolsBeforeRethrowing) {
    ThreadPool first(2);
    ThreadPool second(2);
    std::atomic<size_t> second_pool_completed_count{0};

    first.submit_indexed(1, [](size_t) { throw std::runtime_error("first pool failure"); });
    second.submit_indexed(23, [&second_pool_completed_count](size_t) { ++second_pool_completed_count; });

    try {
        wait_all(first, second);
        FAIL() << "wait_all did not rethrow the first pool exception";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), "first pool failure");
    } catch (...) {
        FAIL() << "wait_all did not preserve the first pool exception type";
    }
    EXPECT_EQ(second_pool_completed_count.load(), 23U);
}

#if defined(__linux__)
class AffinityGuard {
   public:
    explicit AffinityGuard(const cpu_set_t& affinity) : affinity_(affinity) {}

    ~AffinityGuard() { sched_setaffinity(0, sizeof(affinity_), &affinity_); }

    AffinityGuard(const AffinityGuard&) = delete;
    AffinityGuard& operator=(const AffinityGuard&) = delete;

   private:
    cpu_set_t affinity_;
};

TEST(ThreadPoolTest, AvailableCpuCountRespectsProcessAffinity) {
    cpu_set_t original_affinity;
    CPU_ZERO(&original_affinity);
    if (sched_getaffinity(0, sizeof(original_affinity), &original_affinity) != 0) {
        GTEST_SKIP() << "sched_getaffinity is not available";
    }
    AffinityGuard affinity_guard(original_affinity);

    int selected_cpu = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &original_affinity)) {
            selected_cpu = cpu;
            break;
        }
    }
    ASSERT_GE(selected_cpu, 0);

    cpu_set_t single_cpu_affinity;
    CPU_ZERO(&single_cpu_affinity);
    CPU_SET(selected_cpu, &single_cpu_affinity);
    if (sched_setaffinity(0, sizeof(single_cpu_affinity), &single_cpu_affinity) != 0) {
        GTEST_SKIP() << "sched_setaffinity is not permitted";
    }

    EXPECT_EQ(ThreadPool::available_cpu_count(), 1U);
}
#endif

}  // namespace
