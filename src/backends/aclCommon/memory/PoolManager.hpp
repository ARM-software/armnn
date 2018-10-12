//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "IPoolManager.hpp"

#include <arm_compute/runtime/IMemoryPool.h>
#include <arm_compute/core/Error.h>

#include <support/Mutex.h>
#include <support/Semaphore.h>

#include <cstddef>
#include <list>
#include <memory>

namespace armnn
{

class PoolManager : public IPoolManager
{
public:
    PoolManager();

    PoolManager(const PoolManager &) = delete;

    PoolManager &operator=(const PoolManager &) = delete;

    PoolManager(PoolManager &&) = default;

    PoolManager &operator=(PoolManager &&) = default;

    arm_compute::IMemoryPool *lock_pool() override;
    void unlock_pool(arm_compute::IMemoryPool *pool) override;
    void register_pool(std::unique_ptr<arm_compute::IMemoryPool> pool) override;
    size_t num_pools() const override;

    void AllocatePools() override;
    void ReleasePools() override;

private:
    /// List of free pools
    std::list<std::unique_ptr<arm_compute::IMemoryPool>> m_FreePools;

    /// List of occupied pools
    std::list<std::unique_ptr<arm_compute::IMemoryPool>> m_OccupiedPools;

    /// Semaphore to control the queues
    std::unique_ptr<arm_compute::Semaphore> m_Semaphore;

    /// Mutex to control access to the queues
    mutable arm_compute::Mutex m_Mutex;
};

} // namespace armnn