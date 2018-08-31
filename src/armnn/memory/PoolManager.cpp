//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "IMemoryPool.hpp"
#include "PoolManager.hpp"

#include "boost/assert.hpp"
#include "boost/polymorphic_cast.hpp"

#include <algorithm>

namespace armnn
{

PoolManager::PoolManager()
        : m_FreePools()
        , m_OccupiedPools()
        , m_Semaphore()
        , m_Mutex()
{}

arm_compute::IMemoryPool *PoolManager::lock_pool()
{
    BOOST_ASSERT_MSG(!(m_FreePools.empty() && m_OccupiedPools.empty()), "Haven't setup any pools");

    m_Semaphore->wait();
    std::lock_guard<arm_compute::Mutex> lock(m_Mutex);

    BOOST_ASSERT_MSG(!m_FreePools.empty(), "Empty pool must exist as semaphore has been signalled");
    m_OccupiedPools.splice(std::begin(m_OccupiedPools), m_FreePools, std::begin(m_FreePools));

    return m_OccupiedPools.front().get();
}

void PoolManager::unlock_pool(arm_compute::IMemoryPool *pool)
{
    BOOST_ASSERT_MSG(!(m_FreePools.empty() && m_OccupiedPools.empty()), "Haven't setup any pools!");

    std::lock_guard<arm_compute::Mutex> lock(m_Mutex);

    auto it = std::find_if(
            std::begin(m_OccupiedPools),
            std::end(m_OccupiedPools),
            [pool](const std::unique_ptr<arm_compute::IMemoryPool> &poolIterator)
            {
                return poolIterator.get() == pool;
            }
    );

    BOOST_ASSERT_MSG(it != std::end(m_OccupiedPools), "Pool to be unlocked couldn't be found");
    m_FreePools.splice(std::begin(m_FreePools), m_OccupiedPools, it);
    m_Semaphore->signal();
}

void PoolManager::register_pool(std::unique_ptr<arm_compute::IMemoryPool> pool)
{
    std::lock_guard<arm_compute::Mutex> lock(m_Mutex);
    BOOST_ASSERT_MSG(m_OccupiedPools.empty(), "All pools should be free in order to register a new one");

    // Set pool
    m_FreePools.push_front(std::move(pool));

    // Update semaphore
    m_Semaphore = std::make_unique<arm_compute::Semaphore>(m_FreePools.size());
}

size_t PoolManager::num_pools() const
{
    std::lock_guard<arm_compute::Mutex> lock(m_Mutex);

    return m_FreePools.size() + m_OccupiedPools.size();
}

void PoolManager::AllocatePools()
{
    std::lock_guard<arm_compute::Mutex> lock(m_Mutex);

    for (auto& pool : m_FreePools)
    {
        boost::polymorphic_downcast<IMemoryPool*>(pool.get())->AllocatePool();
    }

    for (auto& pool : m_OccupiedPools)
    {
        boost::polymorphic_downcast<IMemoryPool*>(pool.get())->AllocatePool();
    }
}

void PoolManager::ReleasePools()
{
    std::lock_guard<arm_compute::Mutex> lock(m_Mutex);

    for (auto& pool : m_FreePools)
    {
        boost::polymorphic_downcast<IMemoryPool*>(pool.get())->ReleasePool();
    }

    for (auto& pool : m_OccupiedPools)
    {
        boost::polymorphic_downcast<IMemoryPool*>(pool.get())->ReleasePool();
    }
}

} //namespace armnn