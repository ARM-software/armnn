//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "BaseMemoryManager.hpp"

#if defined(ARMCOMPUTENEON_ENABLED) || defined(ARMCOMPUTECL_ENABLED)
#include "memory/BlobLifetimeManager.hpp"
#include "memory/PoolManager.hpp"
#include "memory/OffsetLifetimeManager.hpp"
#endif

#include <boost/polymorphic_cast.hpp>

namespace armnn
{

#if defined(ARMCOMPUTENEON_ENABLED) || defined(ARMCOMPUTECL_ENABLED)
BaseMemoryManager::BaseMemoryManager(std::unique_ptr<arm_compute::IAllocator> alloc,
                                     MemoryAffinity memoryAffinity)
{
    // (Re)create the memory manager components.
    m_Allocator = std::move(alloc);

    m_IntraLayerMemoryMgr = CreateArmComputeMemoryManager(memoryAffinity);
    m_InterLayerMemoryMgr = CreateArmComputeMemoryManager(memoryAffinity);
}

std::shared_ptr<arm_compute::MemoryManagerOnDemand>
BaseMemoryManager::CreateArmComputeMemoryManager(MemoryAffinity memoryAffinity)
{
    std::shared_ptr<arm_compute::ILifetimeManager> lifetimeManager = nullptr;

    if (memoryAffinity == MemoryAffinity::Buffer)
    {
        lifetimeManager = std::make_shared<BlobLifetimeManager>();
    }
    else
    {
        lifetimeManager = std::make_shared<OffsetLifetimeManager>();
    }

    auto poolManager   = std::make_shared<PoolManager>();
    auto memoryManager = std::make_shared<arm_compute::MemoryManagerOnDemand>(lifetimeManager, poolManager);

    // Set allocator that the memory manager will use
    memoryManager->set_allocator(m_Allocator.get());

    return memoryManager;
}

void BaseMemoryManager::FinalizeMemoryManager(arm_compute::MemoryManagerOnDemand& memoryManager)
{
    // Number of pools that the manager will create. This specifies how many layers you want to run in parallel
    memoryManager.set_num_pools(1);

    // Finalize the memory manager. (Validity checks, memory allocations, etc)
    memoryManager.finalize();
}

void BaseMemoryManager::Finalize()
{
    BOOST_ASSERT(m_IntraLayerMemoryMgr);
    FinalizeMemoryManager(*m_IntraLayerMemoryMgr.get());

    BOOST_ASSERT(m_InterLayerMemoryMgr);
    FinalizeMemoryManager(*m_InterLayerMemoryMgr.get());
}

void BaseMemoryManager::Acquire()
{
    // Allocate memory pools for intra-layer memory manager
    BOOST_ASSERT(m_IntraLayerMemoryMgr);
    IPoolManager* poolManager = boost::polymorphic_downcast<IPoolManager*>(m_IntraLayerMemoryMgr->pool_manager());
    BOOST_ASSERT(poolManager);
    poolManager->AllocatePools();

    // Allocate memory pools for inter-layer memory manager
    BOOST_ASSERT(m_InterLayerMemoryMgr);
    poolManager = boost::polymorphic_downcast<IPoolManager*>(m_InterLayerMemoryMgr->pool_manager());
    BOOST_ASSERT(poolManager);
    poolManager->AllocatePools();

    // Acquire inter-layer memory group. NOTE: This has to come after allocating the pools
    BOOST_ASSERT(m_InterLayerMemoryGroup);
    m_InterLayerMemoryGroup->acquire();
}

void BaseMemoryManager::Release()
{
    // Release inter-layer memory group. NOTE: This has to come before releasing the pools
    BOOST_ASSERT(m_InterLayerMemoryGroup);
    m_InterLayerMemoryGroup->release();

    // Release memory pools managed by intra-layer memory manager
    BOOST_ASSERT(m_IntraLayerMemoryMgr);
    IPoolManager* poolManager = boost::polymorphic_downcast<IPoolManager*>(m_IntraLayerMemoryMgr->pool_manager());
    BOOST_ASSERT(poolManager);
    poolManager->ReleasePools();

    // Release memory pools managed by inter-layer memory manager
    BOOST_ASSERT(m_InterLayerMemoryMgr);
    poolManager = boost::polymorphic_downcast<IPoolManager*>(m_InterLayerMemoryMgr->pool_manager());
    BOOST_ASSERT(poolManager);
    poolManager->ReleasePools();
}
#endif

#ifdef ARMCOMPUTENEON_ENABLED
std::shared_ptr<arm_compute::IMemoryGroup>
NeonMemoryManager::CreateMemoryGroup(const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
{
    return std::make_shared<arm_compute::MemoryGroup>(memoryManager);
}
#endif

#ifdef ARMCOMPUTECL_ENABLED
std::shared_ptr<arm_compute::IMemoryGroup>
ClMemoryManager::CreateMemoryGroup(const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
{
    return std::make_shared<arm_compute::CLMemoryGroup>(memoryManager);
}
#endif

}
