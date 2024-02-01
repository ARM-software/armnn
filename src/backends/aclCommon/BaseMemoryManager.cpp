//
// Copyright © 2017-2024 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "BaseMemoryManager.hpp"

#if defined(ARMCOMPUTENEON_ENABLED) || defined(ARMCOMPUTECL_ENABLED) || defined(ARMCOMPUTEGPUFSA_ENABLED)
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/OffsetLifetimeManager.h"
#endif


namespace armnn
{

#if defined(ARMCOMPUTENEON_ENABLED) || defined(ARMCOMPUTECL_ENABLED) || defined(ARMCOMPUTEGPUFSA_ENABLED)
BaseMemoryManager::BaseMemoryManager(std::shared_ptr<arm_compute::IAllocator> alloc,
                                     MemoryAffinity memoryAffinity)
{
    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(alloc, "A null allocator has been passed to BaseMemoryManager.");
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
        lifetimeManager = std::make_shared<arm_compute::BlobLifetimeManager>();
    }
    else
    {
        lifetimeManager = std::make_shared<arm_compute::OffsetLifetimeManager>();
    }

    auto poolManager   = std::make_shared<arm_compute::PoolManager>();
    auto memoryManager = std::make_shared<arm_compute::MemoryManagerOnDemand>(lifetimeManager, poolManager);

    return memoryManager;
}

void BaseMemoryManager::Acquire()
{
    static const size_t s_NumPools = 1;

    // Allocate memory pools for intra-layer memory manager
    m_IntraLayerMemoryMgr->populate(*m_Allocator, s_NumPools);

    // Allocate memory pools for inter-layer memory manager
    m_InterLayerMemoryMgr->populate(*m_Allocator, s_NumPools);

    // Acquire inter-layer memory group. NOTE: This has to come after allocating the pools
    m_InterLayerMemoryGroup->acquire();
}

void BaseMemoryManager::Release()
{
    // Release inter-layer memory group. NOTE: This has to come before releasing the pools
    m_InterLayerMemoryGroup->release();

    // Release memory pools managed by intra-layer memory manager
    m_IntraLayerMemoryMgr->clear();

    // Release memory pools managed by inter-layer memory manager
    m_InterLayerMemoryMgr->clear();
}
#else
void BaseMemoryManager::Acquire()
{
    // No-op if neither NEON nor CL enabled
}

void BaseMemoryManager::Release()
{
    // No-op if neither NEON nor CL enabled
}
#endif

#if defined(ARMCOMPUTENEON_ENABLED)
std::shared_ptr<arm_compute::IMemoryGroup>
NeonMemoryManager::CreateMemoryGroup(const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
{
    return std::make_shared<arm_compute::MemoryGroup>(memoryManager);
}
#endif

#if defined(ARMCOMPUTECL_ENABLED)
std::shared_ptr<arm_compute::IMemoryGroup>
ClMemoryManager::CreateMemoryGroup(const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
{
    return std::make_shared<arm_compute::MemoryGroup>(memoryManager);
}
#endif

#if defined(ARMCOMPUTEGPUFSA_ENABLED)
std::shared_ptr<arm_compute::IMemoryGroup>
GpuFsaMemoryManager::CreateMemoryGroup(const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
{
    return std::make_shared<arm_compute::MemoryGroup>(memoryManager);
}
#endif

}
