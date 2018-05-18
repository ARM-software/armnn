//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "AclBaseMemoryManager.hpp"

namespace armnn
{

#if ARMCOMPUTENEON_ENABLED || ARMCOMPUTECL_ENABLED
AclBaseMemoryManager::AclBaseMemoryManager(std::unique_ptr<arm_compute::IAllocator> alloc)
{
    // (re)create the memory manager components
    m_Allocator = std::move(alloc);
    m_IntraLayerLifetimeMgr = std::make_shared<arm_compute::BlobLifetimeManager>();
    m_IntraLayerPoolMgr     = std::make_shared<arm_compute::PoolManager>();
    m_IntraLayerMemoryMgr   = std::make_shared<arm_compute::MemoryManagerOnDemand>(m_IntraLayerLifetimeMgr,
                                                                                   m_IntraLayerPoolMgr);
}

void AclBaseMemoryManager::Finalize()
{
    // Set allocator that the memory manager will use
    m_IntraLayerMemoryMgr->set_allocator(m_Allocator.get());
    // Number of pools that the manager will create. This specifies how many layers you want to run in parallel
    m_IntraLayerMemoryMgr->set_num_pools(1);
    // Finalize the memory manager. (Validity checks, memory allocations, etc)
    m_IntraLayerMemoryMgr->finalize();
}
#endif

}
