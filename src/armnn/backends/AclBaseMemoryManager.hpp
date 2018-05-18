//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "WorkloadFactory.hpp"

#if ARMCOMPUTENEON_ENABLED || ARMCOMPUTECL_ENABLED
#include "arm_compute/runtime/IAllocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"

#include <memory>
#endif

namespace armnn
{

// ARM Compute Base Memory Manager
class AclBaseMemoryManager
{
public:

    AclBaseMemoryManager() { }
    virtual ~AclBaseMemoryManager() { }

#if ARMCOMPUTENEON_ENABLED || ARMCOMPUTECL_ENABLED
    AclBaseMemoryManager(std::unique_ptr<arm_compute::IAllocator> alloc);

    void Finalize();

    std::shared_ptr<arm_compute::MemoryManagerOnDemand>& Get() { return m_IntraLayerMemoryMgr; }

protected:

    mutable std::unique_ptr<arm_compute::IAllocator>            m_Allocator;
    mutable std::shared_ptr<arm_compute::BlobLifetimeManager>   m_IntraLayerLifetimeMgr;
    mutable std::shared_ptr<arm_compute::PoolManager>           m_IntraLayerPoolMgr;
    mutable std::shared_ptr<arm_compute::MemoryManagerOnDemand> m_IntraLayerMemoryMgr;
#endif

};

} //namespace armnn
