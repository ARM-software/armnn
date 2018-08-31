//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "backends/WorkloadFactory.hpp"

#ifdef ARMCOMPUTENEON_ENABLED
#include "arm_compute/runtime/MemoryGroup.h"
#endif

#ifdef ARMCOMPUTECL_ENABLED
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#endif

#if defined(ARMCOMPUTENEON_ENABLED) || defined(ARMCOMPUTECL_ENABLED)
#include "arm_compute/runtime/IAllocator.h"
#include "arm_compute/runtime/IMemoryGroup.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#endif

namespace armnn
{

class BaseMemoryManager
{
public:
    enum class MemoryAffinity
    {
        Buffer,
        Offset
    };

    BaseMemoryManager() { }
    virtual ~BaseMemoryManager() { }

#if defined(ARMCOMPUTENEON_ENABLED) || defined(ARMCOMPUTECL_ENABLED)

    BaseMemoryManager(std::unique_ptr<arm_compute::IAllocator> alloc, MemoryAffinity memoryAffinity);

    std::shared_ptr<arm_compute::MemoryManagerOnDemand>& GetIntraLayerManager() { return m_IntraLayerMemoryMgr; }
    std::shared_ptr<arm_compute::MemoryManagerOnDemand>& GetInterLayerManager() { return m_InterLayerMemoryMgr; }
    std::shared_ptr<arm_compute::IMemoryGroup>& GetInterLayerMemoryGroup()      { return m_InterLayerMemoryGroup; }

    void Finalize();
    void Acquire();
    void Release();

protected:

    std::unique_ptr<arm_compute::IAllocator>            m_Allocator;
    std::shared_ptr<arm_compute::MemoryManagerOnDemand> m_IntraLayerMemoryMgr;
    std::shared_ptr<arm_compute::MemoryManagerOnDemand> m_InterLayerMemoryMgr;
    std::shared_ptr<arm_compute::IMemoryGroup>          m_InterLayerMemoryGroup;

    std::shared_ptr<arm_compute::MemoryManagerOnDemand> CreateArmComputeMemoryManager(MemoryAffinity memoryAffinity);

    virtual std::shared_ptr<arm_compute::IMemoryGroup>
    CreateMemoryGroup(const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager) = 0;

    void FinalizeMemoryManager(arm_compute::MemoryManagerOnDemand& memoryManager);
#endif
};

class NeonMemoryManager : public BaseMemoryManager
{
public:
    NeonMemoryManager() {}
    virtual ~NeonMemoryManager() {}

#ifdef ARMCOMPUTENEON_ENABLED
    NeonMemoryManager(std::unique_ptr<arm_compute::IAllocator> alloc, MemoryAffinity memoryAffinity)
    : BaseMemoryManager(std::move(alloc), memoryAffinity)
    {
        m_InterLayerMemoryGroup = CreateMemoryGroup(m_InterLayerMemoryMgr);
    }

protected:
    virtual std::shared_ptr<arm_compute::IMemoryGroup>
    CreateMemoryGroup(const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager) override;
#endif
};

class ClMemoryManager : public BaseMemoryManager
{
public:
    ClMemoryManager() {}
    virtual ~ClMemoryManager() {}

#ifdef ARMCOMPUTECL_ENABLED
    ClMemoryManager(std::unique_ptr<arm_compute::IAllocator> alloc)
    : BaseMemoryManager(std::move(alloc), MemoryAffinity::Buffer)
    {
        m_InterLayerMemoryGroup = CreateMemoryGroup(m_InterLayerMemoryMgr);
    }

protected:
    virtual std::shared_ptr<arm_compute::IMemoryGroup>
    CreateMemoryGroup(const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager) override;
#endif
};

} //namespace armnn