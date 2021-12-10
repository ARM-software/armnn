//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#if defined(ARMCOMPUTENEON_ENABLED) || defined(ARMCOMPUTECL_ENABLED)
#include <arm_compute/runtime/MemoryGroup.h>
#endif

#if defined(ARMCOMPUTENEON_ENABLED) || defined(ARMCOMPUTECL_ENABLED)
#include <arm_compute/runtime/IAllocator.h>
#include <arm_compute/runtime/IMemoryGroup.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>
#endif

#if defined(ARMCOMPUTECL_ENABLED)
#include <arm_compute/runtime/CL/CLTensorAllocator.h>
#endif

namespace armnn
{

class BaseMemoryManager : public IMemoryManager
{
public:
    enum class MemoryAffinity
    {
        Buffer,
        Offset
    };

    BaseMemoryManager() { }
    virtual ~BaseMemoryManager() { }

    void Acquire() override;
    void Release() override;

#if defined(ARMCOMPUTENEON_ENABLED) || defined(ARMCOMPUTECL_ENABLED)
    BaseMemoryManager(std::shared_ptr<arm_compute::IAllocator> alloc, MemoryAffinity memoryAffinity);

    std::shared_ptr<arm_compute::MemoryManagerOnDemand>& GetIntraLayerManager() { return m_IntraLayerMemoryMgr; }
    std::shared_ptr<arm_compute::MemoryManagerOnDemand>& GetInterLayerManager() { return m_InterLayerMemoryMgr; }
    std::shared_ptr<arm_compute::IMemoryGroup>& GetInterLayerMemoryGroup()      { return m_InterLayerMemoryGroup; }

protected:
    std::shared_ptr<arm_compute::IAllocator>            m_Allocator;
    std::shared_ptr<arm_compute::MemoryManagerOnDemand> m_IntraLayerMemoryMgr;
    std::shared_ptr<arm_compute::MemoryManagerOnDemand> m_InterLayerMemoryMgr;
    std::shared_ptr<arm_compute::IMemoryGroup>          m_InterLayerMemoryGroup;

    std::shared_ptr<arm_compute::MemoryManagerOnDemand> CreateArmComputeMemoryManager(MemoryAffinity memoryAffinity);

    virtual std::shared_ptr<arm_compute::IMemoryGroup>
    CreateMemoryGroup(const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager) = 0;
#endif
};

#if defined(ARMCOMPUTENEON_ENABLED)
class NeonMemoryManager : public BaseMemoryManager
{
public:
    NeonMemoryManager() {}
    virtual ~NeonMemoryManager() {}

    NeonMemoryManager(std::unique_ptr<arm_compute::IAllocator> alloc, MemoryAffinity memoryAffinity)
    : BaseMemoryManager(std::move(alloc), memoryAffinity)
    {
        m_InterLayerMemoryGroup = CreateMemoryGroup(m_InterLayerMemoryMgr);
    }

protected:
    std::shared_ptr<arm_compute::IMemoryGroup>
    CreateMemoryGroup(const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager) override;
};
#endif

#if defined(ARMCOMPUTECL_ENABLED)
class ClMemoryManager : public BaseMemoryManager
{
public:
    ClMemoryManager() {}
    virtual ~ClMemoryManager() {}

    ClMemoryManager(std::shared_ptr<arm_compute::IAllocator> alloc)
    : BaseMemoryManager(std::move(alloc), MemoryAffinity::Buffer)
    {
        arm_compute::CLTensorAllocator::set_global_allocator(alloc.get());
        m_InterLayerMemoryGroup = CreateMemoryGroup(m_InterLayerMemoryMgr);
    }

protected:
    std::shared_ptr<arm_compute::IMemoryGroup>
    CreateMemoryGroup(const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager) override;
};
#endif

} //namespace armnn
