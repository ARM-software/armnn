//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "IMemoryPool.hpp"

#include "arm_compute/runtime/IAllocator.h"
#include "arm_compute/runtime/Types.h"

namespace armnn
{

class OffsetMemoryPool : public IMemoryPool
{
public:
    OffsetMemoryPool(arm_compute::IAllocator* allocator, size_t blobSize);

    ~OffsetMemoryPool();

    OffsetMemoryPool(const OffsetMemoryPool&) = delete;

    OffsetMemoryPool& operator=(const OffsetMemoryPool&) = delete;

    OffsetMemoryPool(OffsetMemoryPool&&) = default;

    OffsetMemoryPool& operator=(OffsetMemoryPool &&) = default;

    void acquire(arm_compute::MemoryMappings& handles) override;
    void release(arm_compute::MemoryMappings& handles) override;

    arm_compute::MappingType mapping_type() const override;

    std::unique_ptr<arm_compute::IMemoryPool> duplicate() override;

    void AllocatePool() override;
    void ReleasePool() override;

private:
    /// Allocator to use for internal allocation
    arm_compute::IAllocator* m_Allocator;

    /// Memory blob
    void* m_Blob;

    /// Size of the allocated memory blob
    size_t m_BlobSize;

    /// Flag indicating whether memory has been allocated for the pool
    bool m_MemoryAllocated;
};

} // namespace armnn