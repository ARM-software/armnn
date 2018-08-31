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

/** Blob memory pool */
class BlobMemoryPool : public IMemoryPool
{
public:
    BlobMemoryPool(arm_compute::IAllocator* allocator, std::vector<size_t> blobSizes);

    ~BlobMemoryPool();

    BlobMemoryPool(const BlobMemoryPool&) = delete;

    BlobMemoryPool& operator=(const BlobMemoryPool&) = delete;

    BlobMemoryPool(BlobMemoryPool&&) = default;

    BlobMemoryPool& operator=(BlobMemoryPool&&) = default;

    void acquire(arm_compute::MemoryMappings &handles) override;
    void release(arm_compute::MemoryMappings &handles) override;

    arm_compute::MappingType mapping_type() const override;

    std::unique_ptr<arm_compute::IMemoryPool> duplicate() override;

    void AllocatePool() override;
    void ReleasePool() override;

private:
    /// Allocator to use for internal allocation
    arm_compute::IAllocator* m_Allocator;

    /// Vector holding all the memory blobs
    std::vector<void*> m_Blobs;

    /// Sizes of each memory blob
    std::vector<size_t> m_BlobSizes;

    /// Flag indicating whether memory has been allocated for the pool
    bool m_MemoryAllocated;
};

} // namespace armnn