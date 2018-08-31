//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "BlobMemoryPool.hpp"

#include <boost/assert.hpp>

namespace armnn
{

BlobMemoryPool::BlobMemoryPool(arm_compute::IAllocator* allocator, std::vector<size_t> blobSizes)
        : m_Allocator(allocator)
        , m_Blobs()
        , m_BlobSizes(std::move(blobSizes))
        , m_MemoryAllocated(false)
{
    AllocatePool();
}

BlobMemoryPool::~BlobMemoryPool()
{
    ReleasePool();
}

void BlobMemoryPool::acquire(arm_compute::MemoryMappings& handles)
{
    // Set memory to handlers
    for (auto& handle : handles)
    {
        BOOST_ASSERT(handle.first);
        *handle.first = m_Blobs[handle.second];
    }
}

void BlobMemoryPool::release(arm_compute::MemoryMappings &handles)
{
    for (auto& handle : handles)
    {
        BOOST_ASSERT(handle.first);
        *handle.first = nullptr;
    }
}

arm_compute::MappingType BlobMemoryPool::mapping_type() const
{
    return arm_compute::MappingType::BLOBS;
}

std::unique_ptr<arm_compute::IMemoryPool> BlobMemoryPool::duplicate()
{
    BOOST_ASSERT(m_Allocator);
    return std::make_unique<BlobMemoryPool>(m_Allocator, m_BlobSizes);
}

void BlobMemoryPool::AllocatePool()
{
    if (!m_MemoryAllocated)
    {
        BOOST_ASSERT(m_Allocator);

        for (const auto& blobSize : m_BlobSizes)
        {
            m_Blobs.push_back(m_Allocator->allocate(blobSize, 0));
        }

        m_MemoryAllocated = true;
    }
}

void BlobMemoryPool::ReleasePool()
{
    if (m_MemoryAllocated)
    {
        BOOST_ASSERT(m_Allocator);

        for (auto& blob : m_Blobs)
        {
            m_Allocator->free(blob);
        }

        m_Blobs.clear();

        m_MemoryAllocated = false;
    }
}

} // namespace armnn