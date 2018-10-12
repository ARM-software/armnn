//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "OffsetMemoryPool.hpp"

#include <boost/assert.hpp>

#include <algorithm>

namespace armnn
{

OffsetMemoryPool::OffsetMemoryPool(arm_compute::IAllocator* allocator, size_t blobSize)
    : m_Allocator(allocator)
    , m_Blob()
    , m_BlobSize(blobSize)
    , m_MemoryAllocated(false)
{
    AllocatePool();
}

OffsetMemoryPool::~OffsetMemoryPool()
{
    ReleasePool();
}

void OffsetMemoryPool::acquire(arm_compute::MemoryMappings& handles)
{
    BOOST_ASSERT(m_Blob);

    // Set memory to handlers
    for(auto& handle : handles)
    {
        BOOST_ASSERT(handle.first);
        *handle.first = reinterpret_cast<uint8_t*>(m_Blob) + handle.second;
    }
}

void OffsetMemoryPool::release(arm_compute::MemoryMappings &handles)
{
    for(auto& handle : handles)
    {
        BOOST_ASSERT(handle.first);
        *handle.first = nullptr;
    }
}

arm_compute::MappingType OffsetMemoryPool::mapping_type() const
{
    return arm_compute::MappingType::OFFSETS;
}

std::unique_ptr<arm_compute::IMemoryPool> OffsetMemoryPool::duplicate()
{
    BOOST_ASSERT(m_Allocator);
    return std::make_unique<OffsetMemoryPool>(m_Allocator, m_BlobSize);
}

void OffsetMemoryPool::AllocatePool()
{
    if (!m_MemoryAllocated)
    {
        BOOST_ASSERT(m_Allocator);
        m_Blob = m_Allocator->allocate(m_BlobSize, 0);

        m_MemoryAllocated = true;
    }
}

void OffsetMemoryPool::ReleasePool()
{
    if (m_MemoryAllocated)
    {
        BOOST_ASSERT(m_Allocator);

        m_Allocator->free(m_Blob);
        m_Blob = nullptr;

        m_MemoryAllocated = false;
    }
}

} // namespace armnn