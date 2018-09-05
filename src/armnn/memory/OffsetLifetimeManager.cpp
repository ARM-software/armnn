//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "OffsetLifetimeManager.hpp"
#include "OffsetMemoryPool.hpp"

#include "arm_compute/runtime/IMemoryGroup.h"

#include <numeric>

#include "boost/assert.hpp"

namespace armnn
{

OffsetLifetimeManager::OffsetLifetimeManager()
    : m_BlobSize(0)
{
}

std::unique_ptr<arm_compute::IMemoryPool> OffsetLifetimeManager::create_pool(arm_compute::IAllocator* allocator)
{
    BOOST_ASSERT(allocator);
    return std::make_unique<OffsetMemoryPool>(allocator, m_BlobSize);
}

arm_compute::MappingType OffsetLifetimeManager::mapping_type() const
{
    return arm_compute::MappingType::OFFSETS;
}

void OffsetLifetimeManager::update_blobs_and_mappings()
{
    BOOST_ASSERT(are_all_finalized());
    BOOST_ASSERT(_active_group);

    // Update blob size
    size_t maxGroupSize = std::accumulate(std::begin(_free_blobs), std::end(_free_blobs),
        static_cast<size_t>(0), [](size_t s, const Blob& b)
    {
        return s + b.max_size;
    });
    m_BlobSize = std::max(m_BlobSize, maxGroupSize);

    // Calculate group mappings
    auto& groupMappings = _active_group->mappings();
    size_t offset = 0;
    for(auto& freeBlob : _free_blobs)
    {
        for(auto& boundElementId : freeBlob.bound_elements)
        {
            BOOST_ASSERT(_active_elements.find(boundElementId) != std::end(_active_elements));
            Element& boundElement = _active_elements[boundElementId];
            groupMappings[boundElement.handle] = offset;
        }
        offset += freeBlob.max_size;
        BOOST_ASSERT(offset <= m_BlobSize);
    }
}

} // namespace armnn