//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "BlobLifetimeManager.hpp"
#include "BlobMemoryPool.hpp"

#include <arm_compute/runtime/IMemoryGroup.h>

#include "boost/assert.hpp"

#include <algorithm>

namespace armnn
{

BlobLifetimeManager::BlobLifetimeManager()
    : m_BlobSizes()
{
}

arm_compute::MappingType BlobLifetimeManager::mapping_type() const
{
    return arm_compute::MappingType::BLOBS;
}

void BlobLifetimeManager::update_blobs_and_mappings()
{
    using namespace arm_compute;

    BOOST_ASSERT(are_all_finalized());
    BOOST_ASSERT(_active_group);

    // Sort free blobs requirements in descending order.
    _free_blobs.sort([](const Blob & ba, const Blob & bb)
                     {
                         return ba.max_size > bb.max_size;
                     });
    std::vector<size_t> groupSizes;
    std::transform(std::begin(_free_blobs), std::end(_free_blobs), std::back_inserter(groupSizes), [](const Blob & b)
    {
        return b.max_size;
    });

    // Update blob sizes
    size_t max_size = std::max(m_BlobSizes.size(), groupSizes.size());
    m_BlobSizes.resize(max_size, 0);
    groupSizes.resize(max_size, 0);
    std::transform(std::begin(m_BlobSizes), std::end(m_BlobSizes), std::begin(groupSizes),
        std::begin(m_BlobSizes), [](size_t lhs, size_t rhs)
    {
        return std::max(lhs, rhs);
    });

    // Calculate group mappings
    auto& groupMappings  = _active_group->mappings();
    unsigned int blobIdx = 0;

    for(auto& freeBlob : _free_blobs)
    {
        for(auto& boundElementId : freeBlob.bound_elements)
        {
            BOOST_ASSERT(_active_elements.find(boundElementId) != std::end(_active_elements));

            Element& boundElement = _active_elements[boundElementId];
            groupMappings[boundElement.handle] = blobIdx;
        }

        ++blobIdx;
    }
}

std::unique_ptr<arm_compute::IMemoryPool> BlobLifetimeManager::create_pool(arm_compute::IAllocator* allocator)
{
    BOOST_ASSERT(allocator);
    return std::make_unique<BlobMemoryPool>(allocator, m_BlobSizes);
}

} // namespace armnn