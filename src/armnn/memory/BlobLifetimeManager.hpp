//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "arm_compute/runtime/ISimpleLifetimeManager.h"

namespace armnn
{

class BlobLifetimeManager : public arm_compute::ISimpleLifetimeManager
{
public:
    BlobLifetimeManager();

    BlobLifetimeManager(const BlobLifetimeManager&) = delete;

    BlobLifetimeManager& operator=(const BlobLifetimeManager&) = delete;

    BlobLifetimeManager(BlobLifetimeManager&&) = default;

    BlobLifetimeManager& operator=(BlobLifetimeManager&&) = default;

    std::unique_ptr<arm_compute::IMemoryPool> create_pool(arm_compute::IAllocator* allocator) override;

    arm_compute::MappingType mapping_type() const override;

private:
    void update_blobs_and_mappings() override;

    std::vector<size_t> m_BlobSizes;
};

} // namespace armnn