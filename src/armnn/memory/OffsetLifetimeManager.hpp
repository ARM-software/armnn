//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "arm_compute/runtime/ISimpleLifetimeManager.h"

namespace armnn
{

class OffsetLifetimeManager : public arm_compute::ISimpleLifetimeManager
{
public:
    OffsetLifetimeManager();

    OffsetLifetimeManager(const OffsetLifetimeManager&) = delete;

    OffsetLifetimeManager& operator=(const OffsetLifetimeManager&) = delete;

    OffsetLifetimeManager(OffsetLifetimeManager&&) = default;

    OffsetLifetimeManager& operator=(OffsetLifetimeManager&&) = default;

    std::unique_ptr<arm_compute::IMemoryPool> create_pool(arm_compute::IAllocator* allocator) override;

    arm_compute::MappingType mapping_type() const override;

private:
    void update_blobs_and_mappings() override;

private:
    /// Memory blob size
    size_t m_BlobSize;
};

} // namespace armnn