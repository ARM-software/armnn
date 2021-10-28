//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <memory>
#include <armnn/MemorySources.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

/** Default Memory Allocator class returned from IBackendInternal::GetDefaultAllocator(MemorySource) */
class DefaultAllocator : public armnn::ICustomAllocator
{
public:
    DefaultAllocator() = default;

    void* allocate(size_t size, size_t alignment = 0) override
    {
        IgnoreUnused(alignment);
        return ::operator new(size_t(size));
    }

    void free(void* ptr) override
    {
        ::operator delete(ptr);
    }

    armnn::MemorySource GetMemorySourceType() override
    {
        return armnn::MemorySource::Malloc;
    }

    void* GetMemoryRegionAtOffset(void* buffer, size_t offset, size_t alignment = 0) override
    {
        IgnoreUnused(alignment);
        return static_cast<char*>(buffer) + offset;
    }
};

} // namespace armnn