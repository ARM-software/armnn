//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <memory>

#include <armnn/MemorySources.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

/**
* Default Memory Allocator class returned from IBackendInternal::GetDefaultAllocator(MemorySource)
*/
class GpuFsaBackendDefaultAllocator : public ICustomAllocator
{
public:
    GpuFsaBackendDefaultAllocator() = default;

    void* allocate(size_t size, size_t alignment = 0) override
    {
        IgnoreUnused(alignment);
        cl_mem buf{ clCreateBuffer(arm_compute::CLScheduler::get().context().get(),
                                   CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                                   size,
                                   nullptr,
                                   nullptr)};
        return static_cast<void *>(buf);
    }

    void free(void* ptr) override
    {
        ARM_COMPUTE_ERROR_ON(ptr == nullptr);
        clReleaseMemObject(static_cast<cl_mem>(ptr));
    }

    MemorySource GetMemorySourceType() override
    {
        return MemorySource::Gralloc;
    }

    void* GetMemoryRegionAtOffset(void* buffer, size_t offset, size_t alignment = 0) override
    {
        IgnoreUnused(alignment);
        return static_cast<char*>(buffer) + offset;
    }
};
} // namespace armnn