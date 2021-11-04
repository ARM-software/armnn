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
/** Custom Allocator interface */
class ICustomAllocator
{
public:
    /** Default virtual destructor. */
    virtual ~ICustomAllocator() = default;

    /** Interface to be implemented by the child class to allocate bytes
     *
     * @param[in] size      Size to allocate
     * @param[in] alignment Alignment that the returned pointer should comply with
     *
     * @return A pointer to the allocated memory
     * The returned pointer must be host write accessible
     */
    virtual void* allocate(size_t size, size_t alignment) = 0;

    /** Interface to be implemented by the child class to free the allocated bytes */
    virtual void free(void* ptr) = 0;

    /**  Used to specify what type of memory is being allocated by this allocator.
     *  Supported types are:
     *       MemorySource::Malloc
     *       MemorySource::DmaBuf
     *       MemorySource::DmaBufProtected
     */
    virtual armnn::MemorySource GetMemorySourceType() = 0;

    /** Interface that may be implemented to allow retrieval of Memory Region
     *  from allocated buffer at a certain offset
     */
    virtual void* GetMemoryRegionAtOffset(void* buffer, size_t offset, size_t alignment = 0)
    {
        IgnoreUnused(offset);
        IgnoreUnused(alignment);
        IgnoreUnused(buffer);
        throw armnn::Exception(
                "ICustomerAllocator::GetMemoryRegionAtOffset(): This function should be overridden in subclass.");
    }

};
} // namespace armnn