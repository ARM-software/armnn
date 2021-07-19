//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <memory>

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
     */
    virtual void *allocate(size_t size, size_t alignment) = 0;
    /** Interface to be implemented by the child class to free the allocated tensor */
    virtual void free(void *ptr) = 0;

    // Utility Function to define the Custom Memory Allocators capabilities
    virtual bool SupportsProtectedMemory() = 0;

};
} // namespace armnn