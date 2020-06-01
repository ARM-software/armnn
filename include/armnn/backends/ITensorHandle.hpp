//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/MemorySources.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

class TensorShape;

class ITensorHandle
{
public:
    virtual ~ITensorHandle(){}

    /// Indicate to the memory manager that this resource is active.
    /// This is used to compute overlapping lifetimes of resources.
    virtual void Manage() = 0;

    /// Indicate to the memory manager that this resource is no longer active.
    /// This is used to compute overlapping lifetimes of resources.
    virtual void Allocate() = 0;

    /// Get the parent tensor if this is a subtensor.
    /// \return a pointer to the parent tensor. Otherwise nullptr if not a subtensor.
    virtual ITensorHandle* GetParent() const = 0;

    /// Map the tensor data for access.
    /// \param blocking hint to block the calling thread until all other accesses are complete. (backend dependent)
    /// \return pointer to the first element of the mapped data.
    virtual const void* Map(bool blocking=true) const = 0;

    /// Unmap the tensor data
    virtual void Unmap() const = 0;

    /// Map the tensor data for access. Must be paired with call to Unmap().
    /// \param blocking hint to block the calling thread until all other accesses are complete. (backend dependent)
    /// \return pointer to the first element of the mapped data.
    void* Map(bool blocking=true)
    {
        return const_cast<void*>(static_cast<const ITensorHandle*>(this)->Map(blocking));
    }

    /// Unmap the tensor data that was previously mapped with call to Map().
    void Unmap()
    {
        return static_cast<const ITensorHandle*>(this)->Unmap();
    }

    /// Get the strides for each dimension ordered from largest to smallest where
    /// the smallest value is the same as the size of a single element in the tensor.
    /// \return a TensorShape filled with the strides for each dimension
    virtual TensorShape GetStrides() const = 0;

    /// Get the number of elements for each dimension ordered from slowest iterating dimension
    /// to fastest iterating dimension.
    /// \return a TensorShape filled with the number of elements for each dimension.
    virtual TensorShape GetShape() const = 0;

    /// Testing support to be able to verify and set tensor data content
    virtual void CopyOutTo(void* memory) const = 0;
    virtual void CopyInFrom(const void* memory) = 0;

    /// Get flags describing supported import sources.
    virtual unsigned int GetImportFlags() const { return 0; }

    /// Import externally allocated memory
    /// \param memory base address of the memory being imported.
    /// \param source source of the allocation for the memory being imported.
    /// \return true on success or false on failure
    virtual bool Import(void* memory, MemorySource source)
    {
        IgnoreUnused(memory, source);
        return false;
    };
};

}
