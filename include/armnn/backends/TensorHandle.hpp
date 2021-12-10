//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ITensorHandle.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnnUtils/CompatibleTypes.hpp>

#include <algorithm>

namespace armnn
{

// Get a TensorShape representing the strides (in bytes) for each dimension
// of a tensor, assuming fully packed data with no padding
TensorShape GetUnpaddedTensorStrides(const TensorInfo& tensorInfo);

// Abstract tensor handles wrapping a readable region of memory, interpreting it as tensor data.
class ConstTensorHandle : public ITensorHandle
{
public:
    template <typename T>
    const T* GetConstTensor() const
    {
        if (armnnUtils::CompatibleTypes<T>(GetTensorInfo().GetDataType()))
        {
            return reinterpret_cast<const T*>(m_Memory);
        }
        else
        {
            throw armnn::Exception("Attempting to get not compatible type tensor!");
        }
    }

    const TensorInfo& GetTensorInfo() const
    {
        return m_TensorInfo;
    }

    virtual void Manage() override {}

    virtual ITensorHandle* GetParent() const override { return nullptr; }

    virtual const void* Map(bool /* blocking = true */) const override { return m_Memory; }
    virtual void Unmap() const override {}

    TensorShape GetStrides() const override
    {
        return GetUnpaddedTensorStrides(m_TensorInfo);
    }
    TensorShape GetShape() const override { return m_TensorInfo.GetShape(); }

protected:
    ConstTensorHandle(const TensorInfo& tensorInfo);

    void SetConstMemory(const void* mem) { m_Memory = mem; }

private:
    // Only used for testing
    void CopyOutTo(void *) const override { ARMNN_ASSERT_MSG(false, "Unimplemented"); }
    void CopyInFrom(const void*) override { ARMNN_ASSERT_MSG(false, "Unimplemented"); }

    ConstTensorHandle(const ConstTensorHandle& other) = delete;
    ConstTensorHandle& operator=(const ConstTensorHandle& other) = delete;

    TensorInfo m_TensorInfo;
    const void* m_Memory;
};

template<>
const void* ConstTensorHandle::GetConstTensor<void>() const;

// Abstract specialization of ConstTensorHandle that allows write access to the same data.
class TensorHandle : public ConstTensorHandle
{
public:
    template <typename T>
    T* GetTensor() const
    {
        if (armnnUtils::CompatibleTypes<T>(GetTensorInfo().GetDataType()))
        {
            return reinterpret_cast<T*>(m_MutableMemory);
        }
        else
        {
            throw armnn::Exception("Attempting to get not compatible type tensor!");
        }
    }

protected:
    TensorHandle(const TensorInfo& tensorInfo);

    void SetMemory(void* mem)
    {
        m_MutableMemory = mem;
        SetConstMemory(m_MutableMemory);
    }

private:

    TensorHandle(const TensorHandle& other) = delete;
    TensorHandle& operator=(const TensorHandle& other) = delete;
    void* m_MutableMemory;
};

template <>
void* TensorHandle::GetTensor<void>() const;

// A TensorHandle that owns the wrapped memory region.
class ScopedTensorHandle : public TensorHandle
{
public:
    explicit ScopedTensorHandle(const TensorInfo& tensorInfo);

    // Copies contents from Tensor.
    explicit ScopedTensorHandle(const ConstTensor& tensor);

    // Copies contents from ConstTensorHandle
    explicit ScopedTensorHandle(const ConstTensorHandle& tensorHandle);

    ScopedTensorHandle(const ScopedTensorHandle& other);
    ScopedTensorHandle& operator=(const ScopedTensorHandle& other);
    ~ScopedTensorHandle();

    virtual void Allocate() override;

private:
    // Only used for testing
    void CopyOutTo(void* memory) const override;
    void CopyInFrom(const void* memory) override;

    void CopyFrom(const ScopedTensorHandle& other);
    void CopyFrom(const void* srcMemory, unsigned int numBytes);
};

// A TensorHandle that wraps an already allocated memory region.
//
// Clients must make sure the passed in memory region stays alive for the lifetime of
// the PassthroughTensorHandle instance.
//
// Note there is no polymorphism to/from ConstPassthroughTensorHandle.
class PassthroughTensorHandle : public TensorHandle
{
public:
    PassthroughTensorHandle(const TensorInfo& tensorInfo, void* mem)
    :   TensorHandle(tensorInfo)
    {
        SetMemory(mem);
    }

    virtual void Allocate() override;
};

// A ConstTensorHandle that wraps an already allocated memory region.
//
// This allows users to pass in const memory to a network.
// Clients must make sure the passed in memory region stays alive for the lifetime of
// the PassthroughTensorHandle instance.
//
// Note there is no polymorphism to/from PassthroughTensorHandle.
class ConstPassthroughTensorHandle : public ConstTensorHandle
{
public:
    ConstPassthroughTensorHandle(const TensorInfo& tensorInfo, const void* mem)
    :   ConstTensorHandle(tensorInfo)
    {
        SetConstMemory(mem);
    }

    virtual void Allocate() override;
};


// Template specializations.

template <>
const void* ConstTensorHandle::GetConstTensor() const;

template <>
void* TensorHandle::GetTensor() const;

class ManagedConstTensorHandle
{

public:
    explicit ManagedConstTensorHandle(std::shared_ptr<ConstTensorHandle> ptr)
        : m_Mapped(false)
        , m_TensorHandle(std::move(ptr)) {};

    /// RAII Managed resource Unmaps MemoryArea once out of scope
    const void* Map(bool blocking = true)
    {
        if (m_TensorHandle)
        {
            auto pRet = m_TensorHandle->Map(blocking);
            m_Mapped = true;
            return pRet;
        }
        else
        {
            throw armnn::Exception("Attempting to Map null TensorHandle");
        }

    }

    // Delete copy constructor as it's unnecessary
    ManagedConstTensorHandle(const ConstTensorHandle& other) = delete;

    // Delete copy assignment as it's unnecessary
    ManagedConstTensorHandle& operator=(const ManagedConstTensorHandle& other) = delete;

    // Delete move assignment as it's unnecessary
    ManagedConstTensorHandle& operator=(ManagedConstTensorHandle&& other) noexcept = delete;

    ~ManagedConstTensorHandle()
    {
        // Bias tensor handles need to be initialized empty before entering scope of if statement checking if enabled
        if (m_TensorHandle)
        {
            Unmap();
        }
    }

    void Unmap()
    {
        // Only unmap if mapped and TensorHandle exists.
        if (m_Mapped && m_TensorHandle)
        {
            m_TensorHandle->Unmap();
            m_Mapped = false;
        }
    }

    const TensorInfo& GetTensorInfo() const
    {
        return m_TensorHandle->GetTensorInfo();
    }

    bool IsMapped() const
    {
        return m_Mapped;
    }

private:
    bool m_Mapped;
    std::shared_ptr<ConstTensorHandle> m_TensorHandle;
};

using ConstCpuTensorHandle ARMNN_DEPRECATED_MSG_REMOVAL_DATE("ConstCpuTensorHandle is deprecated, "
                                                "use ConstTensorHandle instead", "22.05") = ConstTensorHandle;
using CpuTensorHandle ARMNN_DEPRECATED_MSG_REMOVAL_DATE("CpuTensorHandle is deprecated, "
                                           "use TensorHandle instead", "22.05") = TensorHandle;
using ScopedCpuTensorHandle ARMNN_DEPRECATED_MSG_REMOVAL_DATE("ScopedCpuTensorHandle is deprecated, "
                                                 "use ScopedTensorHandle instead", "22.05") = ScopedTensorHandle;
using PassthroughCpuTensorHandle ARMNN_DEPRECATED_MSG_REMOVAL_DATE("PassthroughCpuTensorHandle is deprecated, use "
                                                      "PassthroughTensorHandle instead",
                                                      "22.05") = PassthroughTensorHandle;
using ConstPassthroughCpuTensorHandle ARMNN_DEPRECATED_MSG_REMOVAL_DATE("ConstPassthroughCpuTensorHandle is "
                                                           "deprecated, use ConstPassthroughTensorHandle "
                                                           "instead", "22.05") = ConstPassthroughTensorHandle;

} // namespace armnn
