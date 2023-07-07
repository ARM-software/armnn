//
// Copyright © 2019-2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/TensorHandle.hpp>

#include "RefMemoryManager.hpp"

namespace armnn
{

class RefTensorHandleDecorator;
// An implementation of ITensorHandle with simple "bump the pointer" memory-management behaviour
class RefTensorHandle : public ITensorHandle
{
public:
    RefTensorHandle(const TensorInfo& tensorInfo, std::shared_ptr<RefMemoryManager>& memoryManager);

    RefTensorHandle(const TensorInfo& tensorInfo);

    RefTensorHandle(const TensorInfo& tensorInfo, const RefTensorHandle& parent);

    ~RefTensorHandle();

    virtual void Manage() override;

    virtual void Allocate() override;

    virtual ITensorHandle* GetParent() const override
    {
        return nullptr;
    }

    virtual const void* Map(bool /* blocking = true */) const override;
    using ITensorHandle::Map;

    virtual void Unmap() const override
    {}

    TensorShape GetStrides() const override
    {
        return GetUnpaddedTensorStrides(m_TensorInfo);
    }

    TensorShape GetShape() const override
    {
        return m_TensorInfo.GetShape();
    }

    const TensorInfo& GetTensorInfo() const
    {
        return m_TensorInfo;
    }

    virtual MemorySourceFlags GetImportFlags() const override;

    virtual bool Import(void* memory, MemorySource source) override;
    virtual bool CanBeImported(void* memory, MemorySource source) override;

    virtual std::shared_ptr<ITensorHandle> DecorateTensorHandle(const TensorInfo& tensorInfo) override;

private:
    // Only used for testing
    void CopyOutTo(void*) const override;
    void CopyInFrom(const void*) override;

    void* GetPointer() const;

    RefTensorHandle(const RefTensorHandle& other) = delete; // noncopyable
    RefTensorHandle& operator=(const RefTensorHandle& other) = delete; //noncopyable

    TensorInfo m_TensorInfo;

    mutable std::shared_ptr<RefMemoryManager> m_MemoryManager;
    RefMemoryManager::Pool* m_Pool;
    mutable void* m_UnmanagedMemory;
    void* m_ImportedMemory;
    std::vector<std::shared_ptr<RefTensorHandleDecorator>> m_Decorated;
};

class RefTensorHandleDecorator : public RefTensorHandle
{
public:
    RefTensorHandleDecorator(const TensorInfo& tensorInfo, const RefTensorHandle& parent);

    ~RefTensorHandleDecorator() = default;

    virtual void Manage() override;

    virtual void Allocate() override;

    virtual ITensorHandle* GetParent() const override
    {
        return nullptr;
    }

    virtual const void* Map(bool /* blocking = true */) const override;
    using ITensorHandle::Map;

    virtual void Unmap() const override
    {}

    TensorShape GetStrides() const override
    {
        return GetUnpaddedTensorStrides(m_TensorInfo);
    }

    TensorShape GetShape() const override
    {
        return m_TensorInfo.GetShape();
    }

    const TensorInfo& GetTensorInfo() const
    {
        return m_TensorInfo;
    }

    virtual MemorySourceFlags GetImportFlags() const override;

    virtual bool Import(void* memory, MemorySource source) override;
    virtual bool CanBeImported(void* memory, MemorySource source) override;

    virtual std::shared_ptr<ITensorHandle> DecorateTensorHandle(const TensorInfo& tensorInfo) override;

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

    /// Testing support to be able to verify and set tensor data content
    void CopyOutTo(void* /* memory */) const override
    {};

    void CopyInFrom(const void* /* memory */) override
    {};

    /// Unimport externally allocated memory
    void Unimport() override
    {};

private:
    TensorInfo m_TensorInfo;
    const RefTensorHandle& m_Parent;
};

}

