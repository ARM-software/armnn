//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/TensorHandle.hpp>

#include "RefMemoryManager.hpp"

namespace armnn
{

// An implementation of ITensorHandle with simple "bump the pointer" memory-management behaviour
class RefTensorHandle : public ITensorHandle
{
public:
    RefTensorHandle(const TensorInfo& tensorInfo, std::shared_ptr<RefMemoryManager> &memoryManager);

    RefTensorHandle(const TensorInfo& tensorInfo);

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

private:
    // Only used for testing
    void CopyOutTo(void*) const override;
    void CopyInFrom(const void*) override;

    void* GetPointer() const;

    RefTensorHandle(const RefTensorHandle& other) = delete; // noncopyable
    RefTensorHandle& operator=(const RefTensorHandle& other) = delete; //noncopyable

    TensorInfo m_TensorInfo;

    std::shared_ptr<RefMemoryManager> m_MemoryManager;
    RefMemoryManager::Pool* m_Pool;
    mutable void* m_UnmanagedMemory;
    void* m_ImportedMemory;
};

}
