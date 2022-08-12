//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/TensorHandle.hpp>

#include "TosaRefMemoryManager.hpp"

namespace armnn
{

// An implementation of ITensorHandle with simple "bump the pointer" memory-management behaviour
class TosaRefTensorHandle : public ITensorHandle
{
public:
    TosaRefTensorHandle(const TensorInfo& tensorInfo, std::shared_ptr<TosaRefMemoryManager> &memoryManager);

    TosaRefTensorHandle(const TensorInfo& tensorInfo, MemorySourceFlags importFlags);

    ~TosaRefTensorHandle();

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

    virtual MemorySourceFlags GetImportFlags() const override
    {
        return m_ImportFlags;
    }

    virtual bool Import(void* memory, MemorySource source) override;
    virtual bool CanBeImported(void* memory, MemorySource source) override;

private:
    // Only used for testing
    void CopyOutTo(void*) const override;
    void CopyInFrom(const void*) override;

    void* GetPointer() const;

    TosaRefTensorHandle(const TosaRefTensorHandle& other) = delete; // noncopyable
    TosaRefTensorHandle& operator=(const TosaRefTensorHandle& other) = delete; //noncopyable

    TensorInfo m_TensorInfo;

    std::shared_ptr<TosaRefMemoryManager> m_MemoryManager;
    TosaRefMemoryManager::Pool* m_Pool;
    mutable void* m_UnmanagedMemory;
    MemorySourceFlags m_ImportFlags;
    bool m_Imported;
    bool m_IsImportEnabled;
};

}
