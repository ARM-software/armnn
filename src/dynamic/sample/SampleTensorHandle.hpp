//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backendsCommon/CpuTensorHandle.hpp>

#include "SampleMemoryManager.hpp"

namespace armnn
{

// An implementation of ITensorHandle with simple "bump the pointer" memory-management behaviour
class SampleTensorHandle : public ITensorHandle
{
public:
    SampleTensorHandle(const TensorInfo& tensorInfo, std::shared_ptr<SampleMemoryManager> &memoryManager);

    SampleTensorHandle(const TensorInfo& tensorInfo,
                       std::shared_ptr<SampleMemoryManager> &memoryManager,
                       MemorySourceFlags importFlags);

    ~SampleTensorHandle();

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

private:
    void* GetPointer() const;

    SampleTensorHandle(const SampleTensorHandle& other) = delete; // noncopyable
    SampleTensorHandle& operator=(const SampleTensorHandle& other) = delete; //noncopyable

    TensorInfo m_TensorInfo;

    std::shared_ptr<SampleMemoryManager> m_MemoryManager;
    SampleMemoryManager::Pool* m_Pool;
    mutable void *m_UnmanagedMemory;
    MemorySourceFlags m_ImportFlags;
    bool m_Imported;
};

}
