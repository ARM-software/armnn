//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/TensorHandle.hpp>

#include "SampleMemoryManager.hpp"

namespace sdb // sample dynamic backend
{

// An implementation of ITensorHandle with simple "bump the pointer" memory-management behaviour
class SampleTensorHandle : public armnn::ITensorHandle
{
public:
    SampleTensorHandle(const armnn::TensorInfo& tensorInfo, std::shared_ptr<SampleMemoryManager> &memoryManager);

    SampleTensorHandle(const armnn::TensorInfo& tensorInfo, armnn::MemorySourceFlags importFlags);

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

    armnn::TensorShape GetStrides() const override
    {
        return GetUnpaddedTensorStrides(m_TensorInfo);
    }

    armnn::TensorShape GetShape() const override
    {
        return m_TensorInfo.GetShape();
    }

    const armnn::TensorInfo& GetTensorInfo() const
    {
        return m_TensorInfo;
    }

    virtual armnn::MemorySourceFlags GetImportFlags() const override
    {
        return m_ImportFlags;
    }

    virtual bool Import(void* memory, armnn::MemorySource source) override;

private:
    // Only used for testing
    void CopyOutTo(void*) const override;
    void CopyInFrom(const void*) override;

    void* GetPointer() const;

    SampleTensorHandle(const SampleTensorHandle& other) = delete; // noncopyable
    SampleTensorHandle& operator=(const SampleTensorHandle& other) = delete; //noncopyable

    armnn::TensorInfo m_TensorInfo;

    std::shared_ptr<SampleMemoryManager> m_MemoryManager;
    SampleMemoryManager::Pool* m_Pool;
    mutable void *m_UnmanagedMemory;
    armnn::MemorySourceFlags m_ImportFlags;
    bool m_Imported;
};

} // namespace sdb
