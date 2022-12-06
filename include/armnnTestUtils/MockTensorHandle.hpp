//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "MockMemoryManager.hpp"

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/MemorySources.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>
#include <armnn/backends/ITensorHandle.hpp>
#include <memory>

namespace armnn
{

// An implementation of ITensorHandle with simple "bump the pointer" memory-management behaviour
class MockTensorHandle : public ITensorHandle
{
public:
    MockTensorHandle(const TensorInfo& tensorInfo, std::shared_ptr<MockMemoryManager>& memoryManager);

    MockTensorHandle(const TensorInfo& tensorInfo, MemorySourceFlags importFlags);

    ~MockTensorHandle() override;

    void Manage() override;

    void Allocate() override;

    ITensorHandle* GetParent() const override
    {
        return nullptr;
    }

    const void* Map(bool /* blocking = true */) const override;
    using ITensorHandle::Map;

    void Unmap() const override
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

    MemorySourceFlags GetImportFlags() const override
    {
        return m_ImportFlags;
    }

    bool Import(void* memory, MemorySource source) override;
    bool CanBeImported(void* memory, MemorySource source) override;

private:
    // Only used for testing
    void CopyOutTo(void*) const override;
    void CopyInFrom(const void*) override;

    void* GetPointer() const;

    MockTensorHandle(const MockTensorHandle& other) = delete;               // noncopyable
    MockTensorHandle& operator=(const MockTensorHandle& other) = delete;    //noncopyable

    TensorInfo m_TensorInfo;

    std::shared_ptr<MockMemoryManager> m_MemoryManager;
    MockMemoryManager::Pool* m_Pool;
    mutable void* m_UnmanagedMemory;
    MemorySourceFlags m_ImportFlags;
    bool m_Imported;
    bool m_IsImportEnabled;
};

}    // namespace armnn
