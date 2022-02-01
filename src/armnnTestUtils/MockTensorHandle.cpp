//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnTestUtils/MockTensorHandle.hpp"

namespace armnn
{

MockTensorHandle::MockTensorHandle(const TensorInfo& tensorInfo, std::shared_ptr<MockMemoryManager>& memoryManager)
    : m_TensorInfo(tensorInfo)
    , m_MemoryManager(memoryManager)
    , m_Pool(nullptr)
    , m_UnmanagedMemory(nullptr)
    , m_ImportFlags(static_cast<MemorySourceFlags>(MemorySource::Undefined))
    , m_Imported(false)
    , m_IsImportEnabled(false)
{}

MockTensorHandle::MockTensorHandle(const TensorInfo& tensorInfo, MemorySourceFlags importFlags)
    : m_TensorInfo(tensorInfo)
    , m_Pool(nullptr)
    , m_UnmanagedMemory(nullptr)
    , m_ImportFlags(importFlags)
    , m_Imported(false)
    , m_IsImportEnabled(true)
{}

MockTensorHandle::~MockTensorHandle()
{
    if (!m_Pool)
    {
        // unmanaged
        if (!m_Imported)
        {
            ::operator delete(m_UnmanagedMemory);
        }
    }
}

void MockTensorHandle::Manage()
{
    if (!m_IsImportEnabled)
    {
        ARMNN_ASSERT_MSG(!m_Pool, "MockTensorHandle::Manage() called twice");
        ARMNN_ASSERT_MSG(!m_UnmanagedMemory, "MockTensorHandle::Manage() called after Allocate()");

        m_Pool = m_MemoryManager->Manage(m_TensorInfo.GetNumBytes());
    }
}

void MockTensorHandle::Allocate()
{
    // If import is enabled, do not allocate the tensor
    if (!m_IsImportEnabled)
    {

        if (!m_UnmanagedMemory)
        {
            if (!m_Pool)
            {
                // unmanaged
                m_UnmanagedMemory = ::operator new(m_TensorInfo.GetNumBytes());
            }
            else
            {
                m_MemoryManager->Allocate(m_Pool);
            }
        }
        else
        {
            throw InvalidArgumentException("MockTensorHandle::Allocate Trying to allocate a MockTensorHandle"
                                           "that already has allocated memory.");
        }
    }
}

const void* MockTensorHandle::Map(bool /*unused*/) const
{
    return GetPointer();
}

void* MockTensorHandle::GetPointer() const
{
    if (m_UnmanagedMemory)
    {
        return m_UnmanagedMemory;
    }
    else if (m_Pool)
    {
        return m_MemoryManager->GetPointer(m_Pool);
    }
    else
    {
        throw NullPointerException("MockTensorHandle::GetPointer called on unmanaged, unallocated tensor handle");
    }
}

void MockTensorHandle::CopyOutTo(void* dest) const
{
    const void* src = GetPointer();
    ARMNN_ASSERT(src);
    memcpy(dest, src, m_TensorInfo.GetNumBytes());
}

void MockTensorHandle::CopyInFrom(const void* src)
{
    void* dest = GetPointer();
    ARMNN_ASSERT(dest);
    memcpy(dest, src, m_TensorInfo.GetNumBytes());
}

bool MockTensorHandle::Import(void* memory, MemorySource source)
{
    if (m_ImportFlags & static_cast<MemorySourceFlags>(source))
    {
        if (m_IsImportEnabled && source == MemorySource::Malloc)
        {
            // Check memory alignment
            if (!CanBeImported(memory, source))
            {
                if (m_Imported)
                {
                    m_Imported        = false;
                    m_UnmanagedMemory = nullptr;
                }

                return false;
            }

            // m_UnmanagedMemory not yet allocated.
            if (!m_Imported && !m_UnmanagedMemory)
            {
                m_UnmanagedMemory = memory;
                m_Imported        = true;
                return true;
            }

            // m_UnmanagedMemory initially allocated with Allocate().
            if (!m_Imported && m_UnmanagedMemory)
            {
                return false;
            }

            // m_UnmanagedMemory previously imported.
            if (m_Imported)
            {
                m_UnmanagedMemory = memory;
                return true;
            }
        }
    }

    return false;
}

bool MockTensorHandle::CanBeImported(void* memory, MemorySource source)
{
    if (m_ImportFlags & static_cast<MemorySourceFlags>(source))
    {
        if (m_IsImportEnabled && source == MemorySource::Malloc)
        {
            uintptr_t alignment = GetDataTypeSize(m_TensorInfo.GetDataType());
            if (reinterpret_cast<uintptr_t>(memory) % alignment)
            {
                return false;
            }

            return true;
        }
    }
    return false;
}

}    // namespace armnn
