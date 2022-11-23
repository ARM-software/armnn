//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "RefTensorHandle.hpp"

namespace armnn
{

RefTensorHandle::RefTensorHandle(const TensorInfo &tensorInfo, std::shared_ptr<RefMemoryManager> &memoryManager):
    m_TensorInfo(tensorInfo),
    m_MemoryManager(memoryManager),
    m_Pool(nullptr),
    m_UnmanagedMemory(nullptr),
    m_ImportedMemory(nullptr)
{

}

RefTensorHandle::RefTensorHandle(const TensorInfo& tensorInfo)
                                 : m_TensorInfo(tensorInfo),
                                   m_Pool(nullptr),
                                   m_UnmanagedMemory(nullptr),
                                   m_ImportedMemory(nullptr)
{

}

RefTensorHandle::~RefTensorHandle()
{
    ::operator delete(m_UnmanagedMemory);
}

void RefTensorHandle::Manage()
{
    ARMNN_ASSERT_MSG(!m_Pool, "RefTensorHandle::Manage() called twice");
    ARMNN_ASSERT_MSG(!m_UnmanagedMemory, "RefTensorHandle::Manage() called after Allocate()");

    if (m_MemoryManager)
    {
        m_Pool = m_MemoryManager->Manage(m_TensorInfo.GetNumBytes());
    }
}

void RefTensorHandle::Allocate()
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
        throw InvalidArgumentException("RefTensorHandle::Allocate Trying to allocate a RefTensorHandle"
                                       "that already has allocated memory.");
    }
}

const void* RefTensorHandle::Map(bool /*unused*/) const
{
    return GetPointer();
}

void* RefTensorHandle::GetPointer() const
{
    if (m_ImportedMemory)
    {
        return m_ImportedMemory;
    }
    else if (m_UnmanagedMemory)
    {
        return m_UnmanagedMemory;
    }
    else if (m_Pool)
    {
        return m_MemoryManager->GetPointer(m_Pool);
    }
    else
    {
        throw NullPointerException("RefTensorHandle::GetPointer called on unmanaged, unallocated tensor handle");
    }
}

void RefTensorHandle::CopyOutTo(void* dest) const
{
    const void *src = GetPointer();
    ARMNN_ASSERT(src);
    memcpy(dest, src, m_TensorInfo.GetNumBytes());
}

void RefTensorHandle::CopyInFrom(const void* src)
{
    void *dest = GetPointer();
    ARMNN_ASSERT(dest);
    memcpy(dest, src, m_TensorInfo.GetNumBytes());
}

MemorySourceFlags RefTensorHandle::GetImportFlags() const
{
    return static_cast<MemorySourceFlags>(MemorySource::Malloc);
}

bool RefTensorHandle::Import(void* memory, MemorySource source)
{
    if (source == MemorySource::Malloc)
    {
        // Check memory alignment
        if(!CanBeImported(memory, source))
        {
            m_ImportedMemory = nullptr;
            return false;
        }

        m_ImportedMemory = memory;
        return true;
    }

    return false;
}

bool RefTensorHandle::CanBeImported(void *memory, MemorySource source)
{
    if (source == MemorySource::Malloc)
    {
        uintptr_t alignment = GetDataTypeSize(m_TensorInfo.GetDataType());
        if (reinterpret_cast<uintptr_t>(memory) % alignment)
        {
            return false;
        }
        return true;
    }
    return false;
}

}
