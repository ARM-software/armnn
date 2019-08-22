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
    m_ImportFlags(static_cast<MemorySourceFlags>(MemorySource::Undefined)),
    m_Imported(false)
{

}

RefTensorHandle::RefTensorHandle(const TensorInfo& tensorInfo, std::shared_ptr<RefMemoryManager> &memoryManager,
                                 MemorySourceFlags importFlags)
                                 : m_TensorInfo(tensorInfo),
                                   m_MemoryManager(memoryManager),
                                   m_Pool(nullptr),
                                   m_UnmanagedMemory(nullptr),
                                   m_ImportFlags(importFlags),
                                   m_Imported(false)
{

}

RefTensorHandle::~RefTensorHandle()
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

void RefTensorHandle::Manage()
{
    BOOST_ASSERT_MSG(!m_Pool, "RefTensorHandle::Manage() called twice");
    BOOST_ASSERT_MSG(!m_UnmanagedMemory, "RefTensorHandle::Manage() called after Allocate()");

    m_Pool = m_MemoryManager->Manage(m_TensorInfo.GetNumBytes());
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
    if (m_UnmanagedMemory)
    {
        return m_UnmanagedMemory;
    }
    else
    {
        BOOST_ASSERT_MSG(m_Pool, "RefTensorHandle::GetPointer called on unmanaged, unallocated tensor handle");
        return m_MemoryManager->GetPointer(m_Pool);
    }
}

void RefTensorHandle::CopyOutTo(void* dest) const
{
    const void *src = GetPointer();
    BOOST_ASSERT(src);
    memcpy(dest, src, m_TensorInfo.GetNumBytes());
}

void RefTensorHandle::CopyInFrom(const void* src)
{
    void *dest = GetPointer();
    BOOST_ASSERT(dest);
    memcpy(dest, src, m_TensorInfo.GetNumBytes());
}

bool RefTensorHandle::Import(void* memory, MemorySource source)
{

    if (m_ImportFlags & static_cast<MemorySourceFlags>(source))
    {
        if (source == MemorySource::Malloc)
        {
            // Checks the 16 byte memory alignment.
            if (reinterpret_cast<uint64_t>(memory) % 16)
            {
                if (m_Imported)
                {
                    m_Imported = false;
                    m_UnmanagedMemory = nullptr;
                }

                return false;
            }

            // m_UnmanagedMemory not yet allocated.
            if (!m_Imported && !m_UnmanagedMemory)
            {
                m_UnmanagedMemory = memory;
                m_Imported = true;
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

}
