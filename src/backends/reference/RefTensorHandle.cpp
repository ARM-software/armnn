//
// Copyright © 2019-2024 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefTensorHandle.hpp"

namespace armnn
{

RefTensorHandle::RefTensorHandle(const TensorInfo& tensorInfo, std::shared_ptr<RefMemoryManager>& memoryManager):
    m_TensorInfo(tensorInfo),
    m_MemoryManager(memoryManager),
    m_Pool(nullptr),
    m_UnmanagedMemory(nullptr),
    m_ImportedMemory(nullptr),
    m_Decorated()
{
}

RefTensorHandle::RefTensorHandle(const TensorInfo& tensorInfo)
                                 : m_TensorInfo(tensorInfo),
                                   m_Pool(nullptr),
                                   m_UnmanagedMemory(nullptr),
                                   m_ImportedMemory(nullptr),
                                   m_Decorated()
{
}

RefTensorHandle::RefTensorHandle(const TensorInfo& tensorInfo, const RefTensorHandle& parent)
        : m_TensorInfo(tensorInfo),
          m_MemoryManager(parent.m_MemoryManager),
          m_Pool(parent.m_Pool),
          m_UnmanagedMemory(parent.m_UnmanagedMemory),
          m_ImportedMemory(parent.m_ImportedMemory),
          m_Decorated()
{
}

RefTensorHandle::~RefTensorHandle()
{
    ::operator delete(m_UnmanagedMemory);
}

void RefTensorHandle::Manage()
{
    ARMNN_THROW_MSG_IF_FALSE(!m_Pool, RuntimeException, "RefTensorHandle::Manage() called twice");
    ARMNN_THROW_MSG_IF_FALSE(!m_UnmanagedMemory, RuntimeException, "RefTensorHandle::Manage() called after Allocate()");

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
    const void* src = GetPointer();
    if (src == nullptr)
    {
        throw NullPointerException("TensorHandle::CopyOutTo called with a null src pointer");
    }
    if (dest == nullptr)
    {
        throw NullPointerException("TensorHandle::CopyOutTo called with a null dest pointer");
    }
    memcpy(dest, src, GetTensorInfo().GetNumBytes());
}

void RefTensorHandle::CopyInFrom(const void* src)
{
    void* dest = GetPointer();
    if (dest == nullptr)
    {
        throw NullPointerException("RefTensorHandle::CopyInFrom called with a null dest pointer");
    }
    if (src == nullptr)
    {
        throw NullPointerException("RefTensorHandle::CopyInFrom called with a null src pointer");
    }
    memcpy(dest, src, GetTensorInfo().GetNumBytes());
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

std::shared_ptr<ITensorHandle> RefTensorHandle::DecorateTensorHandle(const TensorInfo& tensorInfo)
{
    auto decorated = std::make_shared<RefTensorHandleDecorator>(tensorInfo, *this);
    m_Decorated.emplace_back(decorated);
    return decorated;
}

RefTensorHandleDecorator::RefTensorHandleDecorator(const TensorInfo& tensorInfo, const RefTensorHandle& parent)
: RefTensorHandle(tensorInfo)
, m_TensorInfo(tensorInfo)
, m_Parent(parent)
{
}

void RefTensorHandleDecorator::Manage()
{
}

void RefTensorHandleDecorator::Allocate()
{
}

const void* RefTensorHandleDecorator::Map(bool unused) const
{
    return m_Parent.Map(unused);
}

MemorySourceFlags RefTensorHandleDecorator::GetImportFlags() const
{
    return static_cast<MemorySourceFlags>(MemorySource::Malloc);
}

bool RefTensorHandleDecorator::Import(void*, MemorySource )
{
    return false;
}

bool RefTensorHandleDecorator::CanBeImported(void* , MemorySource)
{
    return false;
}

std::shared_ptr<ITensorHandle> RefTensorHandleDecorator::DecorateTensorHandle(const TensorInfo&)
{
    return nullptr;
}


}
