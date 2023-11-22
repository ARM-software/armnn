//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "GpuFsaTensorHandle.hpp"

namespace armnn
{
GpuFsaTensorHandle::GpuFsaTensorHandle(const TensorInfo& tensorInfo,
                                       std::shared_ptr<GpuFsaMemoryManager>& memoryManager)
    : m_TensorInfo(tensorInfo)
    , m_MemoryManager(memoryManager)
    , m_Pool(nullptr)
    , m_UnmanagedMemory(nullptr)
    , m_ImportFlags(static_cast<MemorySourceFlags>(MemorySource::Undefined))
    , m_Imported(false)
    , m_IsImportEnabled(false)
{}

GpuFsaTensorHandle::GpuFsaTensorHandle(const TensorInfo& tensorInfo,
                                       MemorySourceFlags importFlags)
    : m_TensorInfo(tensorInfo)
    , m_Pool(nullptr)
    , m_UnmanagedMemory(nullptr)
    , m_ImportFlags(importFlags)
    , m_Imported(false)
    , m_IsImportEnabled(true)
{}

GpuFsaTensorHandle::~GpuFsaTensorHandle()
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

void GpuFsaTensorHandle::Manage()
{
    if (!m_IsImportEnabled)
    {
        if (m_Pool == nullptr)
        {
            throw MemoryValidationException("GpuFsaTensorHandle::Manage() called twice");
        }
        if (m_UnmanagedMemory == nullptr)
        {
            throw MemoryValidationException("GpuFsaTensorHandle::Manage() called after Allocate()");
        }

        m_Pool = m_MemoryManager->Manage(m_TensorInfo.GetNumBytes());
    }
}

void GpuFsaTensorHandle::Allocate()
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
            throw InvalidArgumentException("GpuFsaTensorHandle::Allocate Trying to allocate a GpuFsaTensorHandle"
                                           "that already has allocated memory.");
        }
    }
}

const void* GpuFsaTensorHandle::Map(bool /*unused*/) const
{
    return GetPointer();
}

void* GpuFsaTensorHandle::GetPointer() const
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
        throw NullPointerException("GpuFsaTensorHandle::GetPointer called on unmanaged, unallocated tensor handle");
    }
}

void GpuFsaTensorHandle::CopyOutTo(void* dest) const
{
    const void *src = GetPointer();
    if (src == nullptr)
    {
    throw MemoryValidationException("GpuFsaTensorhandle: CopyOutTo: Invalid memory src pointer");
    }
    memcpy(dest, src, m_TensorInfo.GetNumBytes());
}

void GpuFsaTensorHandle::CopyInFrom(const void* src)
{
    void *dest = GetPointer();
    if (dest == nullptr)
    {
    throw MemoryValidationException("GpuFsaTensorhandle: CopyInFrom: Invalid memory dest pointer");
    }
    memcpy(dest, src, m_TensorInfo.GetNumBytes());
}

bool GpuFsaTensorHandle::Import(void* memory, MemorySource source)
{
    if (m_ImportFlags & static_cast<MemorySourceFlags>(source))
    {
        if (m_IsImportEnabled && source == MemorySource::Malloc)
        {
            // Check memory alignment
            if(!CanBeImported(memory, source))
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

bool GpuFsaTensorHandle::CanBeImported(void* memory, MemorySource source)
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



}