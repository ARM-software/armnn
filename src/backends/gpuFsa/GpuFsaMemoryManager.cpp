//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "GpuFsaMemoryManager.hpp"
#include "Exceptions.hpp"

#include <algorithm>

namespace armnn
{

GpuFsaMemoryManager::GpuFsaMemoryManager()
{}

GpuFsaMemoryManager::~GpuFsaMemoryManager()
{}

GpuFsaMemoryManager::Pool* GpuFsaMemoryManager::Manage(unsigned int numBytes)
{
    if (!m_FreePools.empty())
    {
        Pool* res = m_FreePools.back();
        m_FreePools.pop_back();
        res->Reserve(numBytes);
        return res;
    }
    else
    {
        m_Pools.push_front(Pool(numBytes));
        return &m_Pools.front();
    }
}

void GpuFsaMemoryManager::Allocate(GpuFsaMemoryManager::Pool* pool)
{
    if (pool == nullptr)
    {
        throw armnn::MemoryValidationException(
            "GpuFsaMemoryManager: Allocate: Attempting to allocate a null memory pool ptr");
    }
    m_FreePools.push_back(pool);
}

void* GpuFsaMemoryManager::GetPointer(GpuFsaMemoryManager::Pool* pool)
{
    return pool->GetPointer();
}

void GpuFsaMemoryManager::Acquire()
{
    for (Pool &pool: m_Pools)
    {
        pool.Acquire();
    }
}

void GpuFsaMemoryManager::Release()
{
    for (Pool &pool: m_Pools)
    {
        pool.Release();
    }
}

GpuFsaMemoryManager::Pool::Pool(unsigned int numBytes)
        : m_Size(numBytes),
          m_Pointer(nullptr)
{}

GpuFsaMemoryManager::Pool::~Pool()
{
    if (m_Pointer)
    {
        Release();
    }
}

void* GpuFsaMemoryManager::Pool::GetPointer()
{
    if (m_Pointer == nullptr)
    {
        throw armnn::MemoryValidationException(
            "GpuFsaMemoryManager::Pool::GetPointer() called when memory not acquired");
    }
    return m_Pointer;
}

void GpuFsaMemoryManager::Pool::Reserve(unsigned int numBytes)
{
    if (m_Pointer != nullptr)
    {
        throw armnn::MemoryValidationException(
            "GpuFsaMemoryManager::Pool::Reserve() cannot be called after memory acquired");
    }
    m_Size = std::max(m_Size, numBytes);
}

void GpuFsaMemoryManager::Pool::Acquire()
{
    if (m_Pointer != nullptr)
    {
        throw armnn::MemoryValidationException(
            "GpuFsaMemoryManager::Pool::Acquire() called when memory already acquired");
    }
    m_Pointer = ::operator new(size_t(m_Size));
}

void GpuFsaMemoryManager::Pool::Release()
{
    if (m_Pointer == nullptr)
    {
        throw armnn::MemoryValidationException(
            "GpuFsaMemoryManager::Pool::Release() called when memory not acquired");
    }
    ::operator delete(m_Pointer);
    m_Pointer = nullptr;
}

}