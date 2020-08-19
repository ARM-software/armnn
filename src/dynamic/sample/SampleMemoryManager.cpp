//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SampleMemoryManager.hpp"

#include <algorithm>

namespace sdb // sample dynamic backend
{

SampleMemoryManager::SampleMemoryManager()
{}

SampleMemoryManager::~SampleMemoryManager()
{}

SampleMemoryManager::Pool* SampleMemoryManager::Manage(unsigned int numBytes)
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

void SampleMemoryManager::Allocate(SampleMemoryManager::Pool* pool)
{
    m_FreePools.push_back(pool);
}

void* SampleMemoryManager::GetPointer(SampleMemoryManager::Pool* pool)
{
    return pool->GetPointer();
}

void SampleMemoryManager::Acquire()
{
    for (Pool &pool: m_Pools)
    {
         pool.Acquire();
    }
}

void SampleMemoryManager::Release()
{
    for (Pool &pool: m_Pools)
    {
         pool.Release();
    }
}

SampleMemoryManager::Pool::Pool(unsigned int numBytes)
    : m_Size(numBytes),
      m_Pointer(nullptr)
{}

SampleMemoryManager::Pool::~Pool()
{
    if (m_Pointer)
    {
        Release();
    }
}

void* SampleMemoryManager::Pool::GetPointer()
{
    return m_Pointer;
}

void SampleMemoryManager::Pool::Reserve(unsigned int numBytes)
{
    m_Size = std::max(m_Size, numBytes);
}

void SampleMemoryManager::Pool::Acquire()
{
    m_Pointer = ::operator new(size_t(m_Size));
}

void SampleMemoryManager::Pool::Release()
{
    ::operator delete(m_Pointer);
    m_Pointer = nullptr;
}

} // namespace sdb
