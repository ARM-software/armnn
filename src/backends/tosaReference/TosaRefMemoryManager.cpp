//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "TosaRefMemoryManager.hpp"

#include <armnn/utility/Assert.hpp>

#include <algorithm>

namespace armnn
{

TosaRefMemoryManager::TosaRefMemoryManager()
{}

TosaRefMemoryManager::~TosaRefMemoryManager()
{}

TosaRefMemoryManager::Pool* TosaRefMemoryManager::Manage(unsigned int numBytes)
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

void TosaRefMemoryManager::Allocate(TosaRefMemoryManager::Pool* pool)
{
    ARMNN_ASSERT(pool);
    m_FreePools.push_back(pool);
}

void* TosaRefMemoryManager::GetPointer(TosaRefMemoryManager::Pool* pool)
{
    return pool->GetPointer();
}

void TosaRefMemoryManager::Acquire()
{
    for (Pool &pool: m_Pools)
    {
         pool.Acquire();
    }
}

void TosaRefMemoryManager::Release()
{
    for (Pool &pool: m_Pools)
    {
         pool.Release();
    }
}

TosaRefMemoryManager::Pool::Pool(unsigned int numBytes)
    : m_Size(numBytes),
      m_Pointer(nullptr)
{}

TosaRefMemoryManager::Pool::~Pool()
{
    if (m_Pointer)
    {
        Release();
    }
}

void* TosaRefMemoryManager::Pool::GetPointer()
{
    ARMNN_ASSERT_MSG(m_Pointer, "TosaRefMemoryManager::Pool::GetPointer() called when memory not acquired");
    return m_Pointer;
}

void TosaRefMemoryManager::Pool::Reserve(unsigned int numBytes)
{
    ARMNN_ASSERT_MSG(!m_Pointer, "TosaRefMemoryManager::Pool::Reserve() cannot be called after memory acquired");
    m_Size = std::max(m_Size, numBytes);
}

void TosaRefMemoryManager::Pool::Acquire()
{
    ARMNN_ASSERT_MSG(!m_Pointer, "TosaRefMemoryManager::Pool::Acquire() called when memory already acquired");
    m_Pointer = ::operator new(size_t(m_Size));
}

void TosaRefMemoryManager::Pool::Release()
{
    ARMNN_ASSERT_MSG(m_Pointer, "TosaRefMemoryManager::Pool::Release() called when memory not acquired");
    ::operator delete(m_Pointer);
    m_Pointer = nullptr;
}

}
