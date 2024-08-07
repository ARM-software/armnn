//
// Copyright © 2017, 2024 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "RefMemoryManager.hpp"

#include <armnn/Exceptions.hpp>

#include <algorithm>

namespace armnn
{

RefMemoryManager::RefMemoryManager()
{}

RefMemoryManager::~RefMemoryManager()
{}

RefMemoryManager::Pool* RefMemoryManager::Manage(unsigned int numBytes)
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

void RefMemoryManager::Allocate(RefMemoryManager::Pool* pool)
{
    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(pool, "Null memory manager passed to RefMemoryManager.");
    m_FreePools.push_back(pool);
}

void* RefMemoryManager::GetPointer(RefMemoryManager::Pool* pool)
{
    return pool->GetPointer();
}

void RefMemoryManager::Acquire()
{
    for (Pool &pool: m_Pools)
    {
         pool.Acquire();
    }
}

void RefMemoryManager::Release()
{
    for (Pool &pool: m_Pools)
    {
         pool.Release();
    }
}

RefMemoryManager::Pool::Pool(unsigned int numBytes)
    : m_Size(numBytes),
      m_Pointer(nullptr)
{}

RefMemoryManager::Pool::~Pool()
{
    if (m_Pointer)
    {
        Release();
    }
}

void* RefMemoryManager::Pool::GetPointer()
{
    ARMNN_THROW_MSG_IF_FALSE(m_Pointer, RuntimeException,
                             "RefMemoryManager::Pool::GetPointer() called when memory not acquired");
    return m_Pointer;
}

void RefMemoryManager::Pool::Reserve(unsigned int numBytes)
{
    ARMNN_THROW_MSG_IF_FALSE(!m_Pointer, RuntimeException,
                             "RefMemoryManager::Pool::Reserve() cannot be called after memory acquired");
    m_Size = std::max(m_Size, numBytes);
}

void RefMemoryManager::Pool::Acquire()
{
    ARMNN_THROW_MSG_IF_FALSE(!m_Pointer, RuntimeException,
                             "RefMemoryManager::Pool::Acquire() called when memory already acquired");
    m_Pointer = ::operator new(size_t(m_Size));
}

void RefMemoryManager::Pool::Release()
{
    ARMNN_THROW_MSG_IF_FALSE(m_Pointer, RuntimeException,
                             "RefMemoryManager::Pool::Release() called when memory not acquired");
    ::operator delete(m_Pointer);
    m_Pointer = nullptr;
}

}
