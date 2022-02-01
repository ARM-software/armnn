//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnTestUtils/MockMemoryManager.hpp"

namespace armnn
{

MockMemoryManager::MockMemoryManager()
{}

MockMemoryManager::~MockMemoryManager()
{}

MockMemoryManager::Pool* MockMemoryManager::Manage(unsigned int numBytes)
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

void MockMemoryManager::Allocate(MockMemoryManager::Pool* pool)
{
    m_FreePools.push_back(pool);
}

void* MockMemoryManager::GetPointer(MockMemoryManager::Pool* pool)
{
    return pool->GetPointer();
}

void MockMemoryManager::Acquire()
{
    for (Pool& pool : m_Pools)
    {
        pool.Acquire();
    }
}

void MockMemoryManager::Release()
{
    for (Pool& pool : m_Pools)
    {
        pool.Release();
    }
}

MockMemoryManager::Pool::Pool(unsigned int numBytes)
    : m_Size(numBytes)
    , m_Pointer(nullptr)
{}

MockMemoryManager::Pool::~Pool()
{
    if (m_Pointer)
    {
        Release();
    }
}

void* MockMemoryManager::Pool::GetPointer()
{
    return m_Pointer;
}

void MockMemoryManager::Pool::Reserve(unsigned int numBytes)
{
    m_Size = std::max(m_Size, numBytes);
}

void MockMemoryManager::Pool::Acquire()
{
    m_Pointer = ::operator new(size_t(m_Size));
}

void MockMemoryManager::Pool::Release()
{
    ::operator delete(m_Pointer);
    m_Pointer = nullptr;
}

}    // namespace armnn
