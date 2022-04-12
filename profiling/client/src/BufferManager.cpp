//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BufferManager.hpp"
#include "PacketBuffer.hpp"

namespace arm
{

namespace pipe
{

BufferManager::BufferManager(unsigned int numberOfBuffers, unsigned int maxPacketSize)
    : m_MaxBufferSize(maxPacketSize),
      m_NumberOfBuffers(numberOfBuffers),
      m_MaxNumberOfBuffers(numberOfBuffers * 3),
      m_CurrentNumberOfBuffers(numberOfBuffers)
{
    Initialize();
}

IPacketBufferPtr BufferManager::Reserve(unsigned int requestedSize, unsigned int& reservedSize)
{
    reservedSize = 0;
#if !defined(ARMNN_DISABLE_THREADS)
    std::unique_lock<std::mutex> availableListLock(m_AvailableMutex, std::defer_lock);
#endif
    if (requestedSize > m_MaxBufferSize)
    {
        return nullptr;
    }
#if !defined(ARMNN_DISABLE_THREADS)
    availableListLock.lock();
#endif
    if (m_AvailableList.empty())
    {
        if (m_CurrentNumberOfBuffers < m_MaxNumberOfBuffers)
        {
            // create a temporary overflow/surge buffer and hand it back
            m_CurrentNumberOfBuffers++;
#if !defined(ARMNN_DISABLE_THREADS)
            availableListLock.unlock();
#endif
            IPacketBufferPtr buffer = std::make_unique<PacketBuffer>(m_MaxBufferSize);
            reservedSize = requestedSize;
            return buffer;
        }
        else
        {
            // we have totally busted the limit. call a halt to new memory allocations.
#if !defined(ARMNN_DISABLE_THREADS)
            availableListLock.unlock();
#endif
            return nullptr;
        }
    }
    IPacketBufferPtr buffer = std::move(m_AvailableList.back());
    m_AvailableList.pop_back();
#if !defined(ARMNN_DISABLE_THREADS)
    availableListLock.unlock();
#endif
    reservedSize = requestedSize;
    return buffer;
}

void BufferManager::Commit(IPacketBufferPtr& packetBuffer, unsigned int size, bool notifyConsumer)
{
#if !defined(ARMNN_DISABLE_THREADS)
    std::unique_lock<std::mutex> readableListLock(m_ReadableMutex, std::defer_lock);
#endif
    packetBuffer->Commit(size);
#if !defined(ARMNN_DISABLE_THREADS)
    readableListLock.lock();
#endif
    m_ReadableList.push(std::move(packetBuffer));
#if !defined(ARMNN_DISABLE_THREADS)
    readableListLock.unlock();
#endif
    if (notifyConsumer)
    {
        FlushReadList();
    }
}

void BufferManager::Initialize()
{
    m_AvailableList.reserve(m_NumberOfBuffers);
    m_CurrentNumberOfBuffers = m_NumberOfBuffers;
    for (unsigned int i = 0; i < m_NumberOfBuffers; ++i)
    {
        IPacketBufferPtr buffer = std::make_unique<PacketBuffer>(m_MaxBufferSize);
        m_AvailableList.emplace_back(std::move(buffer));
    }
}

void BufferManager::Release(IPacketBufferPtr& packetBuffer)
{
#if !defined(ARMNN_DISABLE_THREADS)
    std::unique_lock<std::mutex> availableListLock(m_AvailableMutex, std::defer_lock);
#endif
    packetBuffer->Release();
#if !defined(ARMNN_DISABLE_THREADS)
    availableListLock.lock();
#endif
    if (m_AvailableList.size() <= m_NumberOfBuffers)
    {
        m_AvailableList.push_back(std::move(packetBuffer));
    }
    else
    {
        // we have been handed a temporary overflow/surge buffer get rid of it
        packetBuffer->Destroy();
        if (m_CurrentNumberOfBuffers > m_NumberOfBuffers)
        {
            --m_CurrentNumberOfBuffers;
        }
    }
#if !defined(ARMNN_DISABLE_THREADS)
    availableListLock.unlock();
#endif
}

void BufferManager::Reset()
{
    //This method should only be called once all threads have been joined
#if !defined(ARMNN_DISABLE_THREADS)
    std::lock_guard<std::mutex> readableListLock(m_ReadableMutex);
    std::lock_guard<std::mutex> availableListLock(m_AvailableMutex);
#endif

    m_AvailableList.clear();
    std::queue<IPacketBufferPtr>().swap(m_ReadableList);

    Initialize();
}

IPacketBufferPtr BufferManager::GetReadableBuffer()
{
#if !defined(ARMNN_DISABLE_THREADS)
    std::unique_lock<std::mutex> readableListLock(m_ReadableMutex);
#endif
    if (!m_ReadableList.empty())
    {
        IPacketBufferPtr buffer = std::move(m_ReadableList.front());
        m_ReadableList.pop();
#if !defined(ARMNN_DISABLE_THREADS)
        readableListLock.unlock();
#endif
        return buffer;
    }
    return nullptr;
}

void BufferManager::MarkRead(IPacketBufferPtr& packetBuffer)
{
#if !defined(ARMNN_DISABLE_THREADS)
    std::unique_lock<std::mutex> availableListLock(m_AvailableMutex, std::defer_lock);
#endif
    packetBuffer->MarkRead();
#if !defined(ARMNN_DISABLE_THREADS)
    availableListLock.lock();
#endif
    if (m_AvailableList.size() <= m_NumberOfBuffers)
    {
        m_AvailableList.push_back(std::move(packetBuffer));
    }
    else
    {
        // we have been handed a temporary overflow/surge buffer get rid of it
        packetBuffer->Destroy();
        if (m_CurrentNumberOfBuffers > m_NumberOfBuffers)
        {
            --m_CurrentNumberOfBuffers;
        }
    }
#if !defined(ARMNN_DISABLE_THREADS)
    availableListLock.unlock();
#endif
}

void BufferManager::SetConsumer(IConsumer* consumer)
{
    m_Consumer = consumer;
}

void BufferManager::FlushReadList()
{
    // notify consumer that packet is ready to read
    if (m_Consumer != nullptr)
    {
        m_Consumer->SetReadyToRead();
    }
}

} // namespace pipe

} // namespace arm
