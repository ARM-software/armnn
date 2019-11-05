//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BufferManager.hpp"
#include "PacketBuffer.hpp"
#include "ProfilingUtils.hpp"

#include <armnn/Exceptions.hpp>

namespace armnn
{

namespace profiling
{

BufferManager::BufferManager(unsigned int numberOfBuffers, unsigned int maxPacketSize)
    : m_MaxBufferSize(maxPacketSize)
{
    m_AvailableList.reserve(numberOfBuffers);
    for (unsigned int i = 0; i < numberOfBuffers; ++i)
    {
        IPacketBufferPtr buffer = std::make_unique<PacketBuffer>(maxPacketSize);
        m_AvailableList.emplace_back(std::move(buffer));
    }
    m_ReadableList.reserve(numberOfBuffers);
}

IPacketBufferPtr BufferManager::Reserve(unsigned int requestedSize, unsigned int& reservedSize)
{
    reservedSize = 0;
    std::unique_lock<std::mutex> availableListLock(m_AvailableMutex, std::defer_lock);
    if (requestedSize > m_MaxBufferSize)
    {
        return nullptr;
    }
    availableListLock.lock();
    if (m_AvailableList.empty())
    {
        availableListLock.unlock();
        return nullptr;
    }
    IPacketBufferPtr buffer = std::move(m_AvailableList.back());
    m_AvailableList.pop_back();
    availableListLock.unlock();
    reservedSize = requestedSize;
    return buffer;
}

void BufferManager::Commit(IPacketBufferPtr& packetBuffer, unsigned int size)
{
    std::unique_lock<std::mutex> readableListLock(m_ReadableMutex, std::defer_lock);
    packetBuffer->Commit(size);
    readableListLock.lock();
    m_ReadableList.push_back(std::move(packetBuffer));
    readableListLock.unlock();
    m_ReadDataAvailable.notify_one();
}

void BufferManager::Release(IPacketBufferPtr& packetBuffer)
{
    std::unique_lock<std::mutex> availableListLock(m_AvailableMutex, std::defer_lock);
    packetBuffer->Release();
    availableListLock.lock();
    m_AvailableList.push_back(std::move(packetBuffer));
    availableListLock.unlock();
}

IPacketBufferPtr BufferManager::GetReadableBuffer()
{
    std::unique_lock<std::mutex> readableListLock(m_ReadableMutex);
    if (!m_ReadableList.empty())
    {
        IPacketBufferPtr buffer = std::move(m_ReadableList.back());
        m_ReadableList.pop_back();
        readableListLock.unlock();
        return buffer;
    }
    return nullptr;
}

void BufferManager::MarkRead(IPacketBufferPtr& packetBuffer)
{
    std::unique_lock<std::mutex> availableListLock(m_AvailableMutex, std::defer_lock);
    packetBuffer->MarkRead();
    availableListLock.lock();
    m_AvailableList.push_back(std::move(packetBuffer));
    availableListLock.unlock();
}

} // namespace profiling

} // namespace armnn
