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
        std::unique_ptr<IPacketBuffer> buffer = std::make_unique<PacketBuffer>(maxPacketSize);
        m_AvailableList.emplace_back(std::move(buffer));
    }
    m_ReadableList.reserve(numberOfBuffers);
}

std::unique_ptr<IPacketBuffer> BufferManager::Reserve(unsigned int requestedSize, unsigned int& reservedSize)
{
    std::unique_lock<std::mutex> availableListLock(m_AvailableMutex, std::defer_lock);
    if (requestedSize > m_MaxBufferSize)
    {
        throw armnn::RuntimeException("Maximum buffer size that can be requested is [" +
            std::to_string(m_MaxBufferSize) + "] bytes");
    }
    availableListLock.lock();
    if (m_AvailableList.empty())
    {
        throw armnn::profiling::BufferExhaustion("Buffer not available");
    }
    std::unique_ptr<IPacketBuffer> buffer = std::move(m_AvailableList.back());
    m_AvailableList.pop_back();
    availableListLock.unlock();
    reservedSize = requestedSize;
    return buffer;
}

void BufferManager::Commit(std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int size)
{
    std::unique_lock<std::mutex> readableListLock(m_ReadableMutex, std::defer_lock);
    packetBuffer->Commit(size);
    readableListLock.lock();
    m_ReadableList.push_back(std::move(packetBuffer));
    readableListLock.unlock();
    m_ReadDataAvailable.notify_one();
}

void BufferManager::Release(std::unique_ptr<IPacketBuffer>& packetBuffer)
{
    std::unique_lock<std::mutex> availableListLock(m_AvailableMutex, std::defer_lock);
    packetBuffer->Release();
    availableListLock.lock();
    m_AvailableList.push_back(std::move(packetBuffer));
    availableListLock.unlock();
}

std::unique_ptr<IPacketBuffer> BufferManager::GetReadableBuffer()
{
    std::unique_lock<std::mutex> readableListLock(m_ReadableMutex);
    if (!m_ReadableList.empty())
    {
        std::unique_ptr<IPacketBuffer> buffer = std::move(m_ReadableList.back());
        m_ReadableList.pop_back();
        readableListLock.unlock();
        return buffer;
    }
    return nullptr;
}

void BufferManager::MarkRead(std::unique_ptr<IPacketBuffer>& packetBuffer)
{
    std::unique_lock<std::mutex> availableListLock(m_AvailableMutex, std::defer_lock);
    packetBuffer->MarkRead();
    availableListLock.lock();
    m_AvailableList.push_back(std::move(packetBuffer));
    availableListLock.unlock();
}

} // namespace profiling

} // namespace armnn
