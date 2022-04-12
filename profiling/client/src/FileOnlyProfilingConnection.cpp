//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FileOnlyProfilingConnection.hpp"

#include <common/include/Constants.hpp>
#include <common/include/ProfilingException.hpp>
#include <common/include/PacketVersionResolver.hpp>

#include <algorithm>
#include <iostream>

#if defined(ARMNN_DISABLE_THREADS)
#include <common/include/IgnoreUnused.hpp>
#endif

namespace arm
{

namespace pipe
{

std::vector<uint32_t> StreamMetaDataProcessor::GetHeadersAccepted()
{
    std::vector<uint32_t> headers;
    headers.push_back(m_MetaDataPacketHeader);
    return headers;
}

void StreamMetaDataProcessor::HandlePacket(const arm::pipe::Packet& packet)
{
    if (packet.GetHeader() != m_MetaDataPacketHeader)
    {
        throw arm::pipe::ProfilingException("StreamMetaDataProcessor can only handle Stream Meta Data Packets");
    }
    // determine the endianness of the protocol
    TargetEndianness endianness;
    if (ToUint32(packet.GetData(),TargetEndianness::BeWire) == arm::pipe::PIPE_MAGIC)
    {
        endianness = TargetEndianness::BeWire;
    }
    else if (ToUint32(packet.GetData(), TargetEndianness::LeWire) == arm::pipe::PIPE_MAGIC)
    {
        endianness = TargetEndianness::LeWire;
    }
    else
    {
        throw arm::pipe::ProfilingException("Protocol read error. Unable to read the PIPE_MAGIC value.");
    }
    m_FileOnlyProfilingConnection->SetEndianess(endianness);
    // send back the acknowledgement
    std::unique_ptr<unsigned char[]> uniqueNullPtr = nullptr;
    arm::pipe::Packet returnPacket(0x10000, 0, uniqueNullPtr);
    m_FileOnlyProfilingConnection->ReturnPacket(returnPacket);
}

uint32_t StreamMetaDataProcessor::ToUint32(const unsigned char* data, TargetEndianness endianness)
{
    // Extract the first 4 bytes starting at data and push them into a 32bit integer based on the
    // specified endianness.
    if (endianness == TargetEndianness::BeWire)
    {
        return static_cast<uint32_t>(data[0]) << 24 | static_cast<uint32_t>(data[1]) << 16 |
               static_cast<uint32_t>(data[2]) << 8 | static_cast<uint32_t>(data[3]);
    }
    else
    {
        return static_cast<uint32_t>(data[3]) << 24 | static_cast<uint32_t>(data[2]) << 16 |
               static_cast<uint32_t>(data[1]) << 8 | static_cast<uint32_t>(data[0]);
    }
}

FileOnlyProfilingConnection::~FileOnlyProfilingConnection()
{
    try
    {
        Close();
    }
    catch (...)
    {
        // do nothing
    }
}

bool FileOnlyProfilingConnection::IsOpen() const
{
    // This type of connection is always open.
    return true;
}

void FileOnlyProfilingConnection::Close()
{
    // Dump any unread packets out of the queue.
    size_t initialSize = m_PacketQueue.size();
    for (size_t i = 0; i < initialSize; ++i)
    {
        m_PacketQueue.pop();
    }
    // dispose of the processing thread
    m_KeepRunning.store(false);
#if !defined(ARMNN_DISABLE_THREADS)
    if (m_LocalHandlersThread.joinable())
    {
        // make sure the thread wakes up and sees it has to stop
        m_ConditionPacketReadable.notify_one();
        m_LocalHandlersThread.join();
    }
#endif
}

bool FileOnlyProfilingConnection::WritePacket(const unsigned char* buffer, uint32_t length)
{
    ARM_PIPE_ASSERT(buffer);
    arm::pipe::Packet packet = ReceivePacket(buffer, length);
    ForwardPacketToHandlers(packet);
    return true;
}

void FileOnlyProfilingConnection::ReturnPacket(arm::pipe::Packet& packet)
{
    {
#if !defined(ARMNN_DISABLE_THREADS)
        std::lock_guard<std::mutex> lck(m_PacketAvailableMutex);
#endif
        m_PacketQueue.push(std::move(packet));
    }
#if !defined(ARMNN_DISABLE_THREADS)
    m_ConditionPacketAvailable.notify_one();
#endif
}

arm::pipe::Packet FileOnlyProfilingConnection::ReadPacket(uint32_t timeout)
{
#if !defined(ARMNN_DISABLE_THREADS)
    std::unique_lock<std::mutex> lck(m_PacketAvailableMutex);

    // Here we are using m_PacketQueue.empty() as a predicate variable
    // The conditional variable will wait until packetQueue is not empty or until a timeout
    if (!m_ConditionPacketAvailable.wait_for(lck,
                                             std::chrono::milliseconds(timeout),
                                             [&]{return !m_PacketQueue.empty();}))
    {
        arm::pipe::Packet empty;
        return empty;
    }
#else
    IgnoreUnused(timeout);
#endif

    arm::pipe::Packet returnedPacket = std::move(m_PacketQueue.front());
    m_PacketQueue.pop();
    return returnedPacket;
}

void FileOnlyProfilingConnection::Fail(const std::string& errorMessage)
{
    Close();
    throw arm::pipe::ProfilingException(errorMessage);
}

/// Adds a local packet handler to the FileOnlyProfilingConnection. Invoking this will start
/// a processing thread that will ensure that processing of packets will happen on a separate
/// thread from the profiling services send thread and will therefore protect against the
/// profiling message buffer becoming exhausted because packet handling slows the dispatch.
void FileOnlyProfilingConnection::AddLocalPacketHandler(ILocalPacketHandlerSharedPtr localPacketHandler)
{
    m_PacketHandlers.push_back(std::move(localPacketHandler));
    ILocalPacketHandlerSharedPtr localCopy = m_PacketHandlers.back();
    localCopy->SetConnection(this);
    if (localCopy->GetHeadersAccepted().empty())
    {
        //this is a universal handler
        m_UniversalHandlers.push_back(localCopy);
    }
    else
    {
        for (uint32_t header : localCopy->GetHeadersAccepted())
        {
            auto iter = m_IndexedHandlers.find(header);
            if (iter == m_IndexedHandlers.end())
            {
                std::vector<ILocalPacketHandlerSharedPtr> handlers;
                handlers.push_back(localCopy);
                m_IndexedHandlers.emplace(std::make_pair(header, handlers));
            }
            else
            {
                iter->second.push_back(localCopy);
            }
        }
    }
}

void FileOnlyProfilingConnection::StartProcessingThread()
{
    // check if the thread has already started
    if (m_IsRunning.load())
    {
        return;
    }
    // make sure if there was one running before it is joined
#if !defined(ARMNN_DISABLE_THREADS)
    if (m_LocalHandlersThread.joinable())
    {
        m_LocalHandlersThread.join();
    }
#endif
    m_IsRunning.store(true);
    m_KeepRunning.store(true);
#if !defined(ARMNN_DISABLE_THREADS)
    m_LocalHandlersThread = std::thread(&FileOnlyProfilingConnection::ServiceLocalHandlers, this);
#endif
}

void FileOnlyProfilingConnection::ForwardPacketToHandlers(arm::pipe::Packet& packet)
{
    if (m_PacketHandlers.empty())
    {
        return;
    }
    if (!m_KeepRunning.load())
    {
        return;
    }
    {
#if !defined(ARMNN_DISABLE_THREADS)
        std::unique_lock<std::mutex> readableListLock(m_ReadableMutex);
#endif
        if (!m_KeepRunning.load())
        {
            return;
        }
        m_ReadableList.push(std::move(packet));
    }
#if !defined(ARMNN_DISABLE_THREADS)
    m_ConditionPacketReadable.notify_one();
#endif
}

void FileOnlyProfilingConnection::ServiceLocalHandlers()
{
    do
    {
        arm::pipe::Packet returnedPacket;
        bool readPacket = false;
        {   // only lock while we are taking the packet off the incoming list
#if !defined(ARMNN_DISABLE_THREADS)
            std::unique_lock<std::mutex> lck(m_ReadableMutex);
#endif
            if (m_Timeout < 0)
            {
#if !defined(ARMNN_DISABLE_THREADS)
                m_ConditionPacketReadable.wait(lck,
                                               [&] { return !m_ReadableList.empty(); });
#endif
            }
            else
            {
#if !defined(ARMNN_DISABLE_THREADS)
                m_ConditionPacketReadable.wait_for(lck,
                                                   std::chrono::milliseconds(std::max(m_Timeout, 1000)),
                                                   [&] { return !m_ReadableList.empty(); });
#endif
            }
            if (m_KeepRunning.load())
            {
                if (!m_ReadableList.empty())
                {
                    returnedPacket = std::move(m_ReadableList.front());
                    m_ReadableList.pop();
                    readPacket = true;
                }
            }
            else
            {
                ClearReadableList();
            }
        }
        if (m_KeepRunning.load() && readPacket)
        {
            DispatchPacketToHandlers(returnedPacket);
        }
    } while (m_KeepRunning.load());
    // make sure the readable list is cleared
    ClearReadableList();
    m_IsRunning.store(false);
}

void FileOnlyProfilingConnection::ClearReadableList()
{
    // make sure the incoming packet queue gets emptied
    size_t initialSize = m_ReadableList.size();
    for (size_t i = 0; i < initialSize; ++i)
    {
        m_ReadableList.pop();
    }
}

void FileOnlyProfilingConnection::DispatchPacketToHandlers(const arm::pipe::Packet& packet)
{
    for (auto& delegate : m_UniversalHandlers)
    {
        delegate->HandlePacket(packet);
    }
    auto iter = m_IndexedHandlers.find(packet.GetHeader());
    if (iter != m_IndexedHandlers.end())
    {
        for (auto& delegate : iter->second)
        {
            try
            {
                delegate->HandlePacket(packet);
            }
            catch (const arm::pipe::ProfilingException& ex)
            {
                Fail(ex.what());
            }
            catch (const std::exception& ex)
            {
                Fail(ex.what());
            }
            catch (...)
            {
                Fail("handler failed");
            }
        }
    }
}

}    // namespace pipe

}    // namespace arm
