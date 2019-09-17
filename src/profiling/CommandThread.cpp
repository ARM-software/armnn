//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <atomic>
#include "CommandThread.hpp"

namespace armnn
{

namespace profiling
{

CommandThread::CommandThread(uint32_t timeout,
                             bool stopAfterTimeout,
                             CommandHandlerRegistry& commandHandlerRegistry,
                             PacketVersionResolver& packetVersionResolver,
                             IProfilingConnection& socketProfilingConnection)
    : m_Timeout(timeout)
    , m_StopAfterTimeout(stopAfterTimeout)
    , m_IsRunning(false)
    , m_CommandHandlerRegistry(commandHandlerRegistry)
    , m_PacketVersionResolver(packetVersionResolver)
    , m_SocketProfilingConnection(socketProfilingConnection)
{};

void CommandThread::WaitForPacket()
{
    do {
        try
        {
            Packet packet = m_SocketProfilingConnection.ReadPacket(m_Timeout);
            Version version = m_PacketVersionResolver.ResolvePacketVersion(packet.GetPacketId());

            CommandHandlerFunctor* commandHandlerFunctor =
                m_CommandHandlerRegistry.GetFunctor(packet.GetPacketId(), version.GetEncodedValue());
            commandHandlerFunctor->operator()(packet);
        }
        catch(armnn::TimeoutException)
        {
            if(m_StopAfterTimeout)
            {
                m_IsRunning.store(false, std::memory_order_relaxed);
                return;
            }
        }
        catch(...)
        {
            //might want to differentiate the errors more
            m_IsRunning.store(false, std::memory_order_relaxed);
            return;
        }

    } while(m_KeepRunning.load(std::memory_order_relaxed));

    m_IsRunning.store(false, std::memory_order_relaxed);
}

void CommandThread::Start()
{
    if (!m_CommandThread.joinable() && !IsRunning())
    {
        m_IsRunning.store(true, std::memory_order_relaxed);
        m_KeepRunning.store(true, std::memory_order_relaxed);
        m_CommandThread = std::thread(&CommandThread::WaitForPacket, this);
    }
}

void CommandThread::Stop()
{
    m_KeepRunning.store(false, std::memory_order_relaxed);
}

void CommandThread::Join()
{
    m_CommandThread.join();
}

bool CommandThread::IsRunning() const
{
    return m_IsRunning.load(std::memory_order_relaxed);
}

bool CommandThread::StopAfterTimeout(bool stopAfterTimeout)
{
    if (!IsRunning())
    {
        m_StopAfterTimeout = stopAfterTimeout;
        return true;
    }
    return false;
}

}//namespace profiling

}//namespace armnn