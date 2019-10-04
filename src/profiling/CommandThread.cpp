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

void CommandThread::Start()
{
    if (IsRunning())
    {
        return;
    }

    m_IsRunning.store(true, std::memory_order_relaxed);
    m_KeepRunning.store(true, std::memory_order_relaxed);
    m_CommandThread = std::thread(&CommandThread::WaitForPacket, this);
}

void CommandThread::Stop()
{
    m_KeepRunning.store(false, std::memory_order_relaxed);

    if (m_CommandThread.joinable())
    {
        m_CommandThread.join();
    }
}

bool CommandThread::IsRunning() const
{
    return m_IsRunning.load(std::memory_order_relaxed);
}

void CommandThread::SetTimeout(uint32_t timeout)
{
    m_Timeout.store(timeout, std::memory_order_relaxed);
}

void CommandThread::SetStopAfterTimeout(bool stopAfterTimeout)
{
    m_StopAfterTimeout.store(stopAfterTimeout, std::memory_order_relaxed);
}

void CommandThread::WaitForPacket()
{
    do
    {
        try
        {
            Packet packet = m_SocketProfilingConnection.ReadPacket(m_Timeout);
            Version version = m_PacketVersionResolver.ResolvePacketVersion(packet.GetPacketId());

            CommandHandlerFunctor* commandHandlerFunctor =
                m_CommandHandlerRegistry.GetFunctor(packet.GetPacketId(), version.GetEncodedValue());
            BOOST_ASSERT(commandHandlerFunctor);
            commandHandlerFunctor->operator()(packet);
        }
        catch (const armnn::TimeoutException&)
        {
            if (m_StopAfterTimeout)
            {
                m_KeepRunning.store(false, std::memory_order_relaxed);
            }
        }
        catch (...)
        {
            // Might want to differentiate the errors more
            m_KeepRunning.store(false, std::memory_order_relaxed);
        }

    }
    while (m_KeepRunning.load(std::memory_order_relaxed));

    m_IsRunning.store(false, std::memory_order_relaxed);
}

} // namespace profiling

} // namespace armnn
