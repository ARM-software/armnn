//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandHandler.hpp"

namespace armnn
{

namespace profiling
{

void CommandHandler::Start(IProfilingConnection& profilingConnection)
{
    if (IsRunning())
    {
        return;
    }

    m_IsRunning.store(true);
    m_KeepRunning.store(true);
    m_CommandThread = std::thread(&CommandHandler::HandleCommands, this, std::ref(profilingConnection));
}

void CommandHandler::Stop()
{
    m_KeepRunning.store(false);

    if (m_CommandThread.joinable())
    {
        m_CommandThread.join();
    }
}

void CommandHandler::HandleCommands(IProfilingConnection& profilingConnection)
{
    do
    {
        try
        {
            Packet packet = profilingConnection.ReadPacket(m_Timeout);
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
        catch (const Exception& e)
        {
            // Log the error
            BOOST_LOG_TRIVIAL(warning) << "An error has occurred when handling a command: "
                                       << e.what();

            // Might want to differentiate the errors more
            m_KeepRunning.store(false);
        }
    }
    while (m_KeepRunning.load());

    m_IsRunning.store(false);
}

} // namespace profiling

} // namespace armnn
