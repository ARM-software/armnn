//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandHandler.hpp"

#include <boost/log/trivial.hpp>

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
            Packet packet = profilingConnection.ReadPacket(m_Timeout.load());

            if (packet.IsEmpty())
            {
                // Nothing to do, continue
                continue;
            }

            Version version = m_PacketVersionResolver.ResolvePacketVersion(packet.GetPacketId());

            CommandHandlerFunctor* commandHandlerFunctor =
                m_CommandHandlerRegistry.GetFunctor(packet.GetPacketId(), version.GetEncodedValue());
            BOOST_ASSERT(commandHandlerFunctor);
            commandHandlerFunctor->operator()(packet);
        }
        catch (const armnn::TimeoutException&)
        {
            if (m_StopAfterTimeout.load())
            {
                m_KeepRunning.store(false);
            }
        }
        catch (const Exception& e)
        {
            // Log the error and continue
            BOOST_LOG_TRIVIAL(warning) << "An error has occurred when handling a command: " << e.what() << std::endl;
        }
    }
    while (m_KeepRunning.load());

    m_IsRunning.store(false);
}

} // namespace profiling

} // namespace armnn
