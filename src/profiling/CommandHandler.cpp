//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandHandler.hpp"
#include "ProfilingService.hpp"

#include <armnn/Logging.hpp>

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

    if (m_CommandThread.joinable())
    {
        m_CommandThread.join();
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
            arm::pipe::Packet packet = profilingConnection.ReadPacket(m_Timeout.load());

            if (packet.IsEmpty())
            {
                // Nothing to do, continue
                continue;
            }

            arm::pipe::Version version = m_PacketVersionResolver.ResolvePacketVersion(packet.GetPacketFamily(),
                                                                                      packet.GetPacketId());

            arm::pipe::CommandHandlerFunctor* commandHandlerFunctor =
                m_CommandHandlerRegistry.GetFunctor(packet.GetPacketFamily(),
                                                    packet.GetPacketId(),
                                                    version.GetEncodedValue());
            ARMNN_ASSERT(commandHandlerFunctor);
            commandHandlerFunctor->operator()(packet);
        }
        catch (const armnn::TimeoutException&)
        {
            if (m_StopAfterTimeout.load())
            {
                m_KeepRunning.store(false);
            }
        }
        catch (const arm::pipe::ProfilingException& e)
        {
            // Log the error and continue
            ARMNN_LOG(warning) << "An error has occurred when handling a command: " << e.what();
            // Did we get here because the socket failed?
            if ( !profilingConnection.IsOpen() )
            {
                // We're going to stop processing commands.
                // This will leave the thread idle. There is no mechanism to restart the profiling service when the
                // connection is lost.
                m_KeepRunning.store(false);
            }
        }
        catch (const Exception& e)
        {
            // Log the error and continue
            ARMNN_LOG(warning) << "An error has occurred when handling a command: " << e.what();
            // Did we get here because the socket failed?
            if ( !profilingConnection.IsOpen() )
            {
                // We're going to stop processing commands.
                // This will leave the thread idle. There is no mechanism to restart the profiling service when the
                // connection is lost.
                m_KeepRunning.store(false);
            }
        }
    }
    while (m_KeepRunning.load());

    m_IsRunning.store(false);
}

} // namespace profiling

} // namespace armnn
