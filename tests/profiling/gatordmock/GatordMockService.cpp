//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatordMockService.hpp"

#include <common/include/Assert.hpp>
#include <common/include/CommandHandlerRegistry.hpp>
#include <common/include/CommonProfilingUtils.hpp>
#include <common/include/PacketVersionResolver.hpp>
#include <common/include/NetworkSockets.hpp>

#include <cerrno>
#include <iostream>
#include <string>

namespace armnn
{

namespace gatordmock
{

void GatordMockService::SendConnectionAck()
{
    if (m_EchoPackets)
    {
        std::cout << "Sending connection acknowledgement." << std::endl;
    }
    // The connection ack packet is an empty data packet with packetId == 1.
    m_BasePipeServer.get()->SendPacket(0, 1, nullptr, 0);
}

void GatordMockService::SendRequestCounterDir()
{
    if (m_EchoPackets)
    {
        std::cout << "Sending connection acknowledgement." << std::endl;
    }
    // The request counter directory packet is an empty data packet with packetId == 3.
    m_BasePipeServer.get()->SendPacket(0, 3, nullptr, 0);
}

void GatordMockService::SendActivateTimelinePacket()
{
    if (m_EchoPackets)
    {
        std::cout << "Sending activate timeline packet." << std::endl;
    }
    // The activate timeline packet is an empty data packet with packetId == 6.
    m_BasePipeServer.get()->SendPacket(0, 6, nullptr, 0);
}

void GatordMockService::SendDeactivateTimelinePacket()
{
    if (m_EchoPackets)
    {
        std::cout << "Sending deactivate timeline packet." << std::endl;
    }
    // The deactivate timeline packet is an empty data packet with packetId == 7.
    m_BasePipeServer.get()->SendPacket(0, 7, nullptr, 0);
}

bool GatordMockService::LaunchReceivingThread()
{
    if (m_EchoPackets)
    {
        std::cout << "Launching receiving thread." << std::endl;
    }
    // At this point we want to make the socket non blocking.
    if (!m_BasePipeServer.get()->SetNonBlocking())
    {
        m_BasePipeServer.get()->Close();
        std::cerr << "Failed to set socket as non blocking: " << strerror(errno) << std::endl;
        return false;
    }
    m_ListeningThread = std::thread(&GatordMockService::ReceiveLoop, this);
    return true;
}

void GatordMockService::WaitForReceivingThread()
{
    // The receiving thread may already have died.
    if (m_CloseReceivingThread != true)
    {
        m_CloseReceivingThread.store(true);
    }
    // Check that the receiving thread is running
    if (m_ListeningThread.joinable())
    {
        // Wait for the receiving thread to complete operations
        m_ListeningThread.join();
    }

    if(m_EchoPackets)
    {
        m_TimelineDecoder.print();
    }
}

bool GatordMockService::WaitForStreamMetaData()
{
    return m_BasePipeServer->WaitForStreamMetaData();
}

void GatordMockService::SendPeriodicCounterSelectionList(uint32_t period, std::vector<uint16_t> counters)
{
    // The packet body consists of a UINT32 representing the period following by zero or more
    // UINT16's representing counter UID's. If the list is empty it implies all counters are to
    // be disabled.

    if (m_EchoPackets)
    {
        std::cout << "SendPeriodicCounterSelectionList: Period=" << std::dec << period << "uSec" << std::endl;
        std::cout << "List length=" << counters.size() << std::endl;
    }
    // Start by calculating the length of the packet body in bytes. This will be at least 4.
    uint32_t dataLength = static_cast<uint32_t>(4 + (counters.size() * 2));

    std::unique_ptr<unsigned char[]> uniqueData = std::make_unique<unsigned char[]>(dataLength);
    unsigned char* data                         = reinterpret_cast<unsigned char*>(uniqueData.get());

    uint32_t offset = 0;
    arm::pipe::WriteUint32(data, offset, period);
    offset += 4;
    for (std::vector<uint16_t>::iterator it = counters.begin(); it != counters.end(); ++it)
    {
        arm::pipe::WriteUint16(data, offset, *it);
        offset += 2;
    }

    // Send the packet.
    m_BasePipeServer.get()->SendPacket(0, 4, data, dataLength);
    // There will be an echo response packet sitting in the receive thread. PeriodicCounterSelectionResponseHandler
    // should deal with it.
}

void GatordMockService::WaitCommand(uint32_t timeout)
{
    // Wait for a maximum of timeout microseconds or if the receive thread has closed.
    // There is a certain level of rounding involved in this timing.
    uint32_t iterations = timeout / 1000;
    std::cout << std::dec << "Wait command with timeout of " << timeout << " iterations =  " << iterations << std::endl;
    uint32_t count = 0;
    while ((this->ReceiveThreadRunning() && (count < iterations)))
    {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
        ++count;
    }
    if (m_EchoPackets)
    {
        std::cout << std::dec << "Wait command with timeout of " << timeout << " microseconds completed. " << std::endl;
    }
}

void GatordMockService::ReceiveLoop()
{
    m_CloseReceivingThread.store(false);
    while (!m_CloseReceivingThread.load())
    {
        try
        {
            arm::pipe::Packet packet = m_BasePipeServer.get()->WaitForPacket(500);

            arm::pipe::PacketVersionResolver packetVersionResolver;

            arm::pipe::Version version =
                packetVersionResolver.ResolvePacketVersion(packet.GetPacketFamily(), packet.GetPacketId());

            arm::pipe::CommandHandlerFunctor* commandHandlerFunctor = m_HandlerRegistry.GetFunctor(
                                                                        packet.GetPacketFamily(),
                                                                        packet.GetPacketId(),
                                                                        version.GetEncodedValue());



            ARM_PIPE_ASSERT(commandHandlerFunctor);
            commandHandlerFunctor->operator()(packet);
        }
        catch (const arm::pipe::TimeoutException&)
        {
            // In this case we ignore timeouts and and keep trying to receive.
        }
        catch (const arm::pipe::InvalidArgumentException& e)
        {
            // We couldn't find a functor to handle the packet?
            std::cerr << "Packet received that could not be processed: " << e.what() << std::endl;
        }
        catch (const arm::pipe::ProfilingException& e)
        {
            // A runtime exception occurred which means we must exit the loop.
            std::cerr << "Receive thread closing: " << e.what() << std::endl;
            m_CloseReceivingThread.store(true);
        }
    }
}

}    // namespace gatordmock

}    // namespace armnn
