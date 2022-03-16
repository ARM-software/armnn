//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

// local includes
#include "PeriodicCounterCaptureCommandHandler.hpp"
#include "StreamMetadataCommandHandler.hpp"
#include "StubCommandHandler.hpp"

#include <common/include/CommandHandlerRegistry.hpp>
#include <common/include/Packet.hpp>
#include <common/include/PacketVersionResolver.hpp>

#include <server/include/basePipeServer/BasePipeServer.hpp>

#include <server/include/timelineDecoder/DirectoryCaptureCommandHandler.hpp>
#include <server/include/timelineDecoder/TimelineDecoder.hpp>
#include <server/include/timelineDecoder/TimelineCaptureCommandHandler.hpp>
#include <server/include/timelineDecoder/TimelineDirectoryCaptureCommandHandler.hpp>

#include <atomic>
#include <string>
#include <thread>

namespace armnn
{

namespace gatordmock
{

///  A class that implements a Mock Gatord server. It will listen on a specified Unix domain socket (UDS)
///  namespace for client connections. It will then allow opertaions to manage counters while receiving counter data.
class GatordMockService
{
public:
    /// @param registry reference to a command handler registry.
    /// @param echoPackets if true the raw packets will be printed to stdout.
    GatordMockService(std::unique_ptr<arm::pipe::BasePipeServer> clientConnection, bool echoPackets)
            : m_BasePipeServer(std::move(clientConnection))
            , m_EchoPackets(echoPackets)
            , m_CloseReceivingThread(false)
            , m_PacketVersionResolver()
            , m_HandlerRegistry()
            , m_TimelineDecoder()
            , m_CounterCaptureCommandHandler(
                    0, 4, m_PacketVersionResolver.ResolvePacketVersion(0, 4).GetEncodedValue(), !echoPackets)
            , m_StreamMetadataCommandHandler(
                    0, 0, m_PacketVersionResolver.ResolvePacketVersion(0, 0).GetEncodedValue(), !echoPackets)
            // This stub lets us ignore any counter capture packets we receive without throwing an error
            , m_StubCommandHandler(3, 0, m_PacketVersionResolver.ResolvePacketVersion(0, 3).GetEncodedValue())
            , m_DirectoryCaptureCommandHandler(
                    "ARMNN", 0, 2, m_PacketVersionResolver.ResolvePacketVersion(0, 2).GetEncodedValue(), !echoPackets)
            , m_TimelineCaptureCommandHandler(
                    1, 1, m_PacketVersionResolver.ResolvePacketVersion(1, 1).GetEncodedValue(), m_TimelineDecoder)
            , m_TimelineDirectoryCaptureCommandHandler(
                    1, 0, m_PacketVersionResolver.ResolvePacketVersion(1, 0).GetEncodedValue(),
                    m_TimelineCaptureCommandHandler, !echoPackets)
    {
        m_TimelineDecoder.SetDefaultCallbacks();

        m_HandlerRegistry.RegisterFunctor(&m_CounterCaptureCommandHandler);
        m_HandlerRegistry.RegisterFunctor(&m_StreamMetadataCommandHandler);
        m_HandlerRegistry.RegisterFunctor(&m_StubCommandHandler);
        m_HandlerRegistry.RegisterFunctor(&m_DirectoryCaptureCommandHandler);
        m_HandlerRegistry.RegisterFunctor(&m_TimelineDirectoryCaptureCommandHandler);
        m_HandlerRegistry.RegisterFunctor(&m_TimelineCaptureCommandHandler);
    }

    GatordMockService(const GatordMockService&) = delete;
    GatordMockService& operator=(const GatordMockService&) = delete;

    GatordMockService(GatordMockService&&) = delete;
    GatordMockService& operator=(GatordMockService&&) = delete;

    /// Once the connection is open wait to receive the stream meta data packet from the client. Reading this
    /// packet differs from others as we need to determine endianness.
    /// @return true only if a valid stream met data packet has been received.
    bool WaitForStreamMetaData();

    /// Send a connection acknowledged packet back to the client.
    void SendConnectionAck();

    /// Send a request counter directory packet back to the client.
    void SendRequestCounterDir();

    /// Send a activate timeline packet back to the client.
    void SendActivateTimelinePacket();

    /// Send a deactivate timeline packet back to the client.
    void SendDeactivateTimelinePacket();

    /// Start the thread that will receive all packets and print them nicely to stdout.
    bool LaunchReceivingThread();

    /// Return the total number of periodic counter capture packets received since the receive thread started.
    /// @return number of periodic counter capture packets received.
    uint32_t GetPacketsReceivedCount()
    {
        return m_PacketsReceivedCount.load(std::memory_order_acquire);
    }

    /// This is a placeholder method to prevent main exiting. It can be removed once the
    /// command handling code is added.
    void WaitForReceivingThread();

    // @return true only if the receive thread is closed or closing.
    bool ReceiveThreadRunning()
    {
        return !m_CloseReceivingThread.load();
    }

    /// Send the counter list to ArmNN.
    void SendPeriodicCounterSelectionList(uint32_t period, std::vector<uint16_t> counters);

    /// Execute the WAIT command from the comamnd file.
    void WaitCommand(uint32_t timeout);

    arm::pipe::DirectoryCaptureCommandHandler& GetDirectoryCaptureCommandHandler()
    {
        return m_DirectoryCaptureCommandHandler;
    }

    arm::pipe::TimelineDecoder& GetTimelineDecoder()
    {
        return m_TimelineDecoder;
    }

    arm::pipe::TimelineDirectoryCaptureCommandHandler& GetTimelineDirectoryCaptureCommandHandler()
    {
        return m_TimelineDirectoryCaptureCommandHandler;
    }

private:
    void ReceiveLoop();

    std::unique_ptr<arm::pipe::BasePipeServer>  m_BasePipeServer;

    std::atomic<uint32_t> m_PacketsReceivedCount;

    bool m_EchoPackets;
    std::thread m_ListeningThread;
    std::atomic<bool> m_CloseReceivingThread;

    arm::pipe::PacketVersionResolver m_PacketVersionResolver;
    arm::pipe::CommandHandlerRegistry m_HandlerRegistry;

    arm::pipe::TimelineDecoder m_TimelineDecoder;

    gatordmock::PeriodicCounterCaptureCommandHandler m_CounterCaptureCommandHandler;
    gatordmock::StreamMetadataCommandHandler m_StreamMetadataCommandHandler;
    gatordmock::StubCommandHandler m_StubCommandHandler;

    arm::pipe::DirectoryCaptureCommandHandler m_DirectoryCaptureCommandHandler;

    arm::pipe::TimelineCaptureCommandHandler m_TimelineCaptureCommandHandler;
    arm::pipe::TimelineDirectoryCaptureCommandHandler m_TimelineDirectoryCaptureCommandHandler;
};
}    // namespace gatordmock

}    // namespace armnn
