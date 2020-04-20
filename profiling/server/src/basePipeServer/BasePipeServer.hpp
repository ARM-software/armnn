//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

//#include <NetworkSockets>
#include <string>
#include <atomic>
#include "../../../../src/armnnUtils/NetworkSockets.hpp"
#include "../../../../src/profiling/Packet.hpp"
#include "common/include/SocketConnectionException.hpp"

namespace armnnProfiling
{

enum class TargetEndianness
{
    BeWire,
    LeWire
};

enum class PacketDirection
{
    Sending,
    ReceivedHeader,
    ReceivedData
};
class ConnectionHandler;

class BasePipeServer
{

public:

    BasePipeServer(armnnUtils::Sockets::Socket clientConnection, bool echoPackets)
            : m_ClientConnection(clientConnection)
            , m_EchoPackets(echoPackets)
            {}

    ~BasePipeServer()
    {
        // We have set SOCK_CLOEXEC on this socket but we'll close it to be good citizens.
        armnnUtils::Sockets::Close(m_ClientConnection);
    }

    BasePipeServer(const BasePipeServer&) = delete;
    BasePipeServer& operator=(const BasePipeServer&) = delete;

    BasePipeServer(BasePipeServer&&) = delete;
    BasePipeServer& operator=(BasePipeServer&&) = delete;

    /// Close the client connection
    /// @return 0 if successful
    int Close()
    {
        return armnnUtils::Sockets::Close(m_ClientConnection);
    }

    /// Send a packet to the client
    /// @return true if a valid packet has been sent.
    bool SendPacket(uint32_t packetFamily, uint32_t packetId, const uint8_t* data, uint32_t dataLength);

    /// Set the client socket to nonblocking
    /// @return true if successful.
    bool SetNonBlocking()
    {
        return armnnUtils::Sockets::SetNonBlocking(m_ClientConnection);
    }

    /// Block on the client connection until a complete packet has been received.
    /// @return true if a valid packet has been received.
    armnn::profiling::Packet WaitForPacket(uint32_t timeoutMs);

    /// Once the connection is open wait to receive the stream meta data packet from the client. Reading this
    /// packet differs from others as we need to determine endianness.
    /// @return true only if a valid stream meta data packet has been received.
    bool WaitForStreamMetaData();

    uint32_t GetStreamMetadataVersion()
    {
        return m_StreamMetaDataVersion;
    }

    uint32_t GetStreamMetadataMaxDataLen()
    {
        return m_StreamMetaDataMaxDataLen;
    }

    uint32_t GetStreamMetadataPid()
    {
        return m_StreamMetaDataPid;
    }

private:

    void EchoPacket(PacketDirection direction, uint8_t* packet, size_t lengthInBytes);
    bool ReadFromSocket(uint8_t* packetData, uint32_t expectedLength);
    bool ReadHeader(uint32_t headerAsWords[2]);

    armnn::profiling::Packet ReceivePacket();

    uint32_t ToUint32(uint8_t* data, TargetEndianness endianness);
    void InsertU32(uint32_t value, uint8_t* data, TargetEndianness endianness);

    armnnUtils::Sockets::Socket m_ClientConnection;
    bool m_EchoPackets;
    TargetEndianness m_Endianness;
    static const uint32_t PIPE_MAGIC = 0x45495434;

    uint32_t m_StreamMetaDataVersion;
    uint32_t m_StreamMetaDataMaxDataLen;
    uint32_t m_StreamMetaDataPid;
};

} // namespace armnnProfiling