//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatordMockService.hpp"

#include <CommandHandlerRegistry.hpp>

#include <cerrno>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace armnn
{

namespace gatordmock
{

bool GatordMockService::OpenListeningSocket(std::string udsNamespace)
{
    m_ListeningSocket = socket(PF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (-1 == m_ListeningSocket)
    {
        std::cerr << ": Socket construction failed: " << strerror(errno) << std::endl;
        return false;
    }

    sockaddr_un udsAddress;
    memset(&udsAddress, 0, sizeof(sockaddr_un));
    // We've set the first element of sun_path to be 0, skip over it and copy the namespace after it.
    memcpy(udsAddress.sun_path + 1, udsNamespace.c_str(), strlen(udsNamespace.c_str()));
    udsAddress.sun_family = AF_UNIX;

    // Bind the socket to the UDS namespace.
    if (-1 == bind(m_ListeningSocket, reinterpret_cast<const sockaddr *>(&udsAddress), sizeof(sockaddr_un)))
    {
        std::cerr << ": Binding on socket failed: " << strerror(errno) << std::endl;
        return false;
    }
    // Listen for 1 connection.
    if (-1 == listen(m_ListeningSocket, 1))
    {
        std::cerr << ": Listen call on socket failed: " << strerror(errno) << std::endl;
        return false;
    }
    if (m_EchoPackets)
    {
        std::cout << "Bound to UDS namespace: \\0" << udsNamespace << std::endl;
    }
    return true;
}

int GatordMockService::BlockForOneClient()
{
    if (m_EchoPackets)
    {
        std::cout << "Waiting for client connection." << std::endl;
    }
    m_ClientConnection = accept4(m_ListeningSocket, nullptr, nullptr, SOCK_CLOEXEC);
    if (-1 == m_ClientConnection)
    {
        std::cerr << ": Failure when waiting for a client connection: " << strerror(errno) << std::endl;
        return -1;
    }

    if (m_EchoPackets)
    {
        std::cout << "Client connection established." << std::endl;
    }
    return m_ClientConnection;
}

bool GatordMockService::WaitForStreamMetaData()
{
    if (m_EchoPackets)
    {
        std::cout << "Waiting for stream meta data..." << std::endl;
    }
    // The start of the stream metadata is 2x32bit words, 0 and packet length.
    u_char header[8];
    if (!ReadFromSocket(header, 8))
    {
        return false;
    }
    EchoPacket(PacketDirection::Received, header, 8);
    // The first word, stream_metadata_identifer, should always be 0.
    if (ToUint32(&header[0], TargetEndianness::BeWire) != 0)
    {
        std::cerr << ": Protocol error. The stream_metadata_identifer was not 0." << std::endl;
        return false;
    }

    u_char pipeMagic[4];
    if (!ReadFromSocket(pipeMagic, 4))
    {
        return false;
    }
    EchoPacket(PacketDirection::Received, pipeMagic, 4);

    // Before we interpret the length we need to read the pipe_magic word to determine endianness.
    if (ToUint32(&pipeMagic[0], TargetEndianness::BeWire) == PIPE_MAGIC)
    {
        m_Endianness = TargetEndianness::BeWire;
    }
    else if (ToUint32(&pipeMagic[0], TargetEndianness::LeWire) == PIPE_MAGIC)
    {
        m_Endianness = TargetEndianness::LeWire;
    }
    else
    {
        std::cerr << ": Protocol read error. Unable to read PIPE_MAGIC value." << std::endl;
        return false;
    }
    // Now we know the endianness we can get the length from the header.
    // Remember we already read the pipe magic 4 bytes.
    uint32_t metaDataLength = ToUint32(&header[4], m_Endianness) - 4;
    // Read the entire packet.
    u_char packetData[metaDataLength];
    if (metaDataLength != read(m_ClientConnection, &packetData, metaDataLength))
    {
        std::cerr << ": Protocol read error. Data length mismatch." << std::endl;
        return false;
    }
    EchoPacket(PacketDirection::Received, packetData, metaDataLength);
    m_StreamMetaDataVersion = ToUint32(&packetData[0], m_Endianness);
    m_StreamMetaDataMaxDataLen = ToUint32(&packetData[4], m_Endianness);
    m_StreamMetaDataPid = ToUint32(&packetData[8], m_Endianness);

    return true;
}

void GatordMockService::SendConnectionAck()
{
    if (m_EchoPackets)
    {
        std::cout << "Sending connection acknowledgement." << std::endl;
    }
    // The connection ack packet is an empty data packet with packetId == 1.
    SendPacket(0, 1, nullptr, 0);
}

bool GatordMockService::LaunchReceivingThread()
{
    if (m_EchoPackets)
    {
        std::cout << "Launching receiving thread." << std::endl;
    }
    // At this point we want to make the socket non blocking.
    const int currentFlags = fcntl(m_ClientConnection, F_GETFL);
    if (0 != fcntl(m_ClientConnection, F_SETFL, currentFlags | O_NONBLOCK))
    {
        close(m_ClientConnection);
        std::cerr << "Failed to set socket as non blocking: " << strerror(errno) << std::endl;
        return false;
    }
    m_ListeningThread = std::thread(&GatordMockService::ReceiveLoop, this, std::ref(*this));
    return true;
}

void GatordMockService::WaitForReceivingThread()
{
    m_CloseReceivingThread.store(true);
    // Check that the receiving thread is running
    if (m_ListeningThread.joinable())
    {
        // Wait for the receiving thread to complete operations
        m_ListeningThread.join();
    }
}


void GatordMockService::SendPeriodicCounterSelectionList(uint period, std::vector<uint16_t> counters)
{
    //get the datalength in bytes
    uint32_t datalength = static_cast<uint32_t>(4 + counters.size() * 2);

    u_char data[datalength];

    *data = static_cast<u_char>(period >> 24);
    *(data + 1) = static_cast<u_char>(period >> 16 & 0xFF);
    *(data + 2) = static_cast<u_char>(period >> 8 & 0xFF);
    *(data + 3) = static_cast<u_char>(period & 0xFF);

    for (unsigned long i = 0; i < counters.size(); ++i)
    {
        *(data + 4 + i * 2) = static_cast<u_char>(counters[i] >> 8);
        *(data + 5 + i * 2) = static_cast<u_char>(counters[i] & 0xFF);
    }

    // create packet send packet
    SendPacket(0, 4, data, datalength);
}

void GatordMockService::WaitCommand(uint timeout)
{
    std::this_thread::sleep_for(std::chrono::microseconds(timeout));

    if (m_EchoPackets)
    {
        std::cout << std::dec << "Wait command with timeout of " << timeout << " microseconds completed. " << std::endl;
    }
}

void GatordMockService::ReceiveLoop(GatordMockService& mockService)
{
    m_CloseReceivingThread.store(false);
    while (!m_CloseReceivingThread.load())
    {
        try
        {
            armnn::profiling::Packet packet = mockService.WaitForPacket(500);
        }
        catch(armnn::TimeoutException)
        {
            // In this case we ignore timeouts and and keep trying to receive.
        }
    }
}

armnn::profiling::Packet GatordMockService::WaitForPacket(uint32_t timeoutMs)
{
    // Is there currently more than a headers worth of data waiting to be read?
    int bytes_available;
    ioctl(m_ClientConnection, FIONREAD, &bytes_available);
    if (bytes_available > 8)
    {
        // Yes there is. Read it:
        return ReceivePacket();
    }
    else
    {
        // No there's not. Poll for more data.
        struct pollfd pollingFd[1]{};
        pollingFd[0].fd = m_ClientConnection;
        int pollResult = poll(pollingFd, 1, static_cast<int>(timeoutMs));

        switch (pollResult)
        {
            // Error
            case -1:
                throw armnn::RuntimeException(std::string("File descriptor reported an error during polling: ") +
                                              strerror(errno));

            // Timeout
            case 0:
                throw armnn::TimeoutException("Timeout while waiting to receive packet.");

            // Normal poll return. It could still contain an error signal
            default:

                // Check if the socket reported an error
                if (pollingFd[0].revents & (POLLNVAL | POLLERR | POLLHUP))
                {
                    std::cout << "Error while polling receiving socket." << std::endl;
                    throw armnn::RuntimeException(std::string("File descriptor reported an error during polling: ") +
                                                  strerror(errno));
                }

                // Check if there is data to read
                if (!(pollingFd[0].revents & (POLLIN)))
                {
                    // This is a corner case. The socket as been woken up but not with any data.
                    // We'll throw a timeout exception to loop around again.
                    throw armnn::TimeoutException("File descriptor was polled but no data was available to receive.");
                }
                return ReceivePacket();
        }
    }
}


armnn::profiling::Packet GatordMockService::ReceivePacket()
{
    uint32_t header[2];
    if (!ReadHeader(header))
    {
        return armnn::profiling::Packet();
    }
    // Read data_length bytes from the socket.
    std::unique_ptr<unsigned char[]> uniquePacketData = std::make_unique<unsigned char[]>(header[1]);
    unsigned char *packetData = reinterpret_cast<unsigned char *>(uniquePacketData.get());

    if (!ReadFromSocket(packetData, header[1]))
    {
        return armnn::profiling::Packet();
    }

    // Construct received packet
    armnn::profiling::Packet packetRx = armnn::profiling::Packet(header[0], header[1], uniquePacketData);

    // Pass packet into the handler registry
    if (packetRx.GetHeader()!= 0)
    {
        m_PacketsReceivedCount.operator++(std::memory_order::memory_order_release);
        m_HandlerRegistry.GetFunctor(header[0],1)->operator()(packetRx);
    }

    EchoPacket(PacketDirection::Received, packetData, sizeof(packetData));
    return packetRx;
}

bool GatordMockService::SendPacket(uint32_t packetFamily, uint32_t packetId, const u_char* data, uint32_t dataLength)
{
    // Construct a packet from the id and data given and send it to the client.
    // Encode the header.
    uint32_t header[2];
    header[0] = packetFamily << 26 | packetId << 16;
    header[1] = dataLength;
    // Add the header to the packet.
    u_char packet[8 + dataLength ];
    InsertU32(header[0], packet, m_Endianness);
    InsertU32(header[1], packet + 4, m_Endianness);
    // And the rest of the data if there is any.
    if (dataLength > 0)
    {
        memcpy((packet + 8), data, dataLength);
    }
    EchoPacket(PacketDirection::Sending, packet, sizeof(packet));
    if (-1 == write(m_ClientConnection, packet, sizeof(packet)))
    {
        std::cerr << ": Failure when writing to client socket: " << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

bool GatordMockService::ReadHeader(uint32_t headerAsWords[2])
{
    // The herader will always be 2x32bit words.
    u_char header[8];
    if (!ReadFromSocket(header, 8))
    {
        return false;
    }
    headerAsWords[0] = ToUint32(&header[0], m_Endianness);
    headerAsWords[1] = ToUint32(&header[4], m_Endianness);
    return true;
}

bool GatordMockService::ReadFromSocket(u_char* packetData, uint32_t expectedLength)
{
    // This is a blocking read until either expectedLength has been received or an error is detected.
    ssize_t totalBytesRead = 0;
    while (totalBytesRead < expectedLength)
    {
        ssize_t bytesRead = recv(m_ClientConnection, packetData, expectedLength, 0);
        if (bytesRead < 0)
        {
            std::cerr << ": Failure when reading from client socket: " << strerror(errno) << std::endl;
            return false;
        }
        if (bytesRead == 0)
        {
            std::cerr << ": EOF while reading from client socket." << std::endl;
            return false;
        }
        totalBytesRead += bytesRead;
    }
    return true;
};

void GatordMockService::EchoPacket(PacketDirection direction, u_char* packet, size_t lengthInBytes)
{
    // If enabled print the contents of the data packet to the console.
    if (m_EchoPackets)
    {
        if (direction == PacketDirection::Sending)
        {
            std::cout << "Sending " << std::dec << lengthInBytes << " bytes : ";
        } else
        {
            std::cout << "Received " << std::dec << lengthInBytes << " bytes : ";
        }
        for (unsigned int i = 0; i < lengthInBytes; i++)
        {
            if ((i % 10) == 0)
            {
                std::cout << std::endl;
            }
            std::cout << std::hex << "0x" << static_cast<unsigned int>(packet[i]) << " ";
        }
        std::cout << std::endl;
    }
}

uint32_t GatordMockService::ToUint32(u_char* data, TargetEndianness endianness)
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

void GatordMockService::InsertU32(uint32_t value, u_char* data, TargetEndianness endianness)
{
    // Take the bytes of a 32bit integer and copy them into char array starting at data considering
    // the endianness value.
    if (endianness == TargetEndianness::BeWire)
    {
        *data = static_cast<u_char>((value >> 24) & 0xFF);
        *(data + 1) = static_cast<u_char>((value >> 16) & 0xFF);
        *(data + 2) = static_cast<u_char>((value >> 8) & 0xFF);
        *(data + 3) = static_cast<u_char>(value & 0xFF);
    }
    else
    {
        *(data + 3) = static_cast<u_char>((value >> 24) & 0xFF);
        *(data + 2) = static_cast<u_char>((value >> 16) & 0xFF);
        *(data + 1) = static_cast<u_char>((value >> 8) & 0xFF);
        *data = static_cast<u_char>(value & 0xFF);
    }
}

} // namespace gatordmock

} // namespace armnn
