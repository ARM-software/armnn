//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatordMockService.hpp"

#include <CommandHandlerRegistry.hpp>
#include <PacketVersionResolver.hpp>
#include <ProfilingUtils.hpp>
#include <NetworkSockets.hpp>

#include <cerrno>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <string>

using namespace armnnUtils;

namespace armnn
{

namespace gatordmock
{

bool GatordMockService::OpenListeningSocket(std::string udsNamespace)
{
    Sockets::Initialize();
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
    if (-1 == bind(m_ListeningSocket, reinterpret_cast<const sockaddr*>(&udsAddress), sizeof(sockaddr_un)))
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
    return true;
}

Sockets::Socket GatordMockService::BlockForOneClient()
{
    m_ClientConnection = Sockets::Accept(m_ListeningSocket, nullptr, nullptr, SOCK_CLOEXEC);
    if (-1 == m_ClientConnection)
    {
        std::cerr << ": Failure when waiting for a client connection: " << strerror(errno) << std::endl;
        return -1;
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
    uint8_t header[8];
    if (!ReadFromSocket(header, 8))
    {
        return false;
    }
    EchoPacket(PacketDirection::ReceivedHeader, header, 8);
    // The first word, stream_metadata_identifer, should always be 0.
    if (ToUint32(&header[0], TargetEndianness::BeWire) != 0)
    {
        std::cerr << ": Protocol error. The stream_metadata_identifer was not 0." << std::endl;
        return false;
    }

    uint8_t pipeMagic[4];
    if (!ReadFromSocket(pipeMagic, 4))
    {
        return false;
    }
    EchoPacket(PacketDirection::ReceivedData, pipeMagic, 4);

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
    std::vector<uint8_t> packetData(metaDataLength);
    if (metaDataLength !=
        boost::numeric_cast<uint32_t>(Sockets::Read(m_ClientConnection, packetData.data(), metaDataLength)))
    {
        std::cerr << ": Protocol read error. Data length mismatch." << std::endl;
        return false;
    }
    EchoPacket(PacketDirection::ReceivedData, packetData.data(), metaDataLength);
    m_StreamMetaDataVersion    = ToUint32(&packetData[0], m_Endianness);
    m_StreamMetaDataMaxDataLen = ToUint32(&packetData[4], m_Endianness);
    m_StreamMetaDataPid        = ToUint32(&packetData[8], m_Endianness);

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

void GatordMockService::SendRequestCounterDir()
{
    if (m_EchoPackets)
    {
        std::cout << "Sending connection acknowledgement." << std::endl;
    }
    // The connection ack packet is an empty data packet with packetId == 1.
    SendPacket(0, 3, nullptr, 0);
}

bool GatordMockService::LaunchReceivingThread()
{
    if (m_EchoPackets)
    {
        std::cout << "Launching receiving thread." << std::endl;
    }
    // At this point we want to make the socket non blocking.
    if (!Sockets::SetNonBlocking(m_ClientConnection))
    {
        Sockets::Close(m_ClientConnection);
        std::cerr << "Failed to set socket as non blocking: " << strerror(errno) << std::endl;
        return false;
    }
    m_ListeningThread = std::thread(&GatordMockService::ReceiveLoop, this, std::ref(*this));
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
        ;
    }
    // Start by calculating the length of the packet body in bytes. This will be at least 4.
    uint32_t dataLength = static_cast<uint32_t>(4 + (counters.size() * 2));

    std::unique_ptr<unsigned char[]> uniqueData = std::make_unique<unsigned char[]>(dataLength);
    unsigned char* data                         = reinterpret_cast<unsigned char*>(uniqueData.get());

    uint32_t offset = 0;
    profiling::WriteUint32(data, offset, period);
    offset += 4;
    for (std::vector<uint16_t>::iterator it = counters.begin(); it != counters.end(); ++it)
    {
        profiling::WriteUint16(data, offset, *it);
        offset += 2;
    }

    // Send the packet.
    SendPacket(0, 4, data, dataLength);
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

void GatordMockService::ReceiveLoop(GatordMockService& mockService)
{
    m_CloseReceivingThread.store(false);
    while (!m_CloseReceivingThread.load())
    {
        try
        {
            armnn::profiling::Packet packet = mockService.WaitForPacket(500);
        }
        catch (const armnn::TimeoutException&)
        {
            // In this case we ignore timeouts and and keep trying to receive.
        }
        catch (const armnn::InvalidArgumentException& e)
        {
            // We couldn't find a functor to handle the packet?
            std::cerr << "Packet received that could not be processed: " << e.what() << std::endl;
        }
        catch (const armnn::RuntimeException& e)
        {
            // A runtime exception occurred which means we must exit the loop.
            std::cerr << "Receive thread closing: " << e.what() << std::endl;
            m_CloseReceivingThread.store(true);
        }
    }
}

armnn::profiling::Packet GatordMockService::WaitForPacket(uint32_t timeoutMs)
{
    // Is there currently more than a headers worth of data waiting to be read?
    int bytes_available;
    Sockets::Ioctl(m_ClientConnection, FIONREAD, &bytes_available);
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
        int pollResult  = Sockets::Poll(pollingFd, 1, static_cast<int>(timeoutMs));

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
                    if (pollingFd[0].revents == POLLNVAL)
                    {
                        throw armnn::RuntimeException(std::string("Error while polling receiving socket: POLLNVAL"));
                    }
                    if (pollingFd[0].revents == POLLERR)
                    {
                        throw armnn::RuntimeException(std::string("Error while polling receiving socket: POLLERR: ") +
                                                      strerror(errno));
                    }
                    if (pollingFd[0].revents == POLLHUP)
                    {
                        throw armnn::RuntimeException(std::string("Connection closed by remote client: POLLHUP"));
                    }
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
    unsigned char* packetData                         = reinterpret_cast<unsigned char*>(uniquePacketData.get());

    if (!ReadFromSocket(packetData, header[1]))
    {
        return armnn::profiling::Packet();
    }

    EchoPacket(PacketDirection::ReceivedData, packetData, header[1]);

    // Construct received packet
    armnn::profiling::PacketVersionResolver packetVersionResolver;
    armnn::profiling::Packet packetRx = armnn::profiling::Packet(header[0], header[1], uniquePacketData);
    if (m_EchoPackets)
    {
        std::cout << "Processing packet ID= " << packetRx.GetPacketId() << " Length=" << packetRx.GetLength()
                  << std::endl;
    }

    profiling::Version version =
        packetVersionResolver.ResolvePacketVersion(packetRx.GetPacketFamily(), packetRx.GetPacketId());

    profiling::CommandHandlerFunctor* commandHandlerFunctor =
        m_HandlerRegistry.GetFunctor(packetRx.GetPacketFamily(), packetRx.GetPacketId(), version.GetEncodedValue());
    BOOST_ASSERT(commandHandlerFunctor);
    commandHandlerFunctor->operator()(packetRx);
    return packetRx;
}

bool GatordMockService::SendPacket(uint32_t packetFamily, uint32_t packetId, const uint8_t* data, uint32_t dataLength)
{
    // Construct a packet from the id and data given and send it to the client.
    // Encode the header.
    uint32_t header[2];
    header[0] = packetFamily << 26 | packetId << 16;
    header[1] = dataLength;
    // Add the header to the packet.
    std::vector<uint8_t> packet(8 + dataLength);
    InsertU32(header[0], packet.data(), m_Endianness);
    InsertU32(header[1], packet.data() + 4, m_Endianness);
    // And the rest of the data if there is any.
    if (dataLength > 0)
    {
        memcpy((packet.data() + 8), data, dataLength);
    }
    EchoPacket(PacketDirection::Sending, packet.data(), packet.size());
    if (-1 == Sockets::Write(m_ClientConnection, packet.data(), packet.size()))
    {
        std::cerr << ": Failure when writing to client socket: " << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

bool GatordMockService::ReadHeader(uint32_t headerAsWords[2])
{
    // The header will always be 2x32bit words.
    uint8_t header[8];
    if (!ReadFromSocket(header, 8))
    {
        return false;
    }
    EchoPacket(PacketDirection::ReceivedHeader, header, 8);
    headerAsWords[0] = ToUint32(&header[0], m_Endianness);
    headerAsWords[1] = ToUint32(&header[4], m_Endianness);
    return true;
}

bool GatordMockService::ReadFromSocket(uint8_t* packetData, uint32_t expectedLength)
{
    // This is a blocking read until either expectedLength has been received or an error is detected.
    long totalBytesRead = 0;
    while (boost::numeric_cast<uint32_t>(totalBytesRead) < expectedLength)
    {
        long bytesRead = Sockets::Read(m_ClientConnection, packetData, expectedLength);
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

void GatordMockService::EchoPacket(PacketDirection direction, uint8_t* packet, size_t lengthInBytes)
{
    // If enabled print the contents of the data packet to the console.
    if (m_EchoPackets)
    {
        if (direction == PacketDirection::Sending)
        {
            std::cout << "TX " << std::dec << lengthInBytes << " bytes : ";
        }
        else if (direction == PacketDirection::ReceivedHeader)
        {
            std::cout << "RX Header " << std::dec << lengthInBytes << " bytes : ";
        }
        else
        {
            std::cout << "RX Data " << std::dec << lengthInBytes << " bytes : ";
        }
        for (unsigned int i = 0; i < lengthInBytes; i++)
        {
            if ((i % 10) == 0)
            {
                std::cout << std::endl;
            }
            std::cout << "0x" << std::setfill('0') << std::setw(2) << std::hex << static_cast<unsigned int>(packet[i])
                      << " ";
        }
        std::cout << std::endl;
    }
}

uint32_t GatordMockService::ToUint32(uint8_t* data, TargetEndianness endianness)
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

void GatordMockService::InsertU32(uint32_t value, uint8_t* data, TargetEndianness endianness)
{
    // Take the bytes of a 32bit integer and copy them into char array starting at data considering
    // the endianness value.
    if (endianness == TargetEndianness::BeWire)
    {
        *data       = static_cast<uint8_t>((value >> 24) & 0xFF);
        *(data + 1) = static_cast<uint8_t>((value >> 16) & 0xFF);
        *(data + 2) = static_cast<uint8_t>((value >> 8) & 0xFF);
        *(data + 3) = static_cast<uint8_t>(value & 0xFF);
    }
    else
    {
        *(data + 3) = static_cast<uint8_t>((value >> 24) & 0xFF);
        *(data + 2) = static_cast<uint8_t>((value >> 16) & 0xFF);
        *(data + 1) = static_cast<uint8_t>((value >> 8) & 0xFF);
        *data       = static_cast<uint8_t>(value & 0xFF);
    }
}

}    // namespace gatordmock

}    // namespace armnn
