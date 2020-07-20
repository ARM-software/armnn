//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <server/include/basePipeServer/ConnectionHandler.hpp>

#include <SocketProfilingConnection.hpp>
#include <Processes.hpp>

#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>


BOOST_AUTO_TEST_SUITE(BasePipeServerTests)

using namespace armnn;
using namespace arm::pipe;

BOOST_AUTO_TEST_CASE(BasePipeServerTest)
{
    // Setup the mock service to bind to the UDS.
    std::string udsNamespace = "gatord_namespace";

    // Try to initialize a listening socket through the ConnectionHandler
    BOOST_CHECK_NO_THROW(ConnectionHandler connectionHandler(udsNamespace, true));

    // The socket should close once we leave the scope of BOOST_CHECK_NO_THROW
    // and socketProfilingConnection should fail to connect
    BOOST_CHECK_THROW(profiling::SocketProfilingConnection socketProfilingConnection,
                      arm::pipe::SocketConnectionException);

    // Try to initialize a listening socket through the ConnectionHandler again
    ConnectionHandler connectionHandler(udsNamespace, true);
    // socketProfilingConnection should connect now
    profiling::SocketProfilingConnection socketProfilingConnection;
    BOOST_TEST(socketProfilingConnection.IsOpen());

    auto basePipeServer = connectionHandler.GetNewBasePipeServer(false);
    // GetNewBasePipeServer will return null if it fails to create a socket
    BOOST_TEST(basePipeServer.get());

    profiling::BufferManager bufferManager;
    profiling::SendCounterPacket sendCounterPacket(bufferManager);

    // Check that we can receive a StreamMetaDataPacket
    sendCounterPacket.SendStreamMetaDataPacket();

    auto packetBuffer = bufferManager.GetReadableBuffer();
    const unsigned char* readBuffer = packetBuffer->GetReadableData();
    unsigned int readBufferSize = packetBuffer->GetSize();

    BOOST_TEST(readBuffer);
    BOOST_TEST(readBufferSize > 0u);

    socketProfilingConnection.WritePacket(readBuffer,readBufferSize);
    bufferManager.MarkRead(packetBuffer);

    BOOST_TEST(basePipeServer.get()->WaitForStreamMetaData());
    BOOST_TEST(basePipeServer.get()->GetStreamMetadataPid() == armnnUtils::Processes::GetCurrentId());
    BOOST_TEST(basePipeServer.get()->GetStreamMetadataMaxDataLen() == MAX_METADATA_PACKET_LENGTH);

    // Now try a simple PeriodicCounterSelectionPacket
    sendCounterPacket.SendPeriodicCounterSelectionPacket(50, {1,2,3,4,5});

    packetBuffer = bufferManager.GetReadableBuffer();
    readBuffer = packetBuffer->GetReadableData();
    readBufferSize = packetBuffer->GetSize();

    BOOST_TEST(readBuffer);
    BOOST_TEST(readBufferSize > 0u);

    socketProfilingConnection.WritePacket(readBuffer,readBufferSize);
    bufferManager.MarkRead(packetBuffer);

    auto packet1 = basePipeServer.get()->WaitForPacket(500);

    BOOST_TEST(!packet1.IsEmpty());
    BOOST_TEST(packet1.GetPacketFamily() == 0);
    BOOST_TEST(packet1.GetPacketId() == 4);
    BOOST_TEST(packet1.GetLength() == 14);

    // Try and send the packet back to the client
    basePipeServer.get()->SendPacket(packet1.GetPacketFamily(),
                                     packet1.GetPacketId(),
                                     packet1.GetData(),
                                     packet1.GetLength());

    auto packet2 = socketProfilingConnection.ReadPacket(500);

    BOOST_TEST(!packet2.IsEmpty());
    BOOST_TEST(packet2.GetPacketFamily() == 0);
    BOOST_TEST(packet2.GetPacketId() == 4);
    BOOST_TEST(packet2.GetLength() == 14);

    socketProfilingConnection.Close();
}

BOOST_AUTO_TEST_SUITE_END()