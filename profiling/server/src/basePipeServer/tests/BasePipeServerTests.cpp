//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <server/include/basePipeServer/ConnectionHandler.hpp>

#include <client/src/BufferManager.hpp>
#include <client/src/SendCounterPacket.hpp>
#include <client/src/SocketProfilingConnection.hpp>

#include <common/include/Processes.hpp>

#include <doctest/doctest.h>

TEST_SUITE("BasePipeServerTests")
{
using namespace arm::pipe;

TEST_CASE("BasePipeServerTest")
{
    // Setup the mock service to bind to the UDS.
    std::string udsNamespace = "gatord_namespace";

    // Try to initialize a listening socket through the ConnectionHandler
    CHECK_NOTHROW(ConnectionHandler connectionHandler(udsNamespace, true));

    // The socket should close once we leave the scope of CHECK_NOTHROW
    // and socketProfilingConnection should fail to connect
    CHECK_THROWS_AS(arm::pipe::SocketProfilingConnection socketProfilingConnection,
                      arm::pipe::SocketConnectionException);

    // Try to initialize a listening socket through the ConnectionHandler again
    ConnectionHandler connectionHandler(udsNamespace, true);
    // socketProfilingConnection should connect now
    arm::pipe::SocketProfilingConnection socketProfilingConnection;
    CHECK(socketProfilingConnection.IsOpen());

    auto basePipeServer = connectionHandler.GetNewBasePipeServer(false);
    // GetNewBasePipeServer will return null if it fails to create a socket
    CHECK(basePipeServer.get());

    arm::pipe::BufferManager bufferManager;
    arm::pipe::SendCounterPacket sendCounterPacket(bufferManager, "ArmNN", "Armnn 25.0", "");

    // Check that we can receive a StreamMetaDataPacket
    sendCounterPacket.SendStreamMetaDataPacket();

    auto packetBuffer = bufferManager.GetReadableBuffer();
    const unsigned char* readBuffer = packetBuffer->GetReadableData();
    unsigned int readBufferSize = packetBuffer->GetSize();

    CHECK(readBuffer);
    CHECK(readBufferSize > 0u);

    socketProfilingConnection.WritePacket(readBuffer,readBufferSize);
    bufferManager.MarkRead(packetBuffer);

    CHECK(basePipeServer.get()->WaitForStreamMetaData());
    CHECK(basePipeServer.get()->GetStreamMetadataPid() == arm::pipe::GetCurrentProcessId());
    CHECK(basePipeServer.get()->GetStreamMetadataMaxDataLen() == MAX_METADATA_PACKET_LENGTH);

    // Now try a simple PeriodicCounterSelectionPacket
    sendCounterPacket.SendPeriodicCounterSelectionPacket(50, {1,2,3,4,5});

    packetBuffer = bufferManager.GetReadableBuffer();
    readBuffer = packetBuffer->GetReadableData();
    readBufferSize = packetBuffer->GetSize();

    CHECK(readBuffer);
    CHECK(readBufferSize > 0u);

    socketProfilingConnection.WritePacket(readBuffer,readBufferSize);
    bufferManager.MarkRead(packetBuffer);

    auto packet1 = basePipeServer.get()->WaitForPacket(500);

    CHECK(!packet1.IsEmpty());
    CHECK(packet1.GetPacketFamily() == 0);
    CHECK(packet1.GetPacketId() == 4);
    CHECK(packet1.GetLength() == 14);

    // Try and send the packet back to the client
    basePipeServer.get()->SendPacket(packet1.GetPacketFamily(),
                                     packet1.GetPacketId(),
                                     packet1.GetData(),
                                     packet1.GetLength());

    auto packet2 = socketProfilingConnection.ReadPacket(500);

    CHECK(!packet2.IsEmpty());
    CHECK(packet2.GetPacketFamily() == 0);
    CHECK(packet2.GetPacketId() == 4);
    CHECK(packet2.GetLength() == 14);

    socketProfilingConnection.Close();
}

}
