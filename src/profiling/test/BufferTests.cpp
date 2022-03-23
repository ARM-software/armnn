//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <client/src/BufferManager.hpp>
#include <client/src/PacketBuffer.hpp>
#include <client/src/ProfilingUtils.hpp>

#include <common/include/SwTrace.hpp>

#include <doctest/doctest.h>

using namespace arm::pipe;

TEST_SUITE("BufferTests")
{
TEST_CASE("PacketBufferTest0")
{
    IPacketBufferPtr packetBuffer = std::make_unique<PacketBuffer>(512);

    CHECK(packetBuffer->GetSize() == 0);

    // Write data to the buffer
    WriteUint32(packetBuffer, 0, 10);
    WriteUint32(packetBuffer, 4, 20);
    WriteUint32(packetBuffer, 8, 30);
    WriteUint32(packetBuffer, 12, 40);

    // Commit
    packetBuffer->Commit(16);

    // Size of buffer is equal to committed data
    CHECK(packetBuffer->GetSize() == 16);

    // Read data from the buffer
    auto readBuffer = packetBuffer->GetReadableData();
    uint32_t readData0 = ReadUint32(readBuffer, 0);
    uint32_t readData1 = ReadUint32(readBuffer, 4);
    uint32_t readData2 = ReadUint32(readBuffer, 8);
    uint32_t readData3 = ReadUint32(readBuffer, 12);

    // Check that data is correct
    CHECK(readData0 == 10);
    CHECK(readData1 == 20);
    CHECK(readData2 == 30);
    CHECK(readData3 == 40);

    // Mark read
    packetBuffer->MarkRead();

    // Size of buffer become 0 after marked read
    CHECK(packetBuffer->GetSize() == 0);
}

TEST_CASE("PacketBufferTest1")
{
    IPacketBufferPtr packetBuffer = std::make_unique<PacketBuffer>(512);

    CHECK(packetBuffer->GetSize() == 0);

    // Write data to the buffer using GetWritableData
    auto writeBuffer = packetBuffer->GetWritableData();
    WriteUint32(writeBuffer, 0, 10);
    WriteUint32(writeBuffer, 4, 20);
    WriteUint32(writeBuffer, 8, 30);
    WriteUint32(writeBuffer, 12, 40);

    packetBuffer->Commit(16);

    CHECK(packetBuffer->GetSize() == 16);

    // Read data from the buffer
    auto readBuffer = packetBuffer->GetReadableData();
    uint32_t readData0 = ReadUint32(readBuffer, 0);
    uint32_t readData1 = ReadUint32(readBuffer, 4);
    uint32_t readData2 = ReadUint32(readBuffer, 8);
    uint32_t readData3 = ReadUint32(readBuffer, 12);

    CHECK(readData0 == 10);
    CHECK(readData1 == 20);
    CHECK(readData2 == 30);
    CHECK(readData3 == 40);

    packetBuffer->MarkRead();

    CHECK(packetBuffer->GetSize() == 0);
}

TEST_CASE("PacketBufferReleaseTest")
{
    IPacketBufferPtr packetBuffer = std::make_unique<PacketBuffer>(512);

    CHECK(packetBuffer->GetSize() == 0);

    auto writeBuffer = packetBuffer->GetWritableData();

    WriteUint32(writeBuffer, 0, 10);
    WriteUint32(writeBuffer, 4, 20);
    WriteUint32(writeBuffer, 8, 30);
    WriteUint32(writeBuffer, 12, 40);

    packetBuffer->Release();

    // Size of buffer become 0 after release
    CHECK(packetBuffer->GetSize() == 0);
}

TEST_CASE("PacketBufferCommitErrorTest")
{
    IPacketBufferPtr packetBuffer = std::make_unique<PacketBuffer>(8);

    // Cannot commit data bigger than the max size of the buffer
    CHECK_THROWS_AS(packetBuffer->Commit(16);, arm::pipe::ProfilingException);
}

TEST_CASE("BufferReserveTest")
{
    BufferManager bufferManager(1, 512);
    unsigned int reservedSize = 0;
    auto packetBuffer = bufferManager.Reserve(512, reservedSize);

    // Successfully reserved the buffer with requested size
    CHECK(reservedSize == 512);
    CHECK(packetBuffer.get());
}

TEST_CASE("BufferReserveExceedingSpaceTest")
{
    BufferManager bufferManager(1, 512);
    unsigned int reservedSize = 0;

    // Cannot reserve buffer bigger than maximum buffer size
    auto reservedBuffer = bufferManager.Reserve(1024, reservedSize);
    CHECK(reservedSize == 0);
    CHECK(!reservedBuffer.get());
}

TEST_CASE("BufferExhaustionTest")
{
    BufferManager bufferManager(1, 512);
    unsigned int reservedSize = 0;
    auto packetBuffer = bufferManager.Reserve(512, reservedSize);

    // Successfully reserved the buffer with requested size
    CHECK(reservedSize == 512);
    CHECK(packetBuffer.get());

    // Cannot reserve buffer when buffer is not available
    // NOTE: because the buffer manager now has surge capacity of
    //       initial size * 3 we should be able to reserve three
    //       buffers before exhaustion
    packetBuffer = bufferManager.Reserve(512, reservedSize);

    // Successfully reserved the second buffer with requested size
    CHECK(reservedSize == 512);
    CHECK(packetBuffer.get());

    packetBuffer = bufferManager.Reserve(512, reservedSize);

    // Successfully reserved the third buffer with requested size
    CHECK(reservedSize == 512);
    CHECK(packetBuffer.get());

    auto reservedBuffer = bufferManager.Reserve(512, reservedSize);
    CHECK(reservedSize == 0);
    CHECK(!reservedBuffer.get());
}

TEST_CASE("BufferReserveMultipleTest")
{
    BufferManager bufferManager(3, 512);
    unsigned int reservedSize0 = 0;
    auto packetBuffer0 = bufferManager.Reserve(512, reservedSize0);

    // Successfully reserved the buffer with requested size
    CHECK(reservedSize0 == 512);
    CHECK(packetBuffer0.get());

    unsigned int reservedSize1 = 0;
    auto packetBuffer1 = bufferManager.Reserve(128, reservedSize1);

    // Successfully reserved the buffer with requested size
    CHECK(reservedSize1 == 128);
    CHECK(packetBuffer1.get());

    unsigned int reservedSize2 = 0;
    auto packetBuffer2 = bufferManager.Reserve(512, reservedSize2);

    // Successfully reserved the buffer with requested size
    CHECK(reservedSize2 == 512);
    CHECK(packetBuffer2.get());

    // NOTE: the buffer now has a surge capacity of initial size * 3
    //       so we can grab 9 of them prior to exhaustion now
    for (unsigned int i = 0; i < 6 ; ++i)
    {
        // grab another six buffers to exhaust the surge capacity
        unsigned int reservedSize = 0;
        auto packetBuffer = bufferManager.Reserve(512, reservedSize);

        // Successfully reserved the third buffer with requested size
        CHECK(reservedSize == 512);
        CHECK(packetBuffer.get());
    }

    // Cannot reserve when buffer is not available
    unsigned int reservedSize3 = 0;
    auto reservedBuffer = bufferManager.Reserve(512, reservedSize3);
    CHECK(reservedSize3 == 0);
    CHECK(!reservedBuffer.get());
}

TEST_CASE("BufferReleaseTest")
{
    BufferManager bufferManager(2, 512);
    unsigned int reservedSize0 = 0;
    auto packetBuffer0 = bufferManager.Reserve(512, reservedSize0);

    // Successfully reserved the buffer with requested size
    CHECK(reservedSize0 == 512);
    CHECK(packetBuffer0.get());

    unsigned int reservedSize1 = 0;
    auto packetBuffer1 = bufferManager.Reserve(128, reservedSize1);

    // Successfully reserved the buffer with requested size
    CHECK(reservedSize1 == 128);
    CHECK(packetBuffer1.get());

    // NOTE: now that we have a surge capacity of up to
    //       initial size * 3 we need to allocate four more
    //       buffers to exhaust the manager
    for (unsigned int i = 0; i < 4 ; ++i)
    {
        // grab another six buffers to exhaust the surge capacity
        unsigned int reservedSize = 0;
        auto packetBuffer = bufferManager.Reserve(512, reservedSize);

        // Successfully reserved the third buffer with requested size
        CHECK(reservedSize == 512);
        CHECK(packetBuffer.get());
    }

    // Cannot reserve when buffer is not available
    unsigned int reservedSize2 = 0;
    auto reservedBuffer = bufferManager.Reserve(512, reservedSize2);
    CHECK(reservedSize2 == 0);
    CHECK(!reservedBuffer.get());

    bufferManager.Release(packetBuffer0);

    // Buffer should become available after release
    auto packetBuffer2 = bufferManager.Reserve(128, reservedSize2);

    CHECK(reservedSize2 == 128);
    CHECK(packetBuffer2.get());
}

TEST_CASE("BufferCommitTest")
{
    BufferManager bufferManager(2, 512);
    unsigned int reservedSize0 = 0;
    auto packetBuffer0 = bufferManager.Reserve(512, reservedSize0);

    CHECK(reservedSize0 == 512);
    CHECK(packetBuffer0.get());

    unsigned int reservedSize1 = 0;
    auto packetBuffer1 = bufferManager.Reserve(128, reservedSize1);

    CHECK(reservedSize1 == 128);
    CHECK(packetBuffer1.get());

    // NOTE: now that we have a surge capacity of up to
    //       initial size * 3 we need to allocate four more
    //       buffers to exhaust the manager
    for (unsigned int i = 0; i < 4 ; ++i)
    {
        // grab another six buffers to exhaust the surge capacity
        unsigned int reservedSize = 0;
        auto packetBuffer = bufferManager.Reserve(512, reservedSize);

        // Successfully reserved the third buffer with requested size
        CHECK(reservedSize == 512);
        CHECK(packetBuffer.get());
    }

    unsigned int reservedSize2 = 0;
    auto reservedBuffer = bufferManager.Reserve(512, reservedSize2);
    CHECK(reservedSize2 == 0);
    CHECK(!reservedBuffer.get());

    bufferManager.Commit(packetBuffer0, 256);

    // Buffer should become readable after commit
    auto packetBuffer2 = bufferManager.GetReadableBuffer();
    CHECK(packetBuffer2.get());
    CHECK(packetBuffer2->GetSize() == 256);

    // Buffer not set back to available list after commit
    unsigned int reservedSize = 0;
    reservedBuffer = bufferManager.Reserve(512, reservedSize);
    CHECK(reservedSize == 0);
    CHECK(!reservedBuffer.get());
}

TEST_CASE("BufferMarkReadTest")
{
    BufferManager bufferManager(2, 512);
    unsigned int reservedSize0 = 0;
    auto packetBuffer0 = bufferManager.Reserve(512, reservedSize0);

    CHECK(reservedSize0 == 512);
    CHECK(packetBuffer0.get());

    unsigned int reservedSize1 = 0;
    auto packetBuffer1 = bufferManager.Reserve(128, reservedSize1);

    CHECK(reservedSize1 == 128);
    CHECK(packetBuffer1.get());

    // NOTE: now that we have a surge capacity of up to
    //       initial size * 3 we need to allocate four more
    //       buffers to exhaust the manager
    for (unsigned int i = 0; i < 4 ; ++i)
    {
        // grab another six buffers to exhaust the surge capacity
        unsigned int reservedSize = 0;
        auto packetBuffer = bufferManager.Reserve(512, reservedSize);

        // Successfully reserved the third buffer with requested size
        CHECK(reservedSize == 512);
        CHECK(packetBuffer.get());
    }

    // Cannot reserve when buffer is not available
    unsigned int reservedSize2 = 0;
    auto reservedBuffer = bufferManager.Reserve(512, reservedSize2);
    CHECK(reservedSize2 == 0);
    CHECK(!reservedBuffer.get());

    bufferManager.Commit(packetBuffer0, 256);

    // Buffer should become readable after commit
    auto packetBuffer2 = bufferManager.GetReadableBuffer();
    CHECK(packetBuffer2.get());
    CHECK(packetBuffer2->GetSize() == 256);

    // Buffer not set back to available list after commit
    reservedBuffer = bufferManager.Reserve(512, reservedSize2);
    CHECK(reservedSize2 == 0);
    CHECK(!reservedBuffer.get());

    bufferManager.MarkRead(packetBuffer2);

    //Buffer should set back to available list after marked read and can be reserved
    auto readBuffer = bufferManager.GetReadableBuffer();
    CHECK(!readBuffer);
    unsigned int reservedSize3 = 0;
    auto packetBuffer3 = bufferManager.Reserve(56, reservedSize3);

    CHECK(reservedSize3 == 56);
    CHECK(packetBuffer3.get());
}

TEST_CASE("ReadSwTraceMessageExceptionTest0")
{
    IPacketBufferPtr packetBuffer = std::make_unique<PacketBuffer>(512);

    CHECK(packetBuffer->GetSize() == 0);

    // Write zero data to the buffer
    WriteUint32(packetBuffer, 0, 0);
    WriteUint32(packetBuffer, 4, 0);
    WriteUint32(packetBuffer, 8, 0);
    WriteUint32(packetBuffer, 12, 0);

    // Commit
    packetBuffer->Commit(16);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int offset = uint32_t_size;
    CHECK_THROWS_AS(ReadSwTraceMessage(packetBuffer->GetReadableData(), offset, packetBuffer->GetSize()),
                    arm::pipe::ProfilingException);

}

TEST_CASE("ReadSwTraceMessageExceptionTest1")
{
    IPacketBufferPtr packetBuffer = std::make_unique<PacketBuffer>(512);

    CHECK(packetBuffer->GetSize() == 0);

    // Write data to the buffer
    WriteUint32(packetBuffer, 0, 10);
    WriteUint32(packetBuffer, 4, 20);
    WriteUint32(packetBuffer, 8, 30);
    WriteUint32(packetBuffer, 12, 40);

    // Commit
    packetBuffer->Commit(16);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int offset = uint32_t_size;
    CHECK_THROWS_AS(ReadSwTraceMessage(packetBuffer->GetReadableData(), offset, packetBuffer->GetSize()),
                    arm::pipe::ProfilingException);

}

}
