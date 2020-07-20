//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BufferManager.hpp"
#include "PacketBuffer.hpp"
#include "ProfilingUtils.hpp"

#include <common/include/SwTrace.hpp>

#include <armnn/Exceptions.hpp>

#include <boost/test/unit_test.hpp>

using namespace armnn::profiling;

BOOST_AUTO_TEST_SUITE(BufferTests)

BOOST_AUTO_TEST_CASE(PacketBufferTest0)
{
    IPacketBufferPtr packetBuffer = std::make_unique<PacketBuffer>(512);

    BOOST_TEST(packetBuffer->GetSize() == 0);

    // Write data to the buffer
    WriteUint32(packetBuffer, 0, 10);
    WriteUint32(packetBuffer, 4, 20);
    WriteUint32(packetBuffer, 8, 30);
    WriteUint32(packetBuffer, 12, 40);

    // Commit
    packetBuffer->Commit(16);

    // Size of buffer is equal to committed data
    BOOST_TEST(packetBuffer->GetSize() == 16);

    // Read data from the buffer
    auto readBuffer = packetBuffer->GetReadableData();
    uint32_t readData0 = ReadUint32(readBuffer, 0);
    uint32_t readData1 = ReadUint32(readBuffer, 4);
    uint32_t readData2 = ReadUint32(readBuffer, 8);
    uint32_t readData3 = ReadUint32(readBuffer, 12);

    // Check that data is correct
    BOOST_TEST(readData0 == 10);
    BOOST_TEST(readData1 == 20);
    BOOST_TEST(readData2 == 30);
    BOOST_TEST(readData3 == 40);

    // Mark read
    packetBuffer->MarkRead();

    // Size of buffer become 0 after marked read
    BOOST_TEST(packetBuffer->GetSize() == 0);
}

BOOST_AUTO_TEST_CASE(PacketBufferTest1)
{
    IPacketBufferPtr packetBuffer = std::make_unique<PacketBuffer>(512);

    BOOST_TEST(packetBuffer->GetSize() == 0);

    // Write data to the buffer using GetWritableData
    auto writeBuffer = packetBuffer->GetWritableData();
    WriteUint32(writeBuffer, 0, 10);
    WriteUint32(writeBuffer, 4, 20);
    WriteUint32(writeBuffer, 8, 30);
    WriteUint32(writeBuffer, 12, 40);

    packetBuffer->Commit(16);

    BOOST_TEST(packetBuffer->GetSize() == 16);

    // Read data from the buffer
    auto readBuffer = packetBuffer->GetReadableData();
    uint32_t readData0 = ReadUint32(readBuffer, 0);
    uint32_t readData1 = ReadUint32(readBuffer, 4);
    uint32_t readData2 = ReadUint32(readBuffer, 8);
    uint32_t readData3 = ReadUint32(readBuffer, 12);

    BOOST_TEST(readData0 == 10);
    BOOST_TEST(readData1 == 20);
    BOOST_TEST(readData2 == 30);
    BOOST_TEST(readData3 == 40);

    packetBuffer->MarkRead();

    BOOST_TEST(packetBuffer->GetSize() == 0);
}

BOOST_AUTO_TEST_CASE(PacketBufferReleaseTest) {
    IPacketBufferPtr packetBuffer = std::make_unique<PacketBuffer>(512);

    BOOST_TEST(packetBuffer->GetSize() == 0);

    auto writeBuffer = packetBuffer->GetWritableData();

    WriteUint32(writeBuffer, 0, 10);
    WriteUint32(writeBuffer, 4, 20);
    WriteUint32(writeBuffer, 8, 30);
    WriteUint32(writeBuffer, 12, 40);

    packetBuffer->Release();

    // Size of buffer become 0 after release
    BOOST_TEST(packetBuffer->GetSize() == 0);
}

BOOST_AUTO_TEST_CASE(PacketBufferCommitErrorTest)
{
    IPacketBufferPtr packetBuffer = std::make_unique<PacketBuffer>(8);

    // Cannot commit data bigger than the max size of the buffer
    BOOST_CHECK_THROW(packetBuffer->Commit(16);, armnn::RuntimeException);
}

BOOST_AUTO_TEST_CASE(BufferReserveTest)
{
    BufferManager bufferManager(1, 512);
    unsigned int reservedSize = 0;
    auto packetBuffer = bufferManager.Reserve(512, reservedSize);

    // Successfully reserved the buffer with requested size
    BOOST_TEST(reservedSize == 512);
    BOOST_TEST(packetBuffer.get());
}

BOOST_AUTO_TEST_CASE(BufferReserveExceedingSpaceTest)
{
    BufferManager bufferManager(1, 512);
    unsigned int reservedSize = 0;

    // Cannot reserve buffer bigger than maximum buffer size
    auto reservedBuffer = bufferManager.Reserve(1024, reservedSize);
    BOOST_TEST(reservedSize == 0);
    BOOST_TEST(!reservedBuffer.get());
}

BOOST_AUTO_TEST_CASE(BufferExhaustionTest)
{
    BufferManager bufferManager(1, 512);
    unsigned int reservedSize = 0;
    auto packetBuffer = bufferManager.Reserve(512, reservedSize);

    // Successfully reserved the buffer with requested size
    BOOST_TEST(reservedSize == 512);
    BOOST_TEST(packetBuffer.get());

    // Cannot reserve buffer when buffer is not available
    // NOTE: because the buffer manager now has surge capacity of
    //       initial size * 3 we should be able to reserve three
    //       buffers before exhaustion
    packetBuffer = bufferManager.Reserve(512, reservedSize);

    // Successfully reserved the second buffer with requested size
    BOOST_TEST(reservedSize == 512);
    BOOST_TEST(packetBuffer.get());

    packetBuffer = bufferManager.Reserve(512, reservedSize);

    // Successfully reserved the third buffer with requested size
    BOOST_TEST(reservedSize == 512);
    BOOST_TEST(packetBuffer.get());

    auto reservedBuffer = bufferManager.Reserve(512, reservedSize);
    BOOST_TEST(reservedSize == 0);
    BOOST_TEST(!reservedBuffer.get());
}

BOOST_AUTO_TEST_CASE(BufferReserveMultipleTest)
{
    BufferManager bufferManager(3, 512);
    unsigned int reservedSize0 = 0;
    auto packetBuffer0 = bufferManager.Reserve(512, reservedSize0);

    // Successfully reserved the buffer with requested size
    BOOST_TEST(reservedSize0 == 512);
    BOOST_TEST(packetBuffer0.get());

    unsigned int reservedSize1 = 0;
    auto packetBuffer1 = bufferManager.Reserve(128, reservedSize1);

    // Successfully reserved the buffer with requested size
    BOOST_TEST(reservedSize1 == 128);
    BOOST_TEST(packetBuffer1.get());

    unsigned int reservedSize2 = 0;
    auto packetBuffer2 = bufferManager.Reserve(512, reservedSize2);

    // Successfully reserved the buffer with requested size
    BOOST_TEST(reservedSize2 == 512);
    BOOST_TEST(packetBuffer2.get());

    // NOTE: the buffer now has a surge capacity of initial size * 3
    //       so we can grab 9 of them prior to exhaustion now
    for (unsigned int i = 0; i < 6 ; ++i)
    {
        // grab another six buffers to exhaust the surge capacity
        unsigned int reservedSize = 0;
        auto packetBuffer = bufferManager.Reserve(512, reservedSize);

        // Successfully reserved the third buffer with requested size
        BOOST_TEST(reservedSize == 512);
        BOOST_TEST(packetBuffer.get());
    }

    // Cannot reserve when buffer is not available
    unsigned int reservedSize3 = 0;
    auto reservedBuffer = bufferManager.Reserve(512, reservedSize3);
    BOOST_TEST(reservedSize3 == 0);
    BOOST_TEST(!reservedBuffer.get());
}

BOOST_AUTO_TEST_CASE(BufferReleaseTest)
{
    BufferManager bufferManager(2, 512);
    unsigned int reservedSize0 = 0;
    auto packetBuffer0 = bufferManager.Reserve(512, reservedSize0);

    // Successfully reserved the buffer with requested size
    BOOST_TEST(reservedSize0 == 512);
    BOOST_TEST(packetBuffer0.get());

    unsigned int reservedSize1 = 0;
    auto packetBuffer1 = bufferManager.Reserve(128, reservedSize1);

    // Successfully reserved the buffer with requested size
    BOOST_TEST(reservedSize1 == 128);
    BOOST_TEST(packetBuffer1.get());

    // NOTE: now that we have a surge capacity of up to
    //       initial size * 3 we need to allocate four more
    //       buffers to exhaust the manager
    for (unsigned int i = 0; i < 4 ; ++i)
    {
        // grab another six buffers to exhaust the surge capacity
        unsigned int reservedSize = 0;
        auto packetBuffer = bufferManager.Reserve(512, reservedSize);

        // Successfully reserved the third buffer with requested size
        BOOST_TEST(reservedSize == 512);
        BOOST_TEST(packetBuffer.get());
    }

    // Cannot reserve when buffer is not available
    unsigned int reservedSize2 = 0;
    auto reservedBuffer = bufferManager.Reserve(512, reservedSize2);
    BOOST_TEST(reservedSize2 == 0);
    BOOST_TEST(!reservedBuffer.get());

    bufferManager.Release(packetBuffer0);

    // Buffer should become available after release
    auto packetBuffer2 = bufferManager.Reserve(128, reservedSize2);

    BOOST_TEST(reservedSize2 == 128);
    BOOST_TEST(packetBuffer2.get());
}

BOOST_AUTO_TEST_CASE(BufferCommitTest)
{
    BufferManager bufferManager(2, 512);
    unsigned int reservedSize0 = 0;
    auto packetBuffer0 = bufferManager.Reserve(512, reservedSize0);

    BOOST_TEST(reservedSize0 == 512);
    BOOST_TEST(packetBuffer0.get());

    unsigned int reservedSize1 = 0;
    auto packetBuffer1 = bufferManager.Reserve(128, reservedSize1);

    BOOST_TEST(reservedSize1 == 128);
    BOOST_TEST(packetBuffer1.get());

    // NOTE: now that we have a surge capacity of up to
    //       initial size * 3 we need to allocate four more
    //       buffers to exhaust the manager
    for (unsigned int i = 0; i < 4 ; ++i)
    {
        // grab another six buffers to exhaust the surge capacity
        unsigned int reservedSize = 0;
        auto packetBuffer = bufferManager.Reserve(512, reservedSize);

        // Successfully reserved the third buffer with requested size
        BOOST_TEST(reservedSize == 512);
        BOOST_TEST(packetBuffer.get());
    }

    unsigned int reservedSize2 = 0;
    auto reservedBuffer = bufferManager.Reserve(512, reservedSize2);
    BOOST_TEST(reservedSize2 == 0);
    BOOST_TEST(!reservedBuffer.get());

    bufferManager.Commit(packetBuffer0, 256);

    // Buffer should become readable after commit
    auto packetBuffer2 = bufferManager.GetReadableBuffer();
    BOOST_TEST(packetBuffer2.get());
    BOOST_TEST(packetBuffer2->GetSize() == 256);

    // Buffer not set back to available list after commit
    unsigned int reservedSize = 0;
    reservedBuffer = bufferManager.Reserve(512, reservedSize);
    BOOST_TEST(reservedSize == 0);
    BOOST_TEST(!reservedBuffer.get());
}

BOOST_AUTO_TEST_CASE(BufferMarkReadTest)
{
    BufferManager bufferManager(2, 512);
    unsigned int reservedSize0 = 0;
    auto packetBuffer0 = bufferManager.Reserve(512, reservedSize0);

    BOOST_TEST(reservedSize0 == 512);
    BOOST_TEST(packetBuffer0.get());

    unsigned int reservedSize1 = 0;
    auto packetBuffer1 = bufferManager.Reserve(128, reservedSize1);

    BOOST_TEST(reservedSize1 == 128);
    BOOST_TEST(packetBuffer1.get());

    // NOTE: now that we have a surge capacity of up to
    //       initial size * 3 we need to allocate four more
    //       buffers to exhaust the manager
    for (unsigned int i = 0; i < 4 ; ++i)
    {
        // grab another six buffers to exhaust the surge capacity
        unsigned int reservedSize = 0;
        auto packetBuffer = bufferManager.Reserve(512, reservedSize);

        // Successfully reserved the third buffer with requested size
        BOOST_TEST(reservedSize == 512);
        BOOST_TEST(packetBuffer.get());
    }

    // Cannot reserve when buffer is not available
    unsigned int reservedSize2 = 0;
    auto reservedBuffer = bufferManager.Reserve(512, reservedSize2);
    BOOST_TEST(reservedSize2 == 0);
    BOOST_TEST(!reservedBuffer.get());

    bufferManager.Commit(packetBuffer0, 256);

    // Buffer should become readable after commit
    auto packetBuffer2 = bufferManager.GetReadableBuffer();
    BOOST_TEST(packetBuffer2.get());
    BOOST_TEST(packetBuffer2->GetSize() == 256);

    // Buffer not set back to available list after commit
    reservedBuffer = bufferManager.Reserve(512, reservedSize2);
    BOOST_TEST(reservedSize2 == 0);
    BOOST_TEST(!reservedBuffer.get());

    bufferManager.MarkRead(packetBuffer2);

    //Buffer should set back to available list after marked read and can be reserved
    auto readBuffer = bufferManager.GetReadableBuffer();
    BOOST_TEST(!readBuffer);
    unsigned int reservedSize3 = 0;
    auto packetBuffer3 = bufferManager.Reserve(56, reservedSize3);

    BOOST_TEST(reservedSize3 == 56);
    BOOST_TEST(packetBuffer3.get());
}

BOOST_AUTO_TEST_CASE(ReadSwTraceMessageExceptionTest0)
{
    IPacketBufferPtr packetBuffer = std::make_unique<PacketBuffer>(512);

    BOOST_TEST(packetBuffer->GetSize() == 0);

    // Write zero data to the buffer
    WriteUint32(packetBuffer, 0, 0);
    WriteUint32(packetBuffer, 4, 0);
    WriteUint32(packetBuffer, 8, 0);
    WriteUint32(packetBuffer, 12, 0);

    // Commit
    packetBuffer->Commit(16);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int offset = uint32_t_size;
    BOOST_CHECK_THROW(arm::pipe::ReadSwTraceMessage(packetBuffer->GetReadableData(), offset, packetBuffer->GetSize()),
                      arm::pipe::ProfilingException);

}

BOOST_AUTO_TEST_CASE(ReadSwTraceMessageExceptionTest1)
{
    IPacketBufferPtr packetBuffer = std::make_unique<PacketBuffer>(512);

    BOOST_TEST(packetBuffer->GetSize() == 0);

    // Write data to the buffer
    WriteUint32(packetBuffer, 0, 10);
    WriteUint32(packetBuffer, 4, 20);
    WriteUint32(packetBuffer, 8, 30);
    WriteUint32(packetBuffer, 12, 40);

    // Commit
    packetBuffer->Commit(16);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int offset = uint32_t_size;
    BOOST_CHECK_THROW(arm::pipe::ReadSwTraceMessage(packetBuffer->GetReadableData(), offset, packetBuffer->GetSize()),
                      arm::pipe::ProfilingException);

}

BOOST_AUTO_TEST_SUITE_END()
