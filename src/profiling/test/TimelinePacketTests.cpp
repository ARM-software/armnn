//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ProfilingUtils.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/numeric/conversion/cast.hpp>

using namespace armnn::profiling;

BOOST_AUTO_TEST_SUITE(TimelinePacketTests)

BOOST_AUTO_TEST_CASE(TimelineLabelPacketTest1)
{
    const uint64_t profilingGuid = 123456u;
    const std::string label = "some label";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 nullptr,
                                                                 512u,
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineLabelPacketTest2)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    const std::string label = "some label";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 buffer.data(),
                                                                 0,
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineLabelPacketTest3)
{
    std::vector<unsigned char> buffer(10, 0);

    const uint64_t profilingGuid = 123456u;
    const std::string label = "some label";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 buffer.data(),
                                                                 boost::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineLabelPacketTest4)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    const std::string label = "s0m€ l@b€l";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 buffer.data(),
                                                                 boost::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Error);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineLabelPacketTest5)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    const std::string label = "some label";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 buffer.data(),
                                                                 boost::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Ok);
    BOOST_CHECK(numberOfBytesWritten == 32);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(buffer.data(), offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;
    BOOST_CHECK(packetFamily == 1);
    BOOST_CHECK(packetClass  == 0);
    BOOST_CHECK(packetType   == 1);
    BOOST_CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(buffer.data(), offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    BOOST_CHECK(sequenceNumbered ==  0);
    BOOST_CHECK(dataLength       == 24);

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readProfilingGuid == profilingGuid);

    // Check the SWTrace label
    offset += uint64_t_size;
    uint32_t swTraceLabelLength = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(swTraceLabelLength == 11); // Label length including the null-terminator

    offset += uint32_t_size;
    BOOST_CHECK(std::memcmp(buffer.data() + offset,        // Offset to the label in the buffer
                            label.data(),                  // The original label
                            swTraceLabelLength - 1) == 0); // The length of the label

    offset += swTraceLabelLength * uint32_t_size;
    BOOST_CHECK(buffer[offset] == '\0'); // The null-terminator at the end of the SWTrace label
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(TimelineEntityTests)

BOOST_AUTO_TEST_CASE(TimelineEntityPacketTest1)
{
    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEntityBinaryPacket(profilingGuid,
                                                                 nullptr,
                                                                 512u,
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineEntityPacketTest2)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEntityBinaryPacket(profilingGuid,
                                                                 buffer.data(),
                                                                 0,
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineEntityPacketTest3)
{
    std::vector<unsigned char> buffer(10, 0);

    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEntityBinaryPacket(profilingGuid,
                                                                 buffer.data(),
                                                                 boost::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineEntityPacketTest4)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEntityBinaryPacket(profilingGuid,
                                                                 buffer.data(),
                                                                 boost::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Ok);
    BOOST_CHECK(numberOfBytesWritten == 16);

    unsigned int uint32_t_size = sizeof(uint32_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(buffer.data(), offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;
    BOOST_CHECK(packetFamily == 1);
    BOOST_CHECK(packetClass  == 0);
    BOOST_CHECK(packetType   == 1);
    BOOST_CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(buffer.data(), offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    BOOST_CHECK(sequenceNumbered ==  0);
    BOOST_CHECK(dataLength       == 8);

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readProfilingGuid == profilingGuid);
}

BOOST_AUTO_TEST_SUITE_END()