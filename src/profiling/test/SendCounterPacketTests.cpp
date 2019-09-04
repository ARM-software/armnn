//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../SendCounterPacket.hpp"
#include "../ProfilingUtils.hpp"
#include "../EncodeVersion.hpp"

#include <armnn/Exceptions.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <chrono>
#include <iostream>

BOOST_AUTO_TEST_SUITE(SendCounterPacketTests)

using namespace armnn::profiling;

class MockBuffer : public IBufferWrapper
{
public:
    MockBuffer(unsigned int size)
    : m_BufferSize(size),
      m_Buffer(std::make_unique<unsigned char[]>(size)) {}

    unsigned char* Reserve(unsigned int requestedSize, unsigned int& reservedSize) override
    {
        if (requestedSize > m_BufferSize)
        {
            reservedSize = m_BufferSize;
        }
        else
        {
            reservedSize = requestedSize;
        }

        return m_Buffer.get();
    }

    void Commit(unsigned int size) override {}

    const unsigned char* GetReadBuffer(unsigned int& size) override
    {
        size = static_cast<unsigned int>(strlen(reinterpret_cast<const char*>(m_Buffer.get())) + 1);
        return m_Buffer.get();
    }

    void Release( unsigned int size) override {}

private:
    unsigned int m_BufferSize;
    std::unique_ptr<unsigned char[]> m_Buffer;
};

class MockSendCounterPacket : public ISendCounterPacket
{
public:
    MockSendCounterPacket(IBufferWrapper& sendBuffer) : m_Buffer(sendBuffer) {}

    void SendStreamMetaDataPacket() override
    {
        std::string message("SendStreamMetaDataPacket");
        unsigned int reserved = 0;
        unsigned char* buffer = m_Buffer.Reserve(1024, reserved);
        memcpy(buffer, message.c_str(), static_cast<unsigned int>(message.size()) + 1);
    }

    void SendCounterDirectoryPacket(const Category& category, const std::vector<Counter>& counters) override
    {
        std::string message("SendCounterDirectoryPacket");
        unsigned int reserved = 0;
        unsigned char* buffer = m_Buffer.Reserve(1024, reserved);
        memcpy(buffer, message.c_str(), static_cast<unsigned int>(message.size()) + 1);
    }

    void SendPeriodicCounterCapturePacket(uint64_t timestamp,
                                          const std::vector<std::pair<uint16_t, uint32_t>>& values) override
    {
        std::string message("SendPeriodicCounterCapturePacket");
        unsigned int reserved = 0;
        unsigned char* buffer = m_Buffer.Reserve(1024, reserved);
        memcpy(buffer, message.c_str(), static_cast<unsigned int>(message.size()) + 1);
    }

    void SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                            const std::vector<uint16_t>& selectedCounterIds) override
    {
        std::string message("SendPeriodicCounterSelectionPacket");
        unsigned int reserved = 0;
        unsigned char* buffer = m_Buffer.Reserve(1024, reserved);
        memcpy(buffer, message.c_str(), static_cast<unsigned int>(message.size()) + 1);
        m_Buffer.Commit(reserved);
    }

    void SetReadyToRead() override
    {}

private:
    IBufferWrapper& m_Buffer;
};

BOOST_AUTO_TEST_CASE(MockSendCounterPacketTest)
{
    unsigned int size = 0;

    MockBuffer mockBuffer(512);
    MockSendCounterPacket sendCounterPacket(mockBuffer);

    sendCounterPacket.SendStreamMetaDataPacket();
    const char* buffer = reinterpret_cast<const char*>(mockBuffer.GetReadBuffer(size));

    BOOST_TEST(strcmp(buffer, "SendStreamMetaDataPacket") == 0);

    Category category;
    std::vector<Counter> counters;
    sendCounterPacket.SendCounterDirectoryPacket(category, counters);

    BOOST_TEST(strcmp(buffer, "SendCounterDirectoryPacket") == 0);

    uint64_t timestamp = 0;
    std::vector<std::pair<uint16_t, uint32_t>> indexValuePairs;

    sendCounterPacket.SendPeriodicCounterCapturePacket(timestamp, indexValuePairs);

    BOOST_TEST(strcmp(buffer, "SendPeriodicCounterCapturePacket") == 0);

    uint32_t capturePeriod = 0;
    std::vector<uint16_t> selectedCounterIds;
    sendCounterPacket.SendPeriodicCounterSelectionPacket(capturePeriod, selectedCounterIds);

    BOOST_TEST(strcmp(buffer, "SendPeriodicCounterSelectionPacket") == 0);

}

BOOST_AUTO_TEST_CASE(SendPeriodicCounterSelectionPacketTest)
{
    // Error no space left in buffer
    MockBuffer mockBuffer1(10);
    SendCounterPacket sendPacket1(mockBuffer1);

    uint32_t capturePeriod = 1000;
    std::vector<uint16_t> selectedCounterIds;
    BOOST_CHECK_THROW(sendPacket1.SendPeriodicCounterSelectionPacket(capturePeriod, selectedCounterIds),
                      armnn::RuntimeException);

    // Packet without any counters
    MockBuffer mockBuffer2(512);
    SendCounterPacket sendPacket2(mockBuffer2);

    sendPacket2.SendPeriodicCounterSelectionPacket(capturePeriod, selectedCounterIds);
    unsigned int sizeRead = 0;
    const unsigned char* readBuffer2 = mockBuffer2.GetReadBuffer(sizeRead);

    uint32_t headerWord0 = ReadUint32(readBuffer2, 0);
    uint32_t headerWord1 = ReadUint32(readBuffer2, 4);
    uint32_t period = ReadUint32(readBuffer2, 8);

    BOOST_TEST(((headerWord0 >> 26) & 0x3F) == 0);  // packet family
    BOOST_TEST(((headerWord0 >> 16) & 0x3FF) == 4); // packet id
    BOOST_TEST(headerWord1 == 4);                   // data lenght
    BOOST_TEST(period == 1000);                     // capture period

    // Full packet message
    MockBuffer mockBuffer3(512);
    SendCounterPacket sendPacket3(mockBuffer3);

    selectedCounterIds.reserve(5);
    selectedCounterIds.emplace_back(100);
    selectedCounterIds.emplace_back(200);
    selectedCounterIds.emplace_back(300);
    selectedCounterIds.emplace_back(400);
    selectedCounterIds.emplace_back(500);
    sendPacket3.SendPeriodicCounterSelectionPacket(capturePeriod, selectedCounterIds);
    sizeRead = 0;
    const unsigned char* readBuffer3 = mockBuffer3.GetReadBuffer(sizeRead);

    headerWord0 = ReadUint32(readBuffer3, 0);
    headerWord1 = ReadUint32(readBuffer3, 4);
    period = ReadUint32(readBuffer3, 8);

    BOOST_TEST(((headerWord0 >> 26) & 0x3F) == 0);  // packet family
    BOOST_TEST(((headerWord0 >> 16) & 0x3FF) == 4); // packet id
    BOOST_TEST(headerWord1 == 14);                  // data lenght
    BOOST_TEST(period == 1000);                     // capture period

    uint16_t counterId = 0;
    uint32_t offset = 12;

    // Counter Ids
    for(const uint16_t& id : selectedCounterIds)
    {
        counterId = ReadUint16(readBuffer3, offset);
        BOOST_TEST(counterId == id);
        offset += 2;
    }
}

BOOST_AUTO_TEST_CASE(SendPeriodicCounterCapturePacketTest)
{
    // Error no space left in buffer
    MockBuffer mockBuffer1(10);
    SendCounterPacket sendPacket1(mockBuffer1);

    auto captureTimestamp = std::chrono::steady_clock::now();
    uint64_t time =  static_cast<uint64_t >(captureTimestamp.time_since_epoch().count());
    std::vector<std::pair<uint16_t, uint32_t>> indexValuePairs;

    BOOST_CHECK_THROW(sendPacket1.SendPeriodicCounterCapturePacket(time, indexValuePairs),
                      BufferExhaustion);

    // Packet without any counters
    MockBuffer mockBuffer2(512);
    SendCounterPacket sendPacket2(mockBuffer2);

    sendPacket2.SendPeriodicCounterCapturePacket(time, indexValuePairs);
    unsigned int sizeRead = 0;
    const unsigned char* readBuffer2 = mockBuffer2.GetReadBuffer(sizeRead);

    uint32_t headerWord0 = ReadUint32(readBuffer2, 0);
    uint32_t headerWord1 = ReadUint32(readBuffer2, 4);
    uint64_t readTimestamp = ReadUint64(readBuffer2, 8);

    BOOST_TEST(((headerWord0 >> 26) & 0x3F) == 1);   // packet family
    BOOST_TEST(((headerWord0 >> 19) & 0x3F) == 0);   // packet class
    BOOST_TEST(((headerWord0 >> 16) & 0x3) == 0);    // packet type
    BOOST_TEST(headerWord1 == 8);                    // data length
    BOOST_TEST(time == readTimestamp);               // capture period

    // Full packet message
    MockBuffer mockBuffer3(512);
    SendCounterPacket sendPacket3(mockBuffer3);

    indexValuePairs.reserve(5);
    indexValuePairs.emplace_back(std::make_pair<uint16_t, uint32_t >(0, 100));
    indexValuePairs.emplace_back(std::make_pair<uint16_t, uint32_t >(1, 200));
    indexValuePairs.emplace_back(std::make_pair<uint16_t, uint32_t >(2, 300));
    indexValuePairs.emplace_back(std::make_pair<uint16_t, uint32_t >(3, 400));
    indexValuePairs.emplace_back(std::make_pair<uint16_t, uint32_t >(4, 500));
    sendPacket3.SendPeriodicCounterCapturePacket(time, indexValuePairs);
    sizeRead = 0;
    const unsigned char* readBuffer3 = mockBuffer3.GetReadBuffer(sizeRead);

    headerWord0 = ReadUint32(readBuffer3, 0);
    headerWord1 = ReadUint32(readBuffer3, 4);
    uint64_t readTimestamp2 = ReadUint64(readBuffer3, 8);

    BOOST_TEST(((headerWord0 >> 26) & 0x3F) == 1);   // packet family
    BOOST_TEST(((headerWord0 >> 19) & 0x3F) == 0);   // packet class
    BOOST_TEST(((headerWord0 >> 16) & 0x3) == 0);    // packet type
    BOOST_TEST(headerWord1 == 38);                   // data length
    BOOST_TEST(time == readTimestamp2);              // capture period

    uint16_t counterIndex = 0;
    uint32_t counterValue = 100;
    uint32_t offset = 16;

    // Counter Ids
    for (auto it = indexValuePairs.begin(), end = indexValuePairs.end(); it != end; ++it)
    {
        // Check Counter Index
        uint16_t readIndex = ReadUint16(readBuffer3, offset);
        BOOST_TEST(counterIndex == readIndex);
        counterIndex++;
        offset += 2;

        // Check Counter Value
        uint32_t readValue = ReadUint32(readBuffer3, offset);
        BOOST_TEST(counterValue == readValue);
        counterValue += 100;
        offset += 4;
    }

}

BOOST_AUTO_TEST_CASE(SendStreamMetaDataPacketTest)
{
    using boost::numeric_cast;

    uint32_t sizeUint32 = numeric_cast<uint32_t>(sizeof(uint32_t));

    // Error no space left in buffer
    MockBuffer mockBuffer1(10);
    SendCounterPacket sendPacket1(mockBuffer1);
    BOOST_CHECK_THROW(sendPacket1.SendStreamMetaDataPacket(), armnn::RuntimeException);

    // Full metadata packet

    std::string processName = GetProcessName().substr(0, 60);

    uint32_t infoSize = numeric_cast<uint32_t>(GetSoftwareInfo().size()) > 0 ?
                        numeric_cast<uint32_t>(GetSoftwareInfo().size()) + 1 : 0;
    uint32_t hardwareVersionSize = numeric_cast<uint32_t>(GetHardwareVersion().size()) > 0 ?
                                   numeric_cast<uint32_t>(GetHardwareVersion().size()) + 1 : 0;
    uint32_t softwareVersionSize = numeric_cast<uint32_t>(GetSoftwareVersion().size()) > 0 ?
                                   numeric_cast<uint32_t>(GetSoftwareVersion().size()) + 1 : 0;
    uint32_t processNameSize = numeric_cast<uint32_t>(processName.size()) > 0 ?
                               numeric_cast<uint32_t>(processName.size()) + 1 : 0;

    uint32_t packetEntries = 5;

    MockBuffer mockBuffer2(512);
    SendCounterPacket sendPacket2(mockBuffer2);
    sendPacket2.SendStreamMetaDataPacket();
    unsigned int sizeRead = 0;
    const unsigned char* readBuffer2 = mockBuffer2.GetReadBuffer(sizeRead);

    uint32_t headerWord0 = ReadUint32(readBuffer2, 0);
    uint32_t headerWord1 = ReadUint32(readBuffer2, sizeUint32);

    BOOST_TEST(((headerWord0 >> 26) & 0x3F) == 0); // packet family
    BOOST_TEST(((headerWord0 >> 16) & 0x3FF) == 0); // packet id

    uint32_t totalLength = numeric_cast<uint32_t>(2 * sizeUint32 + 10 * sizeUint32 + infoSize + hardwareVersionSize +
                                                  softwareVersionSize + processNameSize + sizeUint32 +
                                                  2 * packetEntries * sizeUint32);

    BOOST_TEST(headerWord1 == totalLength - (2 * sizeUint32)); // data length

    uint32_t offset = sizeUint32 * 2;
    BOOST_TEST(ReadUint32(readBuffer2, offset) == SendCounterPacket::PIPE_MAGIC); // pipe_magic
    offset += sizeUint32;
    BOOST_TEST(ReadUint32(readBuffer2, offset) == EncodeVersion(1, 0, 0)); // stream_metadata_version
    offset += sizeUint32;
    BOOST_TEST(ReadUint32(readBuffer2, offset) == SendCounterPacket::MAX_METADATA_PACKET_LENGTH); // max_data_len
    offset += sizeUint32;
    BOOST_TEST(ReadUint32(readBuffer2, offset) == numeric_cast<uint32_t>(getpid())); // pid
    offset += sizeUint32;
    uint32_t poolOffset = 10 * sizeUint32;
    BOOST_TEST(ReadUint32(readBuffer2, offset) == (infoSize ? poolOffset : 0)); // offset_info
    offset += sizeUint32;
    poolOffset += infoSize;
    BOOST_TEST(ReadUint32(readBuffer2, offset) == (hardwareVersionSize ? poolOffset : 0)); // offset_hw_version
    offset += sizeUint32;
    poolOffset += hardwareVersionSize;
    BOOST_TEST(ReadUint32(readBuffer2, offset) == (softwareVersionSize ? poolOffset : 0)); // offset_sw_version
    offset += sizeUint32;
    poolOffset += softwareVersionSize;
    BOOST_TEST(ReadUint32(readBuffer2, offset) == (processNameSize ? poolOffset : 0)); // offset_process_name
    offset += sizeUint32;
    poolOffset += processNameSize;
    BOOST_TEST(ReadUint32(readBuffer2, offset) == (packetEntries ? poolOffset : 0)); // offset_packet_version_table
    offset += sizeUint32;
    BOOST_TEST(ReadUint32(readBuffer2, offset) == 0); // reserved

    offset += sizeUint32;
    if (infoSize)
    {
        BOOST_TEST(strcmp(reinterpret_cast<const char *>(&readBuffer2[offset]), GetSoftwareInfo().c_str()) == 0);
        offset += infoSize;
    }

    if (hardwareVersionSize)
    {
        BOOST_TEST(strcmp(reinterpret_cast<const char *>(&readBuffer2[offset]), GetHardwareVersion().c_str()) == 0);
        offset += hardwareVersionSize;
    }

    if (softwareVersionSize)
    {
        BOOST_TEST(strcmp(reinterpret_cast<const char *>(&readBuffer2[offset]), GetSoftwareVersion().c_str()) == 0);
        offset += softwareVersionSize;
    }

    if (processNameSize)
    {
        BOOST_TEST(strcmp(reinterpret_cast<const char *>(&readBuffer2[offset]), GetProcessName().c_str()) == 0);
        offset += processNameSize;
    }

    if (packetEntries)
    {
        BOOST_TEST((ReadUint32(readBuffer2, offset) >> 16) == packetEntries);
        offset += sizeUint32;
        for (uint32_t i = 0; i < packetEntries; ++i)
        {
            BOOST_TEST(((ReadUint32(readBuffer2, offset) >> 26) & 0x3F) == 0);
            BOOST_TEST(((ReadUint32(readBuffer2, offset) >> 16) & 0x3FF) == i);
            offset += sizeUint32;
            BOOST_TEST(ReadUint32(readBuffer2, offset) == EncodeVersion(1, 0, 0));
            offset += sizeUint32;
        }
    }

    BOOST_TEST(offset == totalLength);
}


BOOST_AUTO_TEST_SUITE_END()
