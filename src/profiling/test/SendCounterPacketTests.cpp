//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../SendCounterPacket.hpp"
#include "../ProfilingUtils.hpp"

#include <armnn/Exceptions.hpp>

#include <boost/test/unit_test.hpp>

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

    void SendPeriodicCounterCapturePacket(uint64_t timestamp, const std::vector<uint32_t>& counterValues,
                                          const std::vector<uint16_t>& counterUids) override
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
    std::vector<uint32_t> counterValues;
    std::vector<uint16_t> counterUids;
    sendCounterPacket.SendPeriodicCounterCapturePacket(timestamp, counterValues, counterUids);

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
                      armnn::Exception);

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

BOOST_AUTO_TEST_SUITE_END()
