//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../IBufferWrapper.hpp"
#include "../ISendCounterPacket.hpp"

#include <iostream>
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(SendCounterPacketTests)

using namespace armnn::profiling;

class MockBuffer : public IBufferWrapper
{
public:
    MockBuffer() : m_Buffer() {}

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

        return m_Buffer;
    }

    void Commit(unsigned int size) override {}

    const unsigned char* GetReadBuffer(unsigned int& size) override
    {
        size = static_cast<unsigned int>(strlen(reinterpret_cast<const char*>(m_Buffer)) + 1);
        return m_Buffer;
    }

    void Release( unsigned int size) override {}

private:
    static const unsigned int m_BufferSize = 512;
    unsigned char m_Buffer[m_BufferSize];
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

    MockBuffer mockBuffer;
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

BOOST_AUTO_TEST_SUITE_END()
