//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../SendCounterPacket.hpp"
#include "../ProfilingUtils.hpp"

#include <armnn/Exceptions.hpp>

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <iostream>

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

    void SendCounterDirectoryPacket(const CounterDirectory& counterDirectory) override
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
