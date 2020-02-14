//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingMocks.hpp"

#include <armnn/Logging.hpp>

#include <CommandHandlerFunctor.hpp>
#include <IProfilingConnection.hpp>
#include <ProfilingService.hpp>

#include <boost/polymorphic_cast.hpp>
#include <boost/test/unit_test.hpp>

#include <chrono>
#include <thread>

namespace armnn
{

namespace profiling
{

struct LogLevelSwapper
{
public:
    LogLevelSwapper(armnn::LogSeverity severity)
    {
        // Set the new log level
        armnn::ConfigureLogging(true, true, severity);
    }
    ~LogLevelSwapper()
    {
        // The default log level for unit tests is "Fatal"
        armnn::ConfigureLogging(true, true, armnn::LogSeverity::Fatal);
    }
};

struct StreamRedirector
{
public:
    StreamRedirector(std::ostream& stream, std::streambuf* newStreamBuffer)
        : m_Stream(stream)
        , m_BackupBuffer(m_Stream.rdbuf(newStreamBuffer))
    {}

    ~StreamRedirector() { CancelRedirect(); }

    void CancelRedirect()
    {
        // Only cancel the redirect once.
        if (m_BackupBuffer != nullptr )
        {
            m_Stream.rdbuf(m_BackupBuffer);
            m_BackupBuffer = nullptr;
        }
    }

private:
    std::ostream& m_Stream;
    std::streambuf* m_BackupBuffer;
};

class TestProfilingConnectionBase : public IProfilingConnection
{
public:
    TestProfilingConnectionBase() = default;
    ~TestProfilingConnectionBase() = default;

    bool IsOpen() const override { return true; }

    void Close() override {}

    bool WritePacket(const unsigned char* buffer, uint32_t length) override
    {
        boost::ignore_unused(buffer, length);

        return false;
    }

    Packet ReadPacket(uint32_t timeout) override
    {
        // First time we're called return a connection ack packet. After that always timeout.
        if (m_FirstCall)
        {
            m_FirstCall = false;
            // Return connection acknowledged packet
            return Packet(65536);
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
            throw armnn::TimeoutException("Simulate a timeout error\n");
        }
    }

    bool m_FirstCall = true;
};

class TestProfilingConnectionTimeoutError : public TestProfilingConnectionBase
{
public:
    TestProfilingConnectionTimeoutError()
        : m_ReadRequests(0)
    {}

    Packet ReadPacket(uint32_t timeout) override
    {
        // Return connection acknowledged packet after three timeouts
        if (m_ReadRequests % 3 == 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
            ++m_ReadRequests;
            throw armnn::TimeoutException("Simulate a timeout error\n");
        }

        return Packet(65536);
    }

    int ReadCalledCount()
    {
        return m_ReadRequests.load();
    }

private:
    std::atomic<int> m_ReadRequests;
};

class TestProfilingConnectionArmnnError : public TestProfilingConnectionBase
{
public:
    TestProfilingConnectionArmnnError()
        : m_ReadRequests(0)
    {}

    Packet ReadPacket(uint32_t timeout) override
    {
        boost::ignore_unused(timeout);
        ++m_ReadRequests;
        throw armnn::Exception("Simulate a non-timeout error");
    }

    int ReadCalledCount()
    {
        return m_ReadRequests.load();
    }

private:
    std::atomic<int> m_ReadRequests;
};

class TestProfilingConnectionBadAckPacket : public TestProfilingConnectionBase
{
public:
    Packet ReadPacket(uint32_t timeout) override
    {
        boost::ignore_unused(timeout);
        // Connection Acknowledged Packet header (word 0, word 1 is always zero):
        // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
        // 16:25 [10] packet_id: Packet identifier, value 0b0000000001
        // 8:15  [8]  reserved: Reserved, value 0b00000000
        // 0:7   [8]  reserved: Reserved, value 0b00000000
        uint32_t packetFamily = 0;
        uint32_t packetId     = 37;    // Wrong packet id!!!
        uint32_t header       = ((packetFamily & 0x0000003F) << 26) | ((packetId & 0x000003FF) << 16);

        return Packet(header);
    }
};

class TestFunctorA : public CommandHandlerFunctor
{
public:
    using CommandHandlerFunctor::CommandHandlerFunctor;

    int GetCount() { return m_Count; }

    void operator()(const Packet& packet) override
    {
        boost::ignore_unused(packet);
        m_Count++;
    }

private:
    int m_Count = 0;
};

class TestFunctorB : public TestFunctorA
{
    using TestFunctorA::TestFunctorA;
};

class TestFunctorC : public TestFunctorA
{
    using TestFunctorA::TestFunctorA;
};

class SwapProfilingConnectionFactoryHelper : public ProfilingService
{
public:
    using MockProfilingConnectionFactoryPtr = std::unique_ptr<MockProfilingConnectionFactory>;

    SwapProfilingConnectionFactoryHelper()
        : ProfilingService()
        , m_MockProfilingConnectionFactory(new MockProfilingConnectionFactory())
        , m_BackupProfilingConnectionFactory(nullptr)
    {
        BOOST_CHECK(m_MockProfilingConnectionFactory);
        SwapProfilingConnectionFactory(ProfilingService::Instance(),
                                       m_MockProfilingConnectionFactory.get(),
                                       m_BackupProfilingConnectionFactory);
        BOOST_CHECK(m_BackupProfilingConnectionFactory);
    }
    ~SwapProfilingConnectionFactoryHelper()
    {
        BOOST_CHECK(m_BackupProfilingConnectionFactory);
        IProfilingConnectionFactory* temp = nullptr;
        SwapProfilingConnectionFactory(ProfilingService::Instance(),
                                       m_BackupProfilingConnectionFactory,
                                       temp);
    }

    MockProfilingConnection* GetMockProfilingConnection()
    {
        IProfilingConnection* profilingConnection = GetProfilingConnection(ProfilingService::Instance());
        return boost::polymorphic_downcast<MockProfilingConnection*>(profilingConnection);
    }

    void ForceTransitionToState(ProfilingState newState)
    {
        TransitionToState(ProfilingService::Instance(), newState);
    }

    long WaitForPacketsSent(MockProfilingConnection* mockProfilingConnection,
                            MockProfilingConnection::PacketType packetType,
                            uint32_t length = 0,
                            uint32_t timeout  = 1000)
    {
        long packetCount = mockProfilingConnection->CheckForPacket({packetType, length});
        // The first packet we receive may not be the one we are looking for, so keep looping until till we find it,
        // or until WaitForPacketsSent times out
        while(packetCount == 0 && timeout != 0)
        {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            // Wait for a notification from the send thread
            ProfilingService::WaitForPacketSent(ProfilingService::Instance(), timeout);

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            // We need to make sure the timeout does not reset each time we call WaitForPacketsSent
            uint32_t elapsedTime = static_cast<uint32_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

            packetCount = mockProfilingConnection->CheckForPacket({packetType, length});

            if (elapsedTime > timeout)
            {
                break;
            }

            timeout -= elapsedTime;
        }
        return packetCount;
    }

private:
    MockProfilingConnectionFactoryPtr m_MockProfilingConnectionFactory;
    IProfilingConnectionFactory* m_BackupProfilingConnectionFactory;
};

} // namespace profiling

} // namespace armnn
