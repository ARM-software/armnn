//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingMocks.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <client/src/IProfilingConnection.hpp>
#include <client/src/ProfilingService.hpp>

#include <armnn/profiling/ArmNNProfiling.hpp>

#include <common/include/CommandHandlerFunctor.hpp>
#include <common/include/Logging.hpp>

#include <doctest/doctest.h>

#include <chrono>
#include <thread>

namespace arm
{

namespace pipe
{

class TestProfilingConnectionBase : public IProfilingConnection
{
public:
    TestProfilingConnectionBase() = default;
    ~TestProfilingConnectionBase() = default;

    bool IsOpen() const override { return true; }

    void Close() override {}

    bool WritePacket(const unsigned char* buffer, uint32_t length) override
    {
        arm::pipe::IgnoreUnused(buffer, length);

        return false;
    }

    arm::pipe::Packet ReadPacket(uint32_t timeout) override
    {
        // First time we're called return a connection ack packet. After that always timeout.
        if (m_FirstCall)
        {
            m_FirstCall = false;
            // Return connection acknowledged packet
            return arm::pipe::Packet(65536);
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
            throw arm::pipe::TimeoutException("Simulate a timeout error\n");
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

    arm::pipe::Packet ReadPacket(uint32_t timeout) override
    {
        // Return connection acknowledged packet after three timeouts
        if (m_ReadRequests % 3 == 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
            ++m_ReadRequests;
            throw arm::pipe::TimeoutException("Simulate a timeout error\n");
        }

        return arm::pipe::Packet(65536);
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

    arm::pipe::Packet ReadPacket(uint32_t timeout) override
    {
        arm::pipe::IgnoreUnused(timeout);
        ++m_ReadRequests;
        throw arm::pipe::ProfilingException("Simulate a non-timeout error");
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
    arm::pipe::Packet ReadPacket(uint32_t timeout) override
    {
        arm::pipe::IgnoreUnused(timeout);
        // Connection Acknowledged Packet header (word 0, word 1 is always zero):
        // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
        // 16:25 [10] packet_id: Packet identifier, value 0b0000000001
        // 8:15  [8]  reserved: Reserved, value 0b00000000
        // 0:7   [8]  reserved: Reserved, value 0b00000000
        uint32_t packetFamily = 0;
        uint32_t packetId     = 37;    // Wrong packet id!!!
        uint32_t header       = ((packetFamily & 0x0000003F) << 26) | ((packetId & 0x000003FF) << 16);

        return arm::pipe::Packet(header);
    }
};

class TestFunctorA : public arm::pipe::CommandHandlerFunctor
{
public:
    using CommandHandlerFunctor::CommandHandlerFunctor;

    int GetCount() { return m_Count; }

    void operator()(const arm::pipe::Packet& packet) override
    {
        arm::pipe::IgnoreUnused(packet);
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

    SwapProfilingConnectionFactoryHelper(uint16_t maxGlobalCounterId,
                                         IInitialiseProfilingService& initialiser,
                                         ProfilingService& profilingService)
        : ProfilingService(maxGlobalCounterId,
                           initialiser,
                           arm::pipe::ARMNN_SOFTWARE_INFO,
                           arm::pipe::ARMNN_SOFTWARE_VERSION,
                           arm::pipe::ARMNN_HARDWARE_VERSION)
        , m_ProfilingService(profilingService)
        , m_MockProfilingConnectionFactory(new MockProfilingConnectionFactory())
        , m_BackupProfilingConnectionFactory(nullptr)

    {
        CHECK(m_MockProfilingConnectionFactory);
        SwapProfilingConnectionFactory(m_ProfilingService,
                                       m_MockProfilingConnectionFactory.get(),
                                       m_BackupProfilingConnectionFactory);
        CHECK(m_BackupProfilingConnectionFactory);
    }
    ~SwapProfilingConnectionFactoryHelper()
    {
        CHECK(m_BackupProfilingConnectionFactory);
        IProfilingConnectionFactory* temp = nullptr;
        SwapProfilingConnectionFactory(m_ProfilingService,
                                       m_BackupProfilingConnectionFactory,
                                       temp);
    }

    MockProfilingConnection* GetMockProfilingConnection()
    {
        IProfilingConnection* profilingConnection = GetProfilingConnection(m_ProfilingService);
        return armnn::PolymorphicDowncast<MockProfilingConnection*>(profilingConnection);
    }

    void ForceTransitionToState(ProfilingState newState)
    {
        TransitionToState(m_ProfilingService, newState);
    }

    long WaitForPacketsSent(MockProfilingConnection* mockProfilingConnection,
                            MockProfilingConnection::PacketType packetType,
                            uint32_t length = 0,
                            uint32_t timeout  = 1000)
    {
        long packetCount = mockProfilingConnection->CheckForPacket({ packetType, length });
        // The first packet we receive may not be the one we are looking for, so keep looping until till we find it,
        // or until WaitForPacketsSent times out
        while(packetCount == 0 && timeout != 0)
        {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            // Wait for a notification from the send thread
            ProfilingService::WaitForPacketSent(m_ProfilingService, timeout);

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
    ProfilingService& m_ProfilingService;
    MockProfilingConnectionFactoryPtr m_MockProfilingConnectionFactory;
    IProfilingConnectionFactory* m_BackupProfilingConnectionFactory;
};

} // namespace pipe

} // namespace arm
