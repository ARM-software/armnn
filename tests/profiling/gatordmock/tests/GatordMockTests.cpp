//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../MockUtils.hpp"
#include "../PeriodicCounterCaptureCommandHandler.hpp"

#include "../../src/profiling/CommandHandlerRegistry.hpp"
#include "../GatordMockService.hpp"

#include "../../src/profiling/ProfilingService.hpp"
#include "../../src/profiling/test/SendCounterPacketTests.hpp"

#include <boost/cast.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

BOOST_AUTO_TEST_SUITE(GatordMockTests)

using namespace armnn;
using namespace std::this_thread;    // sleep_for, sleep_until
using namespace std::chrono_literals;

// Required so build succeeds when local variable used only in assert
#define _unused(x) ((void)(x))

BOOST_AUTO_TEST_CASE(CounterCaptureHandlingTest)
{
    using boost::numeric_cast;

    // Initialise functors and register into the CommandHandlerRegistry
    uint32_t headerWord1 = gatordmock::ConstructHeader(1, 0, 0);

    // Create the Command Handler Registry
    profiling::CommandHandlerRegistry registry;

    // Data with timestamp, counter idx & counter values
    std::vector<std::pair<uint16_t, uint32_t>> indexValuePairs;
    indexValuePairs.reserve(5);
    indexValuePairs.emplace_back(std::make_pair<uint16_t, uint32_t>(0, 100));
    indexValuePairs.emplace_back(std::make_pair<uint16_t, uint32_t>(1, 200));
    indexValuePairs.emplace_back(std::make_pair<uint16_t, uint32_t>(2, 300));
    indexValuePairs.emplace_back(std::make_pair<uint16_t, uint32_t>(3, 400));
    indexValuePairs.emplace_back(std::make_pair<uint16_t, uint32_t>(4, 500));

    // ((uint16_t (2 bytes) + uint32_t (4 bytes)) * 5) + word1 + word2
    uint32_t dataLength = 38;

    // Simulate two different packets incoming 500 ms apart
    uint64_t time = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());

    sleep_for(5000us);

    uint64_t time2 = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());

    // UniqueData required for Packet class
    std::unique_ptr<unsigned char[]> uniqueData1 = std::make_unique<unsigned char[]>(dataLength);
    unsigned char* data1                = reinterpret_cast<unsigned char*>(uniqueData1.get());

    std::unique_ptr<unsigned char[]> uniqueData2 = std::make_unique<unsigned char[]>(dataLength);
    unsigned char* data2                = reinterpret_cast<unsigned char*>(uniqueData2.get());

    uint32_t sizeOfUint64 = numeric_cast<uint32_t>(sizeof(uint64_t));
    uint32_t sizeOfUint32 = numeric_cast<uint32_t>(sizeof(uint32_t));
    uint32_t sizeOfUint16 = numeric_cast<uint32_t>(sizeof(uint16_t));
    // Offset index to point to mem address
    uint32_t offset = 0;

    profiling::WriteUint64(data1, offset, time);
    offset += sizeOfUint64;
    for (const auto& pair : indexValuePairs)
    {
        profiling::WriteUint16(data1, offset, pair.first);
        offset += sizeOfUint16;
        profiling::WriteUint32(data1, offset, pair.second);
        offset += sizeOfUint32;
    }

    offset = 0;

    profiling::WriteUint64(data2, offset, time2);
    offset += sizeOfUint64;
    for (const auto& pair : indexValuePairs)
    {
        profiling::WriteUint16(data2, offset, pair.first);
        offset += sizeOfUint16;
        profiling::WriteUint32(data2, offset, pair.second);
        offset += sizeOfUint32;
    }

    // Create packet to send through to the command functor
    profiling::Packet packet1(headerWord1, dataLength, uniqueData1);
    profiling::Packet packet2(headerWord1, dataLength, uniqueData2);

    uint32_t version = 1;
    gatordmock::PeriodicCounterCaptureCommandHandler commandHandler(headerWord1, version, false);

    // Simulate two separate packets coming in to calculate period
    commandHandler(packet1);
    commandHandler(packet2);

    BOOST_ASSERT(4500 < commandHandler.m_CurrentPeriodValue && 5500 > commandHandler.m_CurrentPeriodValue);

    for (size_t i = 0; i < commandHandler.m_CounterCaptureValues.m_Uids.size(); ++i)
    {
        BOOST_ASSERT(commandHandler.m_CounterCaptureValues.m_Uids[i] == i);
    }
}

BOOST_AUTO_TEST_CASE(GatorDMockEndToEnd)
{
    // The purpose of this test is to setup both sides of the profiling service and get to the point of receiving
    // performance data.

    // Initialise functors and register into the CommandHandlerRegistry
    uint32_t counterCaptureHeader = gatordmock::ConstructHeader(1, 0);
    uint32_t version              = 1;

    // Create the Command Handler Registry
    profiling::CommandHandlerRegistry registry;

    // Update with derived functors
    gatordmock::PeriodicCounterCaptureCommandHandler counterCaptureCommandHandler(counterCaptureHeader, version, false);
    // Register different derived functors
    registry.RegisterFunctor(&counterCaptureCommandHandler);

    // Setup the mock service to bind to the UDS.
    std::string udsNamespace = "gatord_namespace";
    gatordmock::GatordMockService mockService(registry, false);
    mockService.OpenListeningSocket(udsNamespace);

    // Enable the profiling service.
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling                     = true;
    profiling::ProfilingService& profilingService = profiling::ProfilingService::Instance();
    profilingService.ResetExternalProfilingOptions(options, true);

    // Bring the profiling service to the "WaitingForAck" state
    BOOST_CHECK(profilingService.GetCurrentState() == profiling::ProfilingState::Uninitialised);
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState() == profiling::ProfilingState::NotConnected);
    profilingService.Update();

    // Connect the profiling service to the mock Gatord.
    int clientFd = mockService.BlockForOneClient();
    if (-1 == clientFd)
    {
        BOOST_FAIL("Failed to connect client");
    }
    // Give the profiling service sending thread time start executing and send the stream metadata.
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // We should now be in WaitingForAck state.
    BOOST_CHECK(profilingService.GetCurrentState() == profiling::ProfilingState::WaitingForAck);
    profilingService.Update();
    // Read the stream metadata on the mock side.
    if (!mockService.WaitForStreamMetaData())
    {
        BOOST_FAIL("Failed to receive StreamMetaData");
    }
    // Send Ack from GatorD
    mockService.SendConnectionAck();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // At this point the service should be in active state.
    BOOST_ASSERT(profilingService.GetCurrentState() == profiling::ProfilingState::Active);

    // Future tests here will add counters to the ProfilingService, increment values and examine
    // PeriodicCounterCapture data received. These are yet to be integrated.

    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_SUITE_END()
