//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <CommandHandlerRegistry.hpp>
#include <DirectoryCaptureCommandHandler.hpp>
#include <ProfilingService.hpp>
#include <GatordMockService.hpp>
#include <PeriodicCounterCaptureCommandHandler.hpp>
#include <StreamMetadataCommandHandler.hpp>
#include <TimelineDirectoryCaptureCommandHandler.hpp>

#include <test/ProfilingMocks.hpp>

#include <boost/cast.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

BOOST_AUTO_TEST_SUITE(GatordMockTests)

using namespace armnn;
using namespace std::this_thread;    // sleep_for, sleep_until
using namespace std::chrono_literals;

BOOST_AUTO_TEST_CASE(CounterCaptureHandlingTest)
{
    using boost::numeric_cast;

    profiling::PacketVersionResolver packetVersionResolver;

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

    uint64_t time2 = time + 5000;

    // UniqueData required for Packet class
    std::unique_ptr<unsigned char[]> uniqueData1 = std::make_unique<unsigned char[]>(dataLength);
    unsigned char* data1                         = reinterpret_cast<unsigned char*>(uniqueData1.get());

    std::unique_ptr<unsigned char[]> uniqueData2 = std::make_unique<unsigned char[]>(dataLength);
    unsigned char* data2                         = reinterpret_cast<unsigned char*>(uniqueData2.get());

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

    uint32_t headerWord1 = packetVersionResolver.ResolvePacketVersion(0, 4).GetEncodedValue();
    // Create packet to send through to the command functor
    profiling::Packet packet1(headerWord1, dataLength, uniqueData1);
    profiling::Packet packet2(headerWord1, dataLength, uniqueData2);

    gatordmock::PeriodicCounterCaptureCommandHandler commandHandler(0, 4, headerWord1, true);

    // Simulate two separate packets coming in to calculate period
    commandHandler(packet1);
    commandHandler(packet2);

    BOOST_ASSERT(commandHandler.m_CurrentPeriodValue == 5000);

    for (size_t i = 0; i < commandHandler.m_CounterCaptureValues.m_Uids.size(); ++i)
    {
        BOOST_ASSERT(commandHandler.m_CounterCaptureValues.m_Uids[i] == i);
    }
}

BOOST_AUTO_TEST_CASE(GatorDMockEndToEnd)
{
    // The purpose of this test is to setup both sides of the profiling service and get to the point of receiving
    // performance data.

    //These variables are used to wait for the profiling service
    uint32_t timeout   = 2000;
    uint32_t sleepTime = 50;
    uint32_t timeSlept = 0;

    profiling::PacketVersionResolver packetVersionResolver;

    // Create the Command Handler Registry
    profiling::CommandHandlerRegistry registry;

    // Update with derived functors
    gatordmock::StreamMetadataCommandHandler streamMetadataCommandHandler(
        0, 0, packetVersionResolver.ResolvePacketVersion(0, 0).GetEncodedValue(), true);

    gatordmock::PeriodicCounterCaptureCommandHandler counterCaptureCommandHandler(
        0, 4, packetVersionResolver.ResolvePacketVersion(0, 4).GetEncodedValue(), true);

    profiling::DirectoryCaptureCommandHandler directoryCaptureCommandHandler(
        0, 2, packetVersionResolver.ResolvePacketVersion(0, 2).GetEncodedValue(), true);

    gatordmock::TimelineDirectoryCaptureCommandHandler timelineDirectoryCaptureCommandHandler(
        1, 0, packetVersionResolver.ResolvePacketVersion(1, 0).GetEncodedValue(), true);

    // Register different derived functors
    registry.RegisterFunctor(&streamMetadataCommandHandler);
    registry.RegisterFunctor(&counterCaptureCommandHandler);
    registry.RegisterFunctor(&directoryCaptureCommandHandler);
    registry.RegisterFunctor(&timelineDirectoryCaptureCommandHandler);
    // Setup the mock service to bind to the UDS.
    std::string udsNamespace = "gatord_namespace";
    gatordmock::GatordMockService mockService(registry, false);
    mockService.OpenListeningSocket(udsNamespace);

    // Enable the profiling service.
    armnn::IRuntime::CreationOptions::ExternalProfilingOptions options;
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
    while (profilingService.GetCurrentState() != profiling::ProfilingState::WaitingForAck)
    {
        if (timeSlept >= timeout)
        {
            BOOST_FAIL("Timeout: Profiling service did not switch to WaitingForAck state");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        timeSlept += sleepTime;
    }

    profilingService.Update();
    // Read the stream metadata on the mock side.
    if (!mockService.WaitForStreamMetaData())
    {
        BOOST_FAIL("Failed to receive StreamMetaData");
    }
    // Send Ack from GatorD
    mockService.SendConnectionAck();

    timeSlept = 0;
    while (profilingService.GetCurrentState() != profiling::ProfilingState::Active)
    {
        if (timeSlept >= timeout)
        {
            BOOST_FAIL("Timeout: Profiling service did not switch to Active state");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        timeSlept += sleepTime;
    }

    mockService.LaunchReceivingThread();
    // As part of the default startup of the profiling service a counter directory packet will be sent.
    timeSlept = 0;
    while (!directoryCaptureCommandHandler.ParsedCounterDirectory())
    {
        if (timeSlept >= timeout)
        {
            BOOST_FAIL("Timeout: MockGatord did not receive counter directory packet");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        timeSlept += sleepTime;
    }

    const profiling::ICounterDirectory& serviceCounterDirectory  = profilingService.GetCounterDirectory();
    const profiling::ICounterDirectory& receivedCounterDirectory = directoryCaptureCommandHandler.GetCounterDirectory();

    // Compare thre basics of the counter directory from the service and the one we received over the wire.
    BOOST_ASSERT(serviceCounterDirectory.GetDeviceCount() == receivedCounterDirectory.GetDeviceCount());
    BOOST_ASSERT(serviceCounterDirectory.GetCounterSetCount() == receivedCounterDirectory.GetCounterSetCount());
    BOOST_ASSERT(serviceCounterDirectory.GetCategoryCount() == receivedCounterDirectory.GetCategoryCount());
    BOOST_ASSERT(serviceCounterDirectory.GetCounterCount() == receivedCounterDirectory.GetCounterCount());

    receivedCounterDirectory.GetDeviceCount();
    serviceCounterDirectory.GetDeviceCount();

    const profiling::Devices& serviceDevices = serviceCounterDirectory.GetDevices();
    for (auto& device : serviceDevices)
    {
        // Find the same device in the received counter directory.
        auto foundDevice = receivedCounterDirectory.GetDevices().find(device.second->m_Uid);
        BOOST_CHECK(foundDevice != receivedCounterDirectory.GetDevices().end());
        BOOST_CHECK(device.second->m_Name.compare((*foundDevice).second->m_Name) == 0);
        BOOST_CHECK(device.second->m_Cores == (*foundDevice).second->m_Cores);
    }

    const profiling::CounterSets& serviceCounterSets = serviceCounterDirectory.GetCounterSets();
    for (auto& counterSet : serviceCounterSets)
    {
        // Find the same counter set in the received counter directory.
        auto foundCounterSet = receivedCounterDirectory.GetCounterSets().find(counterSet.second->m_Uid);
        BOOST_CHECK(foundCounterSet != receivedCounterDirectory.GetCounterSets().end());
        BOOST_CHECK(counterSet.second->m_Name.compare((*foundCounterSet).second->m_Name) == 0);
        BOOST_CHECK(counterSet.second->m_Count == (*foundCounterSet).second->m_Count);
    }

    const profiling::Categories& serviceCategories = serviceCounterDirectory.GetCategories();
    for (auto& category : serviceCategories)
    {
        for (auto& receivedCategory : receivedCounterDirectory.GetCategories())
        {
            if (receivedCategory->m_Name.compare(category->m_Name) == 0)
            {
                // We've found the matching category.
                BOOST_CHECK(category->m_DeviceUid == receivedCategory->m_DeviceUid);
                BOOST_CHECK(category->m_CounterSetUid == receivedCategory->m_CounterSetUid);
                // Now look at the interiors of the counters. Start by sorting them.
                std::sort(category->m_Counters.begin(), category->m_Counters.end());
                std::sort(receivedCategory->m_Counters.begin(), receivedCategory->m_Counters.end());
                // When comparing uid's here we need to translate them.
                std::function<bool(const uint16_t&, const uint16_t&)> comparator =
                    [&directoryCaptureCommandHandler](const uint16_t& first, const uint16_t& second) {
                        uint16_t translated = directoryCaptureCommandHandler.TranslateUIDCopyToOriginal(second);
                        if (translated == first)
                        {
                            return true;
                        }
                        return false;
                    };
                // Then let vector == do the work.
                BOOST_CHECK(std::equal(category->m_Counters.begin(), category->m_Counters.end(),
                                       receivedCategory->m_Counters.begin(), comparator));
                break;
            }
        }
    }

    // Finally check the content of the counters.
    const profiling::Counters& receivedCounters = receivedCounterDirectory.GetCounters();
    for (auto& receivedCounter : receivedCounters)
    {
        // Translate the Uid and find the corresponding counter in the original counter directory.
        // Note we can't check m_MaxCounterUid here as it will likely differ between the two counter directories.
        uint16_t translated = directoryCaptureCommandHandler.TranslateUIDCopyToOriginal(receivedCounter.first);
        const profiling::Counter* serviceCounter = serviceCounterDirectory.GetCounter(translated);
        BOOST_CHECK(serviceCounter->m_DeviceUid == receivedCounter.second->m_DeviceUid);
        BOOST_CHECK(serviceCounter->m_Name.compare(receivedCounter.second->m_Name) == 0);
        BOOST_CHECK(serviceCounter->m_CounterSetUid == receivedCounter.second->m_CounterSetUid);
        BOOST_CHECK(serviceCounter->m_Multiplier == receivedCounter.second->m_Multiplier);
        BOOST_CHECK(serviceCounter->m_Interpolation == receivedCounter.second->m_Interpolation);
        BOOST_CHECK(serviceCounter->m_Class == receivedCounter.second->m_Class);
        BOOST_CHECK(serviceCounter->m_Units.compare(receivedCounter.second->m_Units) == 0);
        BOOST_CHECK(serviceCounter->m_Description.compare(receivedCounter.second->m_Description) == 0);
    }

    mockService.WaitForReceivingThread();
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Future tests here will add counters to the ProfilingService, increment values and examine
    // PeriodicCounterCapture data received. These are yet to be integrated.
}

BOOST_AUTO_TEST_SUITE_END()
