//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../GatordMockService.hpp"
#include "../PeriodicCounterCaptureCommandHandler.hpp"
#include "../DirectoryCaptureCommandHandler.hpp"

#include <CommandHandlerRegistry.hpp>
#include <ProfilingService.hpp>

#include <test/SendCounterPacketTests.hpp>

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

    gatordmock::PeriodicCounterCaptureCommandHandler commandHandler
        (0, 4, headerWord1, true);

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
    u_int32_t timeout = 2000;
    u_int32_t sleepTime = 50;
    u_int32_t timeSlept = 0;

    profiling::PacketVersionResolver packetVersionResolver;

    // Create the Command Handler Registry
    profiling::CommandHandlerRegistry registry;

    // Update with derived functors
    gatordmock::PeriodicCounterCaptureCommandHandler counterCaptureCommandHandler
        (0, 4, packetVersionResolver.ResolvePacketVersion(0, 4).GetEncodedValue(), true);

    gatordmock::DirectoryCaptureCommandHandler directoryCaptureCommandHandler
        (0, 2, packetVersionResolver.ResolvePacketVersion(0, 2).GetEncodedValue(), true);

    // Register different derived functors
    registry.RegisterFunctor(&counterCaptureCommandHandler);
    registry.RegisterFunctor(&directoryCaptureCommandHandler);
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
    mockService.SendRequestCounterDir();

    timeSlept = 0;
    while (directoryCaptureCommandHandler.GetCounterDirectoryCount() == 0)
    {
        if (timeSlept >= timeout)
        {
            BOOST_FAIL("Timeout: MockGatord did not receive counter directory packet");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        timeSlept += sleepTime;
    }

    const profiling::ICounterDirectory& serviceCounterDirectory = profilingService.GetCounterDirectory();
    gatordmock::CounterDirectory mockCounterDirectory = directoryCaptureCommandHandler.GetCounterDirectory();

    BOOST_ASSERT(serviceCounterDirectory.GetDeviceCount() ==  mockCounterDirectory.m_DeviceRecords.size());
    BOOST_ASSERT(serviceCounterDirectory.GetCounterSetCount() ==  mockCounterDirectory.m_CounterSets.size());
    BOOST_ASSERT(serviceCounterDirectory.GetCategoryCount() ==  mockCounterDirectory.m_Categories.size());

    const profiling::Devices& serviceDevices = serviceCounterDirectory.GetDevices();

    uint32_t deviceIndex = 0;
    for (auto& device : serviceDevices)
    {
        BOOST_ASSERT(device.second->m_Name.size() ==
                     mockCounterDirectory.m_DeviceRecords[deviceIndex].m_DeviceName.size());

        BOOST_CHECK(device.second->m_Name == mockCounterDirectory.m_DeviceRecords[deviceIndex].m_DeviceName);
        BOOST_CHECK(device.second->m_Uid == mockCounterDirectory.m_DeviceRecords[deviceIndex].m_DeviceUid);
        BOOST_CHECK(device.second->m_Cores == mockCounterDirectory.m_DeviceRecords[deviceIndex].m_DeviceCores);
        deviceIndex++;
    }

    const profiling::CounterSets & serviceCounterSets = serviceCounterDirectory.GetCounterSets();
    uint32_t counterSetIndex = 0;
    for (auto& counterSet : serviceCounterSets)
    {
        BOOST_ASSERT(counterSet.second->m_Name.size() ==
                     mockCounterDirectory.m_CounterSets[counterSetIndex].m_CounterSetName.size());

        BOOST_CHECK(counterSet.second->m_Name == mockCounterDirectory.m_CounterSets[counterSetIndex].m_CounterSetName);
        BOOST_CHECK(counterSet.second->m_Uid == mockCounterDirectory.m_CounterSets[counterSetIndex].m_CounterSetUid);
        BOOST_CHECK(counterSet.second->m_Count ==
                    mockCounterDirectory.m_CounterSets[counterSetIndex].m_CounterSetCount);
        counterSetIndex++;
    }

    const profiling::Categories& serviceCategories = serviceCounterDirectory.GetCategories();
    const std::vector<gatordmock::CategoryRecord> mockCategories = mockCounterDirectory.m_Categories;

    uint32_t categoryIndex = 0;
    for (auto& category : serviceCategories)
    {
        BOOST_ASSERT(category->m_Name.size() == mockCategories[categoryIndex].m_CategoryName.size());

        BOOST_CHECK(category->m_Name == mockCategories[categoryIndex].m_CategoryName);
        BOOST_CHECK(category->m_CounterSetUid == mockCategories[categoryIndex].m_CounterSet);
        BOOST_CHECK(category->m_DeviceUid == mockCategories[categoryIndex].m_DeviceUid);

        const std::vector<gatordmock::EventRecord> events = mockCategories[categoryIndex].m_EventRecords;
        uint32_t eventIndex = 0;
        for (uint16_t counterUid : category->m_Counters)
        {
            const profiling::Counter* counter = serviceCounterDirectory.GetCounter(counterUid);

            BOOST_CHECK(counterUid == events[eventIndex].m_CounterUid);

            BOOST_ASSERT(counter->m_Name.size() == events[eventIndex].m_CounterName.size());
            BOOST_ASSERT(counter->m_Units.size() == events[eventIndex].m_CounterUnits.size());
            BOOST_ASSERT(counter->m_Description.size() == events[eventIndex].m_CounterDescription.size());

            BOOST_CHECK(counter->m_Name == events[eventIndex].m_CounterName);
            BOOST_CHECK(counter->m_Units == events[eventIndex].m_CounterUnits);
            BOOST_CHECK(counter->m_Description == events[eventIndex].m_CounterDescription);

            BOOST_CHECK(counter->m_CounterSetUid == events[eventIndex].m_CounterSetUid);
            BOOST_CHECK(counter->m_DeviceUid == events[eventIndex].m_DeviceUid);
            BOOST_CHECK(counter->m_Uid == events[eventIndex].m_CounterUid);

            BOOST_CHECK(counter->m_Multiplier == events[eventIndex].m_CounterMultiplier);
            BOOST_CHECK(counter->m_MaxCounterUid == events[eventIndex].m_MaxCounterUid);
            BOOST_CHECK(counter->m_Interpolation == events[eventIndex].m_CounterInterpolation);
            BOOST_CHECK(counter->m_Class == events[eventIndex].m_CounterClass);

            eventIndex++;
        }
        categoryIndex++;
    }

    mockService.WaitForReceivingThread();
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Future tests here will add counters to the ProfilingService, increment values and examine
    // PeriodicCounterCapture data received. These are yet to be integrated.
}

BOOST_AUTO_TEST_SUITE_END()
