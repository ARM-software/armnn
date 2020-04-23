//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <CommandHandlerRegistry.hpp>
#include <DirectoryCaptureCommandHandler.hpp>
#include <GatordMockService.hpp>
#include <LabelsAndEventClasses.hpp>
#include <PeriodicCounterCaptureCommandHandler.hpp>
#include <ProfilingService.hpp>
#include <TimelinePacketWriterFactory.hpp>

#include <TimelineDirectoryCaptureCommandHandler.hpp>
#include <TimelineDecoder.hpp>

#include <Runtime.hpp>
#include "../../src/backends/backendsCommon/test/MockBackend.hpp"

#include <boost/cast.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

BOOST_AUTO_TEST_SUITE(GatordMockTests)

using namespace armnn;
using namespace std::this_thread;
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

    ARMNN_ASSERT(commandHandler.m_CurrentPeriodValue == 5000);

    for (size_t i = 0; i < commandHandler.m_CounterCaptureValues.m_Uids.size(); ++i)
    {
        ARMNN_ASSERT(commandHandler.m_CounterCaptureValues.m_Uids[i] == i);
    }
}

void WaitFor(std::function<bool()> predicate, std::string errorMsg, uint32_t timeout = 2000, uint32_t sleepTime = 50)
{
    uint32_t timeSlept = 0;
    while (!predicate())
    {
        if (timeSlept >= timeout)
        {
            BOOST_FAIL("Timeout: " + errorMsg);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        timeSlept += sleepTime;
    }
}

void CheckTimelineDirectory(timelinedecoder::TimelineDirectoryCaptureCommandHandler& commandHandler)
{
    uint32_t uint8_t_size  = sizeof(uint8_t);
    uint32_t uint32_t_size = sizeof(uint32_t);
    uint32_t uint64_t_size = sizeof(uint64_t);
    uint32_t threadId_size = sizeof(std::thread::id);

    profiling::BufferManager bufferManager(5);
    profiling::TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);

    std::unique_ptr<profiling::ISendTimelinePacket> sendTimelinePacket =
            timelinePacketWriterFactory.GetSendTimelinePacket();

    sendTimelinePacket->SendTimelineMessageDirectoryPackage();
    sendTimelinePacket->Commit();

    std::vector<profiling::SwTraceMessage> swTraceBufferMessages;

    unsigned int offset = uint32_t_size * 2;

    std::unique_ptr<profiling::IPacketBuffer> packetBuffer = bufferManager.GetReadableBuffer();

    uint8_t readStreamVersion = ReadUint8(packetBuffer, offset);
    BOOST_CHECK(readStreamVersion == 4);
    offset += uint8_t_size;
    uint8_t readPointerBytes = ReadUint8(packetBuffer, offset);
    BOOST_CHECK(readPointerBytes == uint64_t_size);
    offset += uint8_t_size;
    uint8_t readThreadIdBytes = ReadUint8(packetBuffer, offset);
    BOOST_CHECK(readThreadIdBytes == threadId_size);
    offset += uint8_t_size;

    uint32_t declarationSize = profiling::ReadUint32(packetBuffer, offset);
    offset += uint32_t_size;
    for(uint32_t i = 0; i < declarationSize; ++i)
    {
        swTraceBufferMessages.push_back(profiling::ReadSwTraceMessage(packetBuffer->GetReadableData(), offset));
    }

    for(uint32_t index = 0; index < declarationSize; ++index)
    {
        profiling::SwTraceMessage& bufferMessage = swTraceBufferMessages[index];
        profiling::SwTraceMessage& handlerMessage = commandHandler.m_SwTraceMessages[index];

        BOOST_CHECK(bufferMessage.m_Name == handlerMessage.m_Name);
        BOOST_CHECK(bufferMessage.m_UiName == handlerMessage.m_UiName);
        BOOST_CHECK(bufferMessage.m_Id == handlerMessage.m_Id);

        BOOST_CHECK(bufferMessage.m_ArgTypes.size() == handlerMessage.m_ArgTypes.size());
        for(uint32_t i = 0; i < bufferMessage.m_ArgTypes.size(); ++i)
        {
            BOOST_CHECK(bufferMessage.m_ArgTypes[i] == handlerMessage.m_ArgTypes[i]);
        }

        BOOST_CHECK(bufferMessage.m_ArgNames.size() == handlerMessage.m_ArgNames.size());
        for(uint32_t i = 0; i < bufferMessage.m_ArgNames.size(); ++i)
        {
            BOOST_CHECK(bufferMessage.m_ArgNames[i] == handlerMessage.m_ArgNames[i]);
        }
    }
}

void CheckTimelinePackets(timelinedecoder::TimelineDecoder& timelineDecoder)
{
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[0].m_Guid == profiling::LabelsAndEventClasses::NAME_GUID);
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[0].m_Name == profiling::LabelsAndEventClasses::NAME_LABEL);

    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[1].m_Guid == profiling::LabelsAndEventClasses::TYPE_GUID);
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[1].m_Name == profiling::LabelsAndEventClasses::TYPE_LABEL);

    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[2].m_Guid == profiling::LabelsAndEventClasses::INDEX_GUID);
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[2].m_Name == profiling::LabelsAndEventClasses::INDEX_LABEL);

    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[3].m_Guid == profiling::LabelsAndEventClasses::BACKENDID_GUID);
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[3].m_Name == profiling::LabelsAndEventClasses::BACKENDID_LABEL);

    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[4].m_Guid == profiling::LabelsAndEventClasses::LAYER_GUID);
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[4].m_Name == profiling::LabelsAndEventClasses::LAYER);

    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[5].m_Guid == profiling::LabelsAndEventClasses::WORKLOAD_GUID);
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[5].m_Name == profiling::LabelsAndEventClasses::WORKLOAD);

    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[6].m_Guid == profiling::LabelsAndEventClasses::NETWORK_GUID);
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[6].m_Name == profiling::LabelsAndEventClasses::NETWORK);

    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[7].m_Guid == profiling::LabelsAndEventClasses::CONNECTION_GUID);
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[7].m_Name == profiling::LabelsAndEventClasses::CONNECTION);

    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[8].m_Guid == profiling::LabelsAndEventClasses::INFERENCE_GUID);
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[8].m_Name == profiling::LabelsAndEventClasses::INFERENCE);

    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[9].m_Guid ==
                profiling::LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID);
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels[9].m_Name == profiling::LabelsAndEventClasses::WORKLOAD_EXECUTION);

    BOOST_CHECK(timelineDecoder.GetModel().m_EventClasses[0].m_Guid ==
                profiling::LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS);
    BOOST_CHECK(timelineDecoder.GetModel().m_EventClasses[1].m_Guid ==
                profiling::LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS);
}

BOOST_AUTO_TEST_CASE(GatorDMockEndToEnd)
{
    // The purpose of this test is to setup both sides of the profiling service and get to the point of receiving
    // performance data.

    // Setup the mock service to bind to the UDS.
    std::string udsNamespace = "gatord_namespace";

    armnnUtils::Sockets::Initialize();
    armnnUtils::Sockets::Socket listeningSocket = socket(PF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);

    if (!gatordmock::GatordMockService::OpenListeningSocket(listeningSocket, udsNamespace))
    {
        BOOST_FAIL("Failed to open Listening Socket");
    }

    // Enable the profiling service.
    armnn::IRuntime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling = true;
    options.m_TimelineEnabled = true;

    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Bring the profiling service to the "WaitingForAck" state
    BOOST_CHECK(profilingService.GetCurrentState() == profiling::ProfilingState::Uninitialised);
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState() == profiling::ProfilingState::NotConnected);
    profilingService.Update();

    // Connect the profiling service to the mock Gatord.
    armnnUtils::Sockets::Socket clientSocket =
            armnnUtils::Sockets::Accept(listeningSocket, nullptr, nullptr, SOCK_CLOEXEC);
    if (-1 == clientSocket)
    {
        BOOST_FAIL("Failed to connect client");
    }

    gatordmock::GatordMockService mockService(clientSocket, false);

    timelinedecoder::TimelineDecoder& timelineDecoder = mockService.GetTimelineDecoder();
    profiling::DirectoryCaptureCommandHandler& directoryCaptureCommandHandler =
         mockService.GetDirectoryCaptureCommandHandler();

    // Give the profiling service sending thread time start executing and send the stream metadata.
    WaitFor([&](){return profilingService.GetCurrentState() == profiling::ProfilingState::WaitingForAck;},
            "Profiling service did not switch to WaitingForAck state");

    profilingService.Update();
    // Read the stream metadata on the mock side.
    if (!mockService.WaitForStreamMetaData())
    {
        BOOST_FAIL("Failed to receive StreamMetaData");
    }
    // Send Ack from GatorD
    mockService.SendConnectionAck();
    // And start to listen for packets
    mockService.LaunchReceivingThread();

    WaitFor([&](){return profilingService.GetCurrentState() == profiling::ProfilingState::Active;},
            "Profiling service did not switch to Active state");

    // As part of the default startup of the profiling service a counter directory packet will be sent.
    WaitFor([&](){return directoryCaptureCommandHandler.ParsedCounterDirectory();},
            "MockGatord did not receive counter directory packet");

    // Following that we will receive a collection of well known timeline labels and event classes
    WaitFor([&](){return timelineDecoder.GetModel().m_EventClasses.size() >= 2;},
            "MockGatord did not receive well known timeline labels and event classes");

    CheckTimelineDirectory(mockService.GetTimelineDirectoryCaptureCommandHandler());
    // Verify the commonly used timeline packets sent when the profiling service enters the active state
    CheckTimelinePackets(timelineDecoder);

    const profiling::ICounterDirectory& serviceCounterDirectory  = profilingService.GetCounterDirectory();
    const profiling::ICounterDirectory& receivedCounterDirectory = directoryCaptureCommandHandler.GetCounterDirectory();

    // Compare the basics of the counter directory from the service and the one we received over the wire.
    BOOST_CHECK(serviceCounterDirectory.GetDeviceCount() == receivedCounterDirectory.GetDeviceCount());
    BOOST_CHECK(serviceCounterDirectory.GetCounterSetCount() == receivedCounterDirectory.GetCounterSetCount());
    BOOST_CHECK(serviceCounterDirectory.GetCategoryCount() == receivedCounterDirectory.GetCategoryCount());
    BOOST_CHECK(serviceCounterDirectory.GetCounterCount() == receivedCounterDirectory.GetCounterCount());

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
    armnnUtils::Sockets::Close(listeningSocket);
    // Future tests here will add counters to the ProfilingService, increment values and examine
    // PeriodicCounterCapture data received. These are yet to be integrated.
}

BOOST_AUTO_TEST_CASE(GatorDMockTimeLineActivation)
{
    armnn::MockBackendInitialiser initialiser;
    // Setup the mock service to bind to the UDS.
    std::string udsNamespace = "gatord_namespace";

    armnnUtils::Sockets::Initialize();
    armnnUtils::Sockets::Socket listeningSocket = socket(PF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);

    if (!gatordmock::GatordMockService::OpenListeningSocket(listeningSocket, udsNamespace))
    {
        BOOST_FAIL("Failed to open Listening Socket");
    }

    armnn::IRuntime::CreationOptions options;
    options.m_ProfilingOptions.m_EnableProfiling = true;
    options.m_ProfilingOptions.m_TimelineEnabled = true;
    armnn::Runtime runtime(options);

    armnnUtils::Sockets::Socket clientConnection;
    clientConnection = armnnUtils::Sockets::Accept(listeningSocket, nullptr, nullptr, SOCK_CLOEXEC);
    gatordmock::GatordMockService mockService(clientConnection, false);

    // Read the stream metadata on the mock side.
    if (!mockService.WaitForStreamMetaData())
    {
        BOOST_FAIL("Failed to receive StreamMetaData");
    }

    armnn::MockBackendProfilingService mockProfilingService = armnn::MockBackendProfilingService::Instance();
    armnn::MockBackendProfilingContext *mockBackEndProfilingContext = mockProfilingService.GetContext();

    // Send Ack from GatorD
    mockService.SendConnectionAck();
    // And start to listen for packets
    mockService.LaunchReceivingThread();

    // Build and optimize a simple network while we wait
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0, "input");

    NormalizationDescriptor descriptor;
    IConnectableLayer* normalize = net->AddNormalizationLayer(descriptor, "normalization");

    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input->GetOutputSlot(0).Connect(normalize->GetInputSlot(0));
    normalize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));
    normalize->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime.GetDeviceSpec());

    WaitFor([&](){return mockService.GetDirectoryCaptureCommandHandler().ParsedCounterDirectory();},
            "MockGatord did not receive counter directory packet");

    timelinedecoder::TimelineDecoder& timelineDecoder = mockService.GetTimelineDecoder();

    WaitFor([&](){return timelineDecoder.GetModel().m_EventClasses.size() >= 2;},
            "MockGatord did not receive well known timeline labels");

    WaitFor([&](){return timelineDecoder.GetModel().m_Entities.size() >= 1;},
            "MockGatord did not receive mock backend test entity");

    // Packets we expect from SendWellKnownLabelsAndEventClassesTest
    BOOST_CHECK(timelineDecoder.GetModel().m_Entities.size() == 1);
    BOOST_CHECK(timelineDecoder.GetModel().m_EventClasses.size()  == 2);
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels.size()  == 10);
    BOOST_CHECK(timelineDecoder.GetModel().m_Relationships.size()  == 0);
    BOOST_CHECK(timelineDecoder.GetModel().m_Events.size()  == 0);

    mockService.SendDeactivateTimelinePacket();

    WaitFor([&](){return !mockBackEndProfilingContext->TimelineReportingEnabled();},
            "Timeline packets were not deactivated");

    // Load the network into runtime now that timeline reporting is disabled
    armnn::NetworkId netId;
    runtime.LoadNetwork(netId, std::move(optNet));

    // Now activate timeline packets
    mockService.SendActivateTimelinePacket();

    WaitFor([&](){return mockBackEndProfilingContext->TimelineReportingEnabled();},
            "Timeline packets were not activated");

    // Once timeline packets have been reactivated the ActivateTimelineReportingCommandHandler will resend the
    // SendWellKnownLabelsAndEventClasses and then send the structure of any loaded networks
    WaitFor([&](){return timelineDecoder.GetModel().m_Labels.size() >= 24;},
            "MockGatord did not receive well known timeline labels");

    // Packets we expect from SendWellKnownLabelsAndEventClassesTest * 2 and the loaded model
    BOOST_CHECK(timelineDecoder.GetModel().m_Entities.size() == 6);
    BOOST_CHECK(timelineDecoder.GetModel().m_EventClasses.size()  == 4);
    BOOST_CHECK(timelineDecoder.GetModel().m_Labels.size()  == 24);
    BOOST_CHECK(timelineDecoder.GetModel().m_Relationships.size()  == 28);
    BOOST_CHECK(timelineDecoder.GetModel().m_Events.size()  == 0);

    mockService.WaitForReceivingThread();
    armnnUtils::Sockets::Close(listeningSocket);

    GetProfilingService(&runtime).Disconnect();
}

BOOST_AUTO_TEST_SUITE_END()
