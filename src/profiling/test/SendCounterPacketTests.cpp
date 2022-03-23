//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingMocks.hpp"
#include "ProfilingTestUtils.hpp"
#include "SendCounterPacketTests.hpp"

#include <client/src/BufferManager.hpp>
#include <client/src/ProfilingUtils.hpp>
#include <client/src/SendCounterPacket.hpp>

#include <armnn/Utils.hpp>

#include <common/include/Assert.hpp>
#include <common/include/Conversion.hpp>
#include <common/include/Constants.hpp>
#include <common/include/CounterDirectory.hpp>
#include <common/include/EncodeVersion.hpp>
#include <common/include/NumericCast.hpp>
#include <common/include/Processes.hpp>
#include <common/include/ProfilingException.hpp>

#include <doctest/doctest.h>

#include <chrono>

using namespace arm::pipe;

namespace
{

// A short delay to wait for the thread to process a packet.
uint16_t constexpr WAIT_UNTIL_READABLE_MS = 20;

void SetNotConnectedProfilingState(ProfilingStateMachine& profilingStateMachine)
{
    ProfilingState currentState = profilingStateMachine.GetCurrentState();
    switch (currentState)
    {
    case ProfilingState::WaitingForAck:
        profilingStateMachine.TransitionToState(ProfilingState::Active);
        ARMNN_FALLTHROUGH;
    case ProfilingState::Uninitialised:
        ARMNN_FALLTHROUGH;
    case ProfilingState::Active:
        profilingStateMachine.TransitionToState(ProfilingState::NotConnected);
        ARMNN_FALLTHROUGH;
    case ProfilingState::NotConnected:
        return;
    default:
        CHECK_MESSAGE(false, "Invalid profiling state");
    }
}

void SetWaitingForAckProfilingState(ProfilingStateMachine& profilingStateMachine)
{
    ProfilingState currentState = profilingStateMachine.GetCurrentState();
    switch (currentState)
    {
    case ProfilingState::Uninitialised:
        ARMNN_FALLTHROUGH;
    case ProfilingState::Active:
        profilingStateMachine.TransitionToState(ProfilingState::NotConnected);
        ARMNN_FALLTHROUGH;
    case ProfilingState::NotConnected:
        profilingStateMachine.TransitionToState(ProfilingState::WaitingForAck);
        ARMNN_FALLTHROUGH;
    case ProfilingState::WaitingForAck:
        return;
    default:
        CHECK_MESSAGE(false, "Invalid profiling state");
    }
}

void SetActiveProfilingState(ProfilingStateMachine& profilingStateMachine)
{
    ProfilingState currentState = profilingStateMachine.GetCurrentState();
    switch (currentState)
    {
    case ProfilingState::Uninitialised:
        profilingStateMachine.TransitionToState(ProfilingState::NotConnected);
        ARMNN_FALLTHROUGH;
    case ProfilingState::NotConnected:
        profilingStateMachine.TransitionToState(ProfilingState::WaitingForAck);
        ARMNN_FALLTHROUGH;
    case ProfilingState::WaitingForAck:
        profilingStateMachine.TransitionToState(ProfilingState::Active);
        ARMNN_FALLTHROUGH;
    case ProfilingState::Active:
        return;
    default:
        CHECK_MESSAGE(false, "Invalid profiling state");
    }
}

} // Anonymous namespace

TEST_SUITE("SendCounterPacketTests")
{
using PacketType = MockProfilingConnection::PacketType;

TEST_CASE("MockSendCounterPacketTest")
{
    MockBufferManager mockBuffer(512);
    MockSendCounterPacket mockSendCounterPacket(mockBuffer);

    mockSendCounterPacket.SendStreamMetaDataPacket();

    auto packetBuffer = mockBuffer.GetReadableBuffer();
    const char* buffer = reinterpret_cast<const char*>(packetBuffer->GetReadableData());

    CHECK(strcmp(buffer, "SendStreamMetaDataPacket") == 0);

    mockBuffer.MarkRead(packetBuffer);

    CounterDirectory counterDirectory;
    mockSendCounterPacket.SendCounterDirectoryPacket(counterDirectory);

    packetBuffer = mockBuffer.GetReadableBuffer();
    buffer = reinterpret_cast<const char*>(packetBuffer->GetReadableData());

    CHECK(strcmp(buffer, "SendCounterDirectoryPacket") == 0);

    mockBuffer.MarkRead(packetBuffer);

    uint64_t timestamp = 0;
    std::vector<CounterValue> indexValuePairs;

    mockSendCounterPacket.SendPeriodicCounterCapturePacket(timestamp, indexValuePairs);

    packetBuffer = mockBuffer.GetReadableBuffer();
    buffer = reinterpret_cast<const char*>(packetBuffer->GetReadableData());

    CHECK(strcmp(buffer, "SendPeriodicCounterCapturePacket") == 0);

    mockBuffer.MarkRead(packetBuffer);

    uint32_t capturePeriod = 0;
    std::vector<uint16_t> selectedCounterIds;
    mockSendCounterPacket.SendPeriodicCounterSelectionPacket(capturePeriod, selectedCounterIds);

    packetBuffer = mockBuffer.GetReadableBuffer();
    buffer = reinterpret_cast<const char*>(packetBuffer->GetReadableData());

    CHECK(strcmp(buffer, "SendPeriodicCounterSelectionPacket") == 0);

    mockBuffer.MarkRead(packetBuffer);
}

TEST_CASE("SendPeriodicCounterSelectionPacketTest")
{
    // Error no space left in buffer
    MockBufferManager mockBuffer1(10);
    SendCounterPacket sendPacket1(mockBuffer1,
                                  arm::pipe::ARMNN_SOFTWARE_INFO,
                                  arm::pipe::ARMNN_SOFTWARE_VERSION,
                                  arm::pipe::ARMNN_HARDWARE_VERSION);

    uint32_t capturePeriod = 1000;
    std::vector<uint16_t> selectedCounterIds;
    CHECK_THROWS_AS(sendPacket1.SendPeriodicCounterSelectionPacket(
                        capturePeriod, selectedCounterIds),
                        BufferExhaustion);

    // Packet without any counters
    MockBufferManager mockBuffer2(512);
    SendCounterPacket sendPacket2(mockBuffer2,
                                  arm::pipe::ARMNN_SOFTWARE_INFO,
                                  arm::pipe::ARMNN_SOFTWARE_VERSION,
                                  arm::pipe::ARMNN_HARDWARE_VERSION);

    sendPacket2.SendPeriodicCounterSelectionPacket(capturePeriod, selectedCounterIds);
    auto readBuffer2 = mockBuffer2.GetReadableBuffer();

    uint32_t headerWord0 = ReadUint32(readBuffer2, 0);
    uint32_t headerWord1 = ReadUint32(readBuffer2, 4);
    uint32_t period = ReadUint32(readBuffer2, 8);

    CHECK(((headerWord0 >> 26) & 0x3F) == 0);  // packet family
    CHECK(((headerWord0 >> 16) & 0x3FF) == 4); // packet id
    CHECK(headerWord1 == 4);                   // data lenght
    CHECK(period == 1000);                     // capture period

    // Full packet message
    MockBufferManager mockBuffer3(512);
    SendCounterPacket sendPacket3(mockBuffer3,
                                  arm::pipe::ARMNN_SOFTWARE_INFO,
                                  arm::pipe::ARMNN_SOFTWARE_VERSION,
                                  arm::pipe::ARMNN_HARDWARE_VERSION);

    selectedCounterIds.reserve(5);
    selectedCounterIds.emplace_back(100);
    selectedCounterIds.emplace_back(200);
    selectedCounterIds.emplace_back(300);
    selectedCounterIds.emplace_back(400);
    selectedCounterIds.emplace_back(500);
    sendPacket3.SendPeriodicCounterSelectionPacket(capturePeriod, selectedCounterIds);
    auto readBuffer3 = mockBuffer3.GetReadableBuffer();

    headerWord0 = ReadUint32(readBuffer3, 0);
    headerWord1 = ReadUint32(readBuffer3, 4);
    period = ReadUint32(readBuffer3, 8);

    CHECK(((headerWord0 >> 26) & 0x3F) == 0);  // packet family
    CHECK(((headerWord0 >> 16) & 0x3FF) == 4); // packet id
    CHECK(headerWord1 == 14);                  // data lenght
    CHECK(period == 1000);                     // capture period

    uint16_t counterId = 0;
    uint32_t offset = 12;

    // Counter Ids
    for(const uint16_t& id : selectedCounterIds)
    {
        counterId = ReadUint16(readBuffer3, offset);
        CHECK(counterId == id);
        offset += 2;
    }
}

TEST_CASE("SendPeriodicCounterCapturePacketTest")
{
    ProfilingStateMachine profilingStateMachine;

    // Error no space left in buffer
    MockBufferManager mockBuffer1(10);
    SendCounterPacket sendPacket1(mockBuffer1,
                                  arm::pipe::ARMNN_SOFTWARE_INFO,
                                  arm::pipe::ARMNN_SOFTWARE_VERSION,
                                  arm::pipe::ARMNN_HARDWARE_VERSION);

    auto captureTimestamp = std::chrono::steady_clock::now();
    uint64_t time =  static_cast<uint64_t >(captureTimestamp.time_since_epoch().count());
    std::vector<CounterValue> indexValuePairs;

    CHECK_THROWS_AS(sendPacket1.SendPeriodicCounterCapturePacket(time, indexValuePairs),
                      BufferExhaustion);

    // Packet without any counters
    MockBufferManager mockBuffer2(512);
    SendCounterPacket sendPacket2(mockBuffer2,
                                  arm::pipe::ARMNN_SOFTWARE_INFO,
                                  arm::pipe::ARMNN_SOFTWARE_VERSION,
                                  arm::pipe::ARMNN_HARDWARE_VERSION);

    sendPacket2.SendPeriodicCounterCapturePacket(time, indexValuePairs);
    auto readBuffer2 = mockBuffer2.GetReadableBuffer();

    uint32_t headerWord0 = ReadUint32(readBuffer2, 0);
    uint32_t headerWord1 = ReadUint32(readBuffer2, 4);
    uint64_t readTimestamp = ReadUint64(readBuffer2, 8);

    CHECK(((headerWord0 >> 26) & 0x0000003F) == 3); // packet family
    CHECK(((headerWord0 >> 19) & 0x0000007F) == 0); // packet class
    CHECK(((headerWord0 >> 16) & 0x00000007) == 0); // packet type
    CHECK(headerWord1 == 8);                        // data length
    CHECK(time == readTimestamp);                   // capture period

    // Full packet message
    MockBufferManager mockBuffer3(512);
    SendCounterPacket sendPacket3(mockBuffer3,
                                  arm::pipe::ARMNN_SOFTWARE_INFO,
                                  arm::pipe::ARMNN_SOFTWARE_VERSION,
                                  arm::pipe::ARMNN_HARDWARE_VERSION);

    indexValuePairs.reserve(5);
    indexValuePairs.emplace_back(CounterValue{0, 100});
    indexValuePairs.emplace_back(CounterValue{1, 200});
    indexValuePairs.emplace_back(CounterValue{2, 300});
    indexValuePairs.emplace_back(CounterValue{3, 400});
    indexValuePairs.emplace_back(CounterValue{4, 500});
    sendPacket3.SendPeriodicCounterCapturePacket(time, indexValuePairs);
    auto readBuffer3 = mockBuffer3.GetReadableBuffer();

    headerWord0 = ReadUint32(readBuffer3, 0);
    headerWord1 = ReadUint32(readBuffer3, 4);
    uint64_t readTimestamp2 = ReadUint64(readBuffer3, 8);

    CHECK(((headerWord0 >> 26) & 0x0000003F) == 3); // packet family
    CHECK(((headerWord0 >> 19) & 0x0000007F) == 0); // packet class
    CHECK(((headerWord0 >> 16) & 0x00000007) == 0); // packet type
    CHECK(headerWord1 == 38);                       // data length
    CHECK(time == readTimestamp2);                  // capture period

    uint16_t counterIndex = 0;
    uint32_t counterValue = 100;
    uint32_t offset = 16;

    // Counter Ids
    for (auto it = indexValuePairs.begin(), end = indexValuePairs.end(); it != end; ++it)
    {
        // Check Counter Index
        uint16_t readIndex = ReadUint16(readBuffer3, offset);
        CHECK(counterIndex == readIndex);
        counterIndex++;
        offset += 2;

        // Check Counter Value
        uint32_t readValue = ReadUint32(readBuffer3, offset);
        CHECK(counterValue == readValue);
        counterValue += 100;
        offset += 4;
    }

}

TEST_CASE("SendStreamMetaDataPacketTest")
{
    uint32_t sizeUint32 = arm::pipe::numeric_cast<uint32_t>(sizeof(uint32_t));

    // Error no space left in buffer
    MockBufferManager mockBuffer1(10);
    SendCounterPacket sendPacket1(mockBuffer1,
                                  arm::pipe::ARMNN_SOFTWARE_INFO,
                                  arm::pipe::ARMNN_SOFTWARE_VERSION,
                                  arm::pipe::ARMNN_HARDWARE_VERSION);
    CHECK_THROWS_AS(sendPacket1.SendStreamMetaDataPacket(), BufferExhaustion);

    // Full metadata packet

    std::string processName = GetProcessName().substr(0, 60);

    uint32_t infoSize =            arm::pipe::numeric_cast<uint32_t>(arm::pipe::ARMNN_SOFTWARE_INFO.size()) + 1;
    uint32_t hardwareVersionSize = arm::pipe::numeric_cast<uint32_t>(arm::pipe::ARMNN_HARDWARE_VERSION.size()) + 1;
    uint32_t softwareVersionSize = arm::pipe::numeric_cast<uint32_t>(arm::pipe::ARMNN_SOFTWARE_VERSION.size()) + 1;
    uint32_t processNameSize =     arm::pipe::numeric_cast<uint32_t>(processName.size()) + 1;

    // Supported Packets
    // Packet Encoding version 1.0.0
    // Control packet family
    //   Stream metadata packet (packet family=0; packet id=0)
    //   Connection Acknowledged packet ( packet family=0, packet id=1) Version 1.0.0
    //   Counter Directory packet (packet family=0; packet id=2) Version 1.0.0
    //   Request Counter Directory packet ( packet family=0, packet id=3) Version 1.0.0
    //   Periodic Counter Selection packet ( packet family=0, packet id=4) Version 1.0.0
    //   Per Job Counter Selection packet ( packet family=0, packet id=5) Version 1.0.0
    //   Activate Timeline Reporting (packet family = 0, packet id = 6) Version 1.0.0
    //   Deactivate Timeline Reporting (packet family = 0, packet id = 7) Version 1.0.0
    // Counter Packet Family
    //   Periodic Counter Capture (packet_family = 3, packet_class = 0, packet_type = 0) Version 1.0.0
    //   Per-Job Counter Capture (packet_family = 3, packet_class = 1, packet_type = 0,1) Version  1.0.0
    // Timeline Packet Family
    //   Timeline Message Directory (packet_family = 1, packet_class = 0, packet_type = 0) Version 1.0.0
    //   Timeline Message (packet_family = 1, packet_class = 0, packet_type = 1) Version 1.0.0
    std::vector<std::pair<uint32_t, uint32_t>> packetVersions;
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 0), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 1), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 2), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 3), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 4), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 5), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 6), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 7), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(3, 0, 0), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(3, 1, 0), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(3, 1, 1), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(1, 0, 0), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(1, 0, 1), arm::pipe::EncodeVersion(1, 0, 0)));

    uint32_t packetEntries = static_cast<uint32_t>(packetVersions.size());

    MockBufferManager mockBuffer2(512);
    SendCounterPacket sendPacket2(mockBuffer2,
                                  arm::pipe::ARMNN_SOFTWARE_INFO,
                                  arm::pipe::ARMNN_SOFTWARE_VERSION,
                                  arm::pipe::ARMNN_HARDWARE_VERSION);
    sendPacket2.SendStreamMetaDataPacket();
    auto readBuffer2 = mockBuffer2.GetReadableBuffer();

    uint32_t headerWord0 = ReadUint32(readBuffer2, 0);
    uint32_t headerWord1 = ReadUint32(readBuffer2, sizeUint32);

    CHECK(((headerWord0 >> 26) & 0x3F) == 0); // packet family
    CHECK(((headerWord0 >> 16) & 0x3FF) == 0); // packet id

    uint32_t totalLength = arm::pipe::numeric_cast<uint32_t>(2 * sizeUint32 +
                                                         10 * sizeUint32 + infoSize +
                                                         hardwareVersionSize + softwareVersionSize +
                                                         processNameSize + sizeUint32 +
                                                         2 * packetEntries * sizeUint32);

    CHECK(headerWord1 == totalLength - (2 * sizeUint32)); // data length

    uint32_t offset = sizeUint32 * 2;
    CHECK(ReadUint32(readBuffer2, offset) == arm::pipe::PIPE_MAGIC); // pipe_magic
    offset += sizeUint32;
    CHECK(ReadUint32(readBuffer2, offset) == arm::pipe::EncodeVersion(1, 0, 0)); // stream_metadata_version
    offset += sizeUint32;
    CHECK(ReadUint32(readBuffer2, offset) == MAX_METADATA_PACKET_LENGTH); // max_data_len
    offset += sizeUint32;
    int pid = arm::pipe::GetCurrentProcessId();
    CHECK(ReadUint32(readBuffer2, offset) == arm::pipe::numeric_cast<uint32_t>(pid));
    offset += sizeUint32;
    uint32_t poolOffset = 10 * sizeUint32;
    CHECK(ReadUint32(readBuffer2, offset) == poolOffset); // offset_info
    offset += sizeUint32;
    poolOffset += infoSize;
    CHECK(ReadUint32(readBuffer2, offset) == poolOffset); // offset_hw_version
    offset += sizeUint32;
    poolOffset += hardwareVersionSize;
    CHECK(ReadUint32(readBuffer2, offset) == poolOffset); // offset_sw_version
    offset += sizeUint32;
    poolOffset += softwareVersionSize;
    CHECK(ReadUint32(readBuffer2, offset) == poolOffset); // offset_process_name
    offset += sizeUint32;
    poolOffset += processNameSize;
    CHECK(ReadUint32(readBuffer2, offset) == poolOffset); // offset_packet_version_table
    offset += sizeUint32;
    CHECK(ReadUint32(readBuffer2, offset) == 0); // reserved

    const unsigned char* readData2 = readBuffer2->GetReadableData();

    offset += sizeUint32;
    if (infoSize)
    {
        CHECK(strcmp(reinterpret_cast<const char *>(&readData2[offset]),
                                                    arm::pipe::ARMNN_SOFTWARE_INFO.c_str()) == 0);
        offset += infoSize;
    }

    if (hardwareVersionSize)
    {
        CHECK(strcmp(reinterpret_cast<const char *>(&readData2[offset]),
                                                    arm::pipe::ARMNN_HARDWARE_VERSION.c_str()) == 0);
        offset += hardwareVersionSize;
    }

    if (softwareVersionSize)
    {
        CHECK(strcmp(reinterpret_cast<const char *>(&readData2[offset]),
                                                    arm::pipe::ARMNN_SOFTWARE_VERSION.c_str()) == 0);
        offset += softwareVersionSize;
    }

    if (processNameSize)
    {
        CHECK(strcmp(reinterpret_cast<const char *>(&readData2[offset]), GetProcessName().c_str()) == 0);
        offset += processNameSize;
    }

    if (packetEntries)
    {
        uint32_t numberOfEntries = ReadUint32(readBuffer2, offset);
        CHECK((numberOfEntries >> 16) == packetEntries);
        offset += sizeUint32;
        for (std::pair<uint32_t, uint32_t>& packetVersion : packetVersions)
        {
            uint32_t readPacketId = ReadUint32(readBuffer2, offset);
            CHECK(packetVersion.first == readPacketId);
            offset += sizeUint32;
            uint32_t readVersion = ReadUint32(readBuffer2, offset);
            CHECK(packetVersion.second == readVersion);
            offset += sizeUint32;
        }
    }

    CHECK(offset == totalLength);
}

TEST_CASE("CreateDeviceRecordTest")
{
    MockBufferManager mockBuffer(0);
    SendCounterPacketTest sendCounterPacketTest(mockBuffer);

    // Create a device for testing
    uint16_t deviceUid = 27;
    const std::string deviceName = "some_device";
    uint16_t deviceCores = 3;
    const DevicePtr device = std::make_unique<Device>(deviceUid, deviceName, deviceCores);

    // Create a device record
    SendCounterPacket::DeviceRecord deviceRecord;
    std::string errorMessage;
    bool result = sendCounterPacketTest.CreateDeviceRecordTest(device, deviceRecord, errorMessage);

    CHECK(result);
    CHECK(errorMessage.empty());
    CHECK(deviceRecord.size() == 6); // Size in words: header [2] + device name [4]

    uint16_t deviceRecordWord0[]
    {
        static_cast<uint16_t>(deviceRecord[0] >> 16),
        static_cast<uint16_t>(deviceRecord[0])
    };
    CHECK(deviceRecordWord0[0] == deviceUid); // uid
    CHECK(deviceRecordWord0[1] == deviceCores); // cores
    CHECK(deviceRecord[1] == 8); // name_offset
    CHECK(deviceRecord[2] == deviceName.size() + 1); // The length of the SWTrace string (name)
    CHECK(std::memcmp(deviceRecord.data() + 3, deviceName.data(), deviceName.size()) == 0); // name
}

TEST_CASE("CreateInvalidDeviceRecordTest")
{
    MockBufferManager mockBuffer(0);
    SendCounterPacketTest sendCounterPacketTest(mockBuffer);

    // Create a device for testing
    uint16_t deviceUid = 27;
    const std::string deviceName = "some€£invalid‡device";
    uint16_t deviceCores = 3;
    const DevicePtr device = std::make_unique<Device>(deviceUid, deviceName, deviceCores);

    // Create a device record
    SendCounterPacket::DeviceRecord deviceRecord;
    std::string errorMessage;
    bool result = sendCounterPacketTest.CreateDeviceRecordTest(device, deviceRecord, errorMessage);

    CHECK(!result);
    CHECK(!errorMessage.empty());
    CHECK(deviceRecord.empty());
}

TEST_CASE("CreateCounterSetRecordTest")
{
    MockBufferManager mockBuffer(0);
    SendCounterPacketTest sendCounterPacketTest(mockBuffer);

    // Create a counter set for testing
    uint16_t counterSetUid = 27;
    const std::string counterSetName = "some_counter_set";
    uint16_t counterSetCount = 3421;
    const CounterSetPtr counterSet = std::make_unique<CounterSet>(counterSetUid, counterSetName, counterSetCount);

    // Create a counter set record
    SendCounterPacket::CounterSetRecord counterSetRecord;
    std::string errorMessage;
    bool result = sendCounterPacketTest.CreateCounterSetRecordTest(counterSet, counterSetRecord, errorMessage);

    CHECK(result);
    CHECK(errorMessage.empty());
    CHECK(counterSetRecord.size() == 8); // Size in words: header [2] + counter set name [6]

    uint16_t counterSetRecordWord0[]
    {
        static_cast<uint16_t>(counterSetRecord[0] >> 16),
        static_cast<uint16_t>(counterSetRecord[0])
    };
    CHECK(counterSetRecordWord0[0] == counterSetUid); // uid
    CHECK(counterSetRecordWord0[1] == counterSetCount); // cores
    CHECK(counterSetRecord[1] == 8); // name_offset
    CHECK(counterSetRecord[2] == counterSetName.size() + 1); // The length of the SWTrace string (name)
    CHECK(std::memcmp(counterSetRecord.data() + 3, counterSetName.data(), counterSetName.size()) == 0); // name
}

TEST_CASE("CreateInvalidCounterSetRecordTest")
{
    MockBufferManager mockBuffer(0);
    SendCounterPacketTest sendCounterPacketTest(mockBuffer);

    // Create a counter set for testing
    uint16_t counterSetUid = 27;
    const std::string counterSetName = "some invalid_counter€£set";
    uint16_t counterSetCount = 3421;
    const CounterSetPtr counterSet = std::make_unique<CounterSet>(counterSetUid, counterSetName, counterSetCount);

    // Create a counter set record
    SendCounterPacket::CounterSetRecord counterSetRecord;
    std::string errorMessage;
    bool result = sendCounterPacketTest.CreateCounterSetRecordTest(counterSet, counterSetRecord, errorMessage);

    CHECK(!result);
    CHECK(!errorMessage.empty());
    CHECK(counterSetRecord.empty());
}

TEST_CASE("CreateEventRecordTest")
{
    MockBufferManager mockBuffer(0);
    SendCounterPacketTest sendCounterPacketTest(mockBuffer);

    // Create a counter for testing
    uint16_t counterUid = 7256;
    uint16_t maxCounterUid = 132;
    uint16_t deviceUid = 132;
    uint16_t counterSetUid = 4497;
    uint16_t counterClass = 1;
    uint16_t counterInterpolation = 1;
    double counterMultiplier = 1234.567f;
    const std::string counterName = "some_valid_counter";
    const std::string counterDescription = "a_counter_for_testing";
    const std::string counterUnits = "Mrads2";
    const CounterPtr counter = std::make_unique<Counter>(armnn::profiling::BACKEND_ID,
                                                         counterUid,
                                                         maxCounterUid,
                                                         counterClass,
                                                         counterInterpolation,
                                                         counterMultiplier,
                                                         counterName,
                                                         counterDescription,
                                                         counterUnits,
                                                         deviceUid,
                                                         counterSetUid);
    ARM_PIPE_ASSERT(counter);

    // Create an event record
    SendCounterPacket::EventRecord eventRecord;
    std::string errorMessage;
    bool result = sendCounterPacketTest.CreateEventRecordTest(counter, eventRecord, errorMessage);

    CHECK(result);
    CHECK(errorMessage.empty());
    CHECK(eventRecord.size() == 24); // Size in words: header [8] + counter name [6] + description [7] + units [3]

    uint16_t eventRecordWord0[]
    {
        static_cast<uint16_t>(eventRecord[0] >> 16),
        static_cast<uint16_t>(eventRecord[0])
    };
    uint16_t eventRecordWord1[]
    {
        static_cast<uint16_t>(eventRecord[1] >> 16),
        static_cast<uint16_t>(eventRecord[1])
    };
    uint16_t eventRecordWord2[]
    {
        static_cast<uint16_t>(eventRecord[2] >> 16),
        static_cast<uint16_t>(eventRecord[2])
    };
    uint32_t eventRecordWord34[]
    {
        eventRecord[3],
        eventRecord[4]
    };

    CHECK(eventRecordWord0[0] == maxCounterUid); // max_counter_uid
    CHECK(eventRecordWord0[1] == counterUid); // counter_uid
    CHECK(eventRecordWord1[0] == deviceUid); // device

    CHECK(eventRecordWord1[1] == counterSetUid); // counter_set
    CHECK(eventRecordWord2[0] == counterClass); // class
    CHECK(eventRecordWord2[1] == counterInterpolation); // interpolation
    CHECK(std::memcmp(eventRecordWord34, &counterMultiplier, sizeof(counterMultiplier)) == 0); // multiplier

    ARM_PIPE_NO_CONVERSION_WARN_BEGIN
    uint32_t eventRecordBlockSize = 8u * sizeof(uint32_t);
    uint32_t counterNameOffset = eventRecordBlockSize; // The name is the first item in pool
    uint32_t counterDescriptionOffset = counterNameOffset + // Counter name offset
                                        4u + // Counter name length (uint32_t)
                                        counterName.size() + // 18u
                                        1u + // Null-terminator
                                        1u; // Rounding to the next word

    size_t counterUnitsOffset = counterDescriptionOffset + // Counter description offset
                                4u + // Counter description length (uint32_t)
                                counterDescription.size() + // 21u
                                1u + // Null-terminator
                                2u;  // Rounding to the next word

    ARM_PIPE_NO_CONVERSION_WARN_END

    CHECK(eventRecord[5] == counterNameOffset); // name_offset
    CHECK(eventRecord[6] == counterDescriptionOffset); // description_offset
    CHECK(eventRecord[7] == counterUnitsOffset); // units_offset

    // Offsets are relative to the start of the eventRecord
    auto eventRecordPool = reinterpret_cast<unsigned char*>(eventRecord.data());
    size_t uint32_t_size = sizeof(uint32_t);

    // The length of the SWTrace string (name)
    CHECK(eventRecordPool[counterNameOffset] == counterName.size() + 1);
    // The counter name
    CHECK(std::memcmp(eventRecordPool +
                            counterNameOffset + // Offset
                            uint32_t_size /* The length of the name */,
                            counterName.data(),
                            counterName.size()) == 0); // name
    // The null-terminator at the end of the name
    CHECK(eventRecordPool[counterNameOffset + uint32_t_size + counterName.size()] == '\0');

    // The length of the SWTrace string (description)
    CHECK(eventRecordPool[counterDescriptionOffset] == counterDescription.size() + 1);
    // The counter description
    CHECK(std::memcmp(eventRecordPool +
                            counterDescriptionOffset + // Offset
                            uint32_t_size /* The length of the description */,
                            counterDescription.data(),
                            counterDescription.size()) == 0); // description
    // The null-terminator at the end of the description
    CHECK(eventRecordPool[counterDescriptionOffset + uint32_t_size + counterDescription.size()] == '\0');

    // The length of the SWTrace namestring (units)
    CHECK(eventRecordPool[counterUnitsOffset] == counterUnits.size() + 1);
    // The counter units
    CHECK(std::memcmp(eventRecordPool +
                            counterUnitsOffset + // Offset
                            uint32_t_size /* The length of the units */,
                            counterUnits.data(),
                            counterUnits.size()) == 0); // units
    // The null-terminator at the end of the units
    CHECK(eventRecordPool[counterUnitsOffset + uint32_t_size + counterUnits.size()] == '\0');
}

TEST_CASE("CreateEventRecordNoUnitsTest")
{
    MockBufferManager mockBuffer(0);
    SendCounterPacketTest sendCounterPacketTest(mockBuffer);

    // Create a counter for testing
    uint16_t counterUid = 44312;
    uint16_t maxCounterUid = 345;
    uint16_t deviceUid = 101;
    uint16_t counterSetUid = 34035;
    uint16_t counterClass = 0;
    uint16_t counterInterpolation = 1;
    double counterMultiplier = 4435.0023f;
    const std::string counterName = "some_valid_counter";
    const std::string counterDescription = "a_counter_for_testing";
    const CounterPtr counter = std::make_unique<Counter>(armnn::profiling::BACKEND_ID,
                                                         counterUid,
                                                         maxCounterUid,
                                                         counterClass,
                                                         counterInterpolation,
                                                         counterMultiplier,
                                                         counterName,
                                                         counterDescription,
                                                         "",
                                                         deviceUid,
                                                         counterSetUid);
    ARM_PIPE_ASSERT(counter);

    // Create an event record
    SendCounterPacket::EventRecord eventRecord;
    std::string errorMessage;
    bool result = sendCounterPacketTest.CreateEventRecordTest(counter, eventRecord, errorMessage);

    CHECK(result);
    CHECK(errorMessage.empty());
    CHECK(eventRecord.size() == 21); // Size in words: header [8] + counter name [6] + description [7]

    uint16_t eventRecordWord0[]
    {
        static_cast<uint16_t>(eventRecord[0] >> 16),
        static_cast<uint16_t>(eventRecord[0])
    };
    uint16_t eventRecordWord1[]
    {
        static_cast<uint16_t>(eventRecord[1] >> 16),
        static_cast<uint16_t>(eventRecord[1])
    };
    uint16_t eventRecordWord2[]
    {
        static_cast<uint16_t>(eventRecord[2] >> 16),
        static_cast<uint16_t>(eventRecord[2])
    };
    uint32_t eventRecordWord34[]
    {
        eventRecord[3],
        eventRecord[4]
    };
    CHECK(eventRecordWord0[0] == maxCounterUid); // max_counter_uid
    CHECK(eventRecordWord0[1] == counterUid); // counter_uid
    CHECK(eventRecordWord1[0] == deviceUid); // device
    CHECK(eventRecordWord1[1] == counterSetUid); // counter_set
    CHECK(eventRecordWord2[0] == counterClass); // class
    CHECK(eventRecordWord2[1] == counterInterpolation); // interpolation
    CHECK(std::memcmp(eventRecordWord34, &counterMultiplier, sizeof(counterMultiplier)) == 0); // multiplier

    ARM_PIPE_NO_CONVERSION_WARN_BEGIN
    uint32_t eventRecordBlockSize = 8u * sizeof(uint32_t);
    uint32_t counterNameOffset = eventRecordBlockSize; // The name is the first item in pool
    uint32_t counterDescriptionOffset = counterNameOffset + // Counter name offset
                                        4u + // Counter name length (uint32_t)
                                        counterName.size() + // 18u
                                        1u + // Null-terminator
                                        1u; // Rounding to the next word
    ARM_PIPE_NO_CONVERSION_WARN_END

    CHECK(eventRecord[5] == counterNameOffset); // name_offset
    CHECK(eventRecord[6] == counterDescriptionOffset); // description_offset
    CHECK(eventRecord[7] == 0); // units_offset

    // Offsets are relative to the start of the eventRecord
    auto eventRecordPool = reinterpret_cast<unsigned char*>(eventRecord.data());
    size_t uint32_t_size = sizeof(uint32_t);

    // The length of the SWTrace string (name)
    CHECK(eventRecordPool[counterNameOffset] == counterName.size() + 1);
    // The counter name
    CHECK(std::memcmp(eventRecordPool +
                            counterNameOffset + // Offset
                            uint32_t_size, // The length of the name
                            counterName.data(),
                            counterName.size()) == 0); // name
    // The null-terminator at the end of the name
    CHECK(eventRecordPool[counterNameOffset + uint32_t_size + counterName.size()] == '\0');

    // The length of the SWTrace string (description)
    CHECK(eventRecordPool[counterDescriptionOffset] == counterDescription.size() + 1);
    // The counter description
    CHECK(std::memcmp(eventRecordPool +
                            counterDescriptionOffset + // Offset
                            uint32_t_size, // The length of the description
                            counterDescription.data(),
                            counterDescription.size()) == 0); // description
    // The null-terminator at the end of the description
    CHECK(eventRecordPool[counterDescriptionOffset + uint32_t_size + counterDescription.size()] == '\0');
}

TEST_CASE("CreateInvalidEventRecordTest1")
{
    MockBufferManager mockBuffer(0);
    SendCounterPacketTest sendCounterPacketTest(mockBuffer);

    // Create a counter for testing
    uint16_t counterUid = 7256;
    uint16_t maxCounterUid = 132;
    uint16_t deviceUid = 132;
    uint16_t counterSetUid = 4497;
    uint16_t counterClass = 1;
    uint16_t counterInterpolation = 1;
    double counterMultiplier = 1234.567f;
    const std::string counterName = "some_invalid_counter £££"; // Invalid name
    const std::string counterDescription = "a_counter_for_testing";
    const std::string counterUnits = "Mrads2";
    const CounterPtr counter = std::make_unique<Counter>(armnn::profiling::BACKEND_ID,
                                                         counterUid,
                                                         maxCounterUid,
                                                         counterClass,
                                                         counterInterpolation,
                                                         counterMultiplier,
                                                         counterName,
                                                         counterDescription,
                                                         counterUnits,
                                                         deviceUid,
                                                         counterSetUid);
    ARM_PIPE_ASSERT(counter);

    // Create an event record
    SendCounterPacket::EventRecord eventRecord;
    std::string errorMessage;
    bool result = sendCounterPacketTest.CreateEventRecordTest(counter, eventRecord, errorMessage);

    CHECK(!result);
    CHECK(!errorMessage.empty());
    CHECK(eventRecord.empty());
}

TEST_CASE("CreateInvalidEventRecordTest2")
{
    MockBufferManager mockBuffer(0);
    SendCounterPacketTest sendCounterPacketTest(mockBuffer);

    // Create a counter for testing
    uint16_t counterUid = 7256;
    uint16_t maxCounterUid = 132;
    uint16_t deviceUid = 132;
    uint16_t counterSetUid = 4497;
    uint16_t counterClass = 1;
    uint16_t counterInterpolation = 1;
    double counterMultiplier = 1234.567f;
    const std::string counterName = "some_invalid_counter";
    const std::string counterDescription = "an invalid d€scription"; // Invalid description
    const std::string counterUnits = "Mrads2";
    const CounterPtr counter = std::make_unique<Counter>(armnn::profiling::BACKEND_ID,
                                                         counterUid,
                                                         maxCounterUid,
                                                         counterClass,
                                                         counterInterpolation,
                                                         counterMultiplier,
                                                         counterName,
                                                         counterDescription,
                                                         counterUnits,
                                                         deviceUid,
                                                         counterSetUid);
    ARM_PIPE_ASSERT(counter);

    // Create an event record
    SendCounterPacket::EventRecord eventRecord;
    std::string errorMessage;
    bool result = sendCounterPacketTest.CreateEventRecordTest(counter, eventRecord, errorMessage);

    CHECK(!result);
    CHECK(!errorMessage.empty());
    CHECK(eventRecord.empty());
}

TEST_CASE("CreateInvalidEventRecordTest3")
{
    MockBufferManager mockBuffer(0);
    SendCounterPacketTest sendCounterPacketTest(mockBuffer);

    // Create a counter for testing
    uint16_t counterUid = 7256;
    uint16_t maxCounterUid = 132;
    uint16_t deviceUid = 132;
    uint16_t counterSetUid = 4497;
    uint16_t counterClass = 1;
    uint16_t counterInterpolation = 1;
    double counterMultiplier = 1234.567f;
    const std::string counterName = "some_invalid_counter";
    const std::string counterDescription = "a valid description";
    const std::string counterUnits = "Mrad s2"; // Invalid units
    const CounterPtr counter = std::make_unique<Counter>(armnn::profiling::BACKEND_ID,
                                                         counterUid,
                                                         maxCounterUid,
                                                         counterClass,
                                                         counterInterpolation,
                                                         counterMultiplier,
                                                         counterName,
                                                         counterDescription,
                                                         counterUnits,
                                                         deviceUid,
                                                         counterSetUid);
    ARM_PIPE_ASSERT(counter);

    // Create an event record
    SendCounterPacket::EventRecord eventRecord;
    std::string errorMessage;
    bool result = sendCounterPacketTest.CreateEventRecordTest(counter, eventRecord, errorMessage);

    CHECK(!result);
    CHECK(!errorMessage.empty());
    CHECK(eventRecord.empty());
}

TEST_CASE("CreateCategoryRecordTest")
{
    MockBufferManager mockBuffer(0);
    SendCounterPacketTest sendCounterPacketTest(mockBuffer);

    // Create a category for testing
    const std::string categoryName = "some_category";
    const CategoryPtr category = std::make_unique<Category>(categoryName);
    ARM_PIPE_ASSERT(category);
    category->m_Counters = { 11u, 23u, 5670u };

    // Create a collection of counters
    Counters counters;
    counters.insert(std::make_pair<uint16_t, CounterPtr>(11,
                                                         CounterPtr(new Counter(armnn::profiling::BACKEND_ID,
                                                                                0,
                                                                                11,
                                                                                0,
                                                                                0,
                                                                                534.0003f,
                                                                                "counter1",
                                                                                "the first counter",
                                                                                "millipi2",
                                                                                0,
                                                                                0))));
    counters.insert(std::make_pair<uint16_t, CounterPtr>(23,
                                                         CounterPtr(new Counter(armnn::profiling::BACKEND_ID,
                                                                                1,
                                                                                23,
                                                                                0,
                                                                                1,
                                                                                534.0003f,
                                                                                "this is counter 2",
                                                                                "the second counter",
                                                                                "",
                                                                                0,
                                                                                0))));
    counters.insert(std::make_pair<uint16_t, CounterPtr>(5670,
                                                         CounterPtr(new Counter(armnn::profiling::BACKEND_ID,
                                                                                2,
                                                                                5670,
                                                                                0,
                                                                                0,
                                                                                534.0003f,
                                                                                "and this is number 3",
                                                                                "the third counter",
                                                                                "blah_per_second",
                                                                                0,
                                                                                0))));
    Counter* counter1 = counters.find(11)->second.get();
    Counter* counter2 = counters.find(23)->second.get();
    Counter* counter3 = counters.find(5670)->second.get();
    ARM_PIPE_ASSERT(counter1);
    ARM_PIPE_ASSERT(counter2);
    ARM_PIPE_ASSERT(counter3);
    uint16_t categoryEventCount = armnn::numeric_cast<uint16_t>(counters.size());

    // Create a category record
    SendCounterPacket::CategoryRecord categoryRecord;
    std::string errorMessage;
    bool result = sendCounterPacketTest.CreateCategoryRecordTest(category, counters, categoryRecord, errorMessage);

    CHECK(result);
    CHECK(errorMessage.empty());
    CHECK(categoryRecord.size() == 79); // Size in words: header [3] + event pointer table [3] +
                                              //                category name [5] + event records [68 = 22 + 20 + 26]

    uint16_t categoryRecordWord1[]
    {
        static_cast<uint16_t>(categoryRecord[0] >> 16),
        static_cast<uint16_t>(categoryRecord[0])
    };
    CHECK(categoryRecordWord1[0] == categoryEventCount); // event_count
    CHECK(categoryRecordWord1[1] == 0); // reserved

    size_t uint32_t_size = sizeof(uint32_t);

    ARM_PIPE_NO_CONVERSION_WARN_BEGIN
    uint32_t categoryRecordBlockSize = 3u * uint32_t_size;
    uint32_t eventPointerTableOffset = categoryRecordBlockSize; // The event pointer table is the first item in pool
    uint32_t categoryNameOffset = eventPointerTableOffset + // Event pointer table offset
                                  categoryEventCount * uint32_t_size; // The size of the event pointer table
    ARM_PIPE_NO_CONVERSION_WARN_END

    CHECK(categoryRecord[1] == eventPointerTableOffset); // event_pointer_table_offset
    CHECK(categoryRecord[2] == categoryNameOffset); // name_offset
    // Offsets are relative to the start of the category record
    auto categoryRecordPool = reinterpret_cast<unsigned char*>(categoryRecord.data());

    // The event pointer table
    uint32_t eventRecord0Offset = categoryRecordPool[eventPointerTableOffset + 0 * uint32_t_size];
    uint32_t eventRecord1Offset = categoryRecordPool[eventPointerTableOffset + 1 * uint32_t_size];
    uint32_t eventRecord2Offset = categoryRecordPool[eventPointerTableOffset + 2 * uint32_t_size];
    CHECK(eventRecord0Offset == 32);
    CHECK(eventRecord1Offset == 120);
    CHECK(eventRecord2Offset == 200);

    // The length of the SWTrace namestring (name)
    CHECK(categoryRecordPool[categoryNameOffset] == categoryName.size() + 1);
    // The category name
    CHECK(std::memcmp(categoryRecordPool +
                            categoryNameOffset + // Offset
                            uint32_t_size, // The length of the name
                            categoryName.data(),
                            categoryName.size()) == 0); // name
    // The null-terminator at the end of the name
    CHECK(categoryRecordPool[categoryNameOffset + uint32_t_size + categoryName.size()] == '\0');

    // For brevity, checking only the UIDs, max counter UIDs and names of the counters in the event records,
    // as the event records already have a number of unit tests dedicated to them

    // Counter1 UID and max counter UID
    uint16_t eventRecord0Word0[2] = { 0u, 0u };
    std::memcpy(eventRecord0Word0, categoryRecordPool + categoryRecordBlockSize + eventRecord0Offset,
                sizeof(eventRecord0Word0));
    CHECK(eventRecord0Word0[0] == counter1->m_Uid);
    CHECK(eventRecord0Word0[1] == counter1->m_MaxCounterUid);

    // Counter1 name
    uint32_t counter1NameOffset = 0;
    std::memcpy(&counter1NameOffset, categoryRecordPool  + eventRecord0Offset + 5u * uint32_t_size, uint32_t_size);
    CHECK(counter1NameOffset == 0);
    // The length of the SWTrace string (name)
    CHECK(categoryRecordPool[eventRecord0Offset +       // Offset to the event record
                                   categoryRecordBlockSize  + // Offset to the end of the category record block
                                   8u * uint32_t_size +       // Offset to the event record pool
                                   counter1NameOffset         // Offset to the name of the counter
                                  ] == counter1->m_Name.size() + 1); // The length of the name including the
                                                                     // null-terminator
    // The counter1 name
    CHECK(std::memcmp(categoryRecordPool +      // The beginning of the category pool
                            categoryRecordBlockSize + // Offset to the end of the category record block
                            eventRecord0Offset +      // Offset to the event record
                            8u * uint32_t_size +      // Offset to the event record pool
                            counter1NameOffset +      // Offset to the name of the counter
                            uint32_t_size,            // The length of the name
                            counter1->m_Name.data(),
                            counter1->m_Name.size()) == 0); // name
    // The null-terminator at the end of the counter1 name
    CHECK(categoryRecordPool[eventRecord0Offset +      // Offset to the event record
                                   categoryRecordBlockSize + // Offset to the end of the category record block
                                   8u * uint32_t_size +      // Offset to the event record pool
                                   counter1NameOffset +      // Offset to the name of the counter
                                   uint32_t_size +           // The length of the name
                                   counter1->m_Name.size()   // The name of the counter
                                   ] == '\0');

    // Counter2 name
    uint32_t counter2NameOffset = 0;
    std::memcpy(&counter2NameOffset, categoryRecordPool +
                                     categoryRecordBlockSize +
                                     eventRecord1Offset +
                                     5u * uint32_t_size,
                                     uint32_t_size);
    CHECK(counter2NameOffset == 8u * uint32_t_size );
    // The length of the SWTrace string (name)

    CHECK(categoryRecordPool[eventRecord1Offset + // Offset to the event record
                                   categoryRecordBlockSize +
                                   counter2NameOffset   // Offset to the name of the counter
                                  ] == counter2->m_Name.size() + 1); // The length of the name including the
                                                                     // null-terminator
    // The counter2 name
    CHECK(std::memcmp(categoryRecordPool +      // The beginning of the category pool
                            categoryRecordBlockSize + // Offset to the end of the category record block
                            eventRecord1Offset +      // Offset to the event record
                            counter2NameOffset +      // Offset to the name of the counter
                            uint32_t_size,            // The length of the name
                            counter2->m_Name.data(),
                            counter2->m_Name.size()) == 0); // name


    // The null-terminator at the end of the counter2 name
    CHECK(categoryRecordPool[eventRecord1Offset +      // Offset to the event record
                                   categoryRecordBlockSize + // Offset to the end of the category record block
                                   counter2NameOffset +      // Offset to the name of the counter
                                   uint32_t_size +           // The length of the name
                                   counter2->m_Name.size()   // The name of the counter
                                   ] == '\0');

    // Counter3 name
    uint32_t counter3NameOffset = 0;
    std::memcpy(&counter3NameOffset, categoryRecordPool + eventRecord2Offset + 5u * uint32_t_size, uint32_t_size);
    CHECK(counter3NameOffset == 0);
    // The length of the SWTrace string (name)
    CHECK(categoryRecordPool[eventRecord2Offset + // Offset to the event record
                                   categoryRecordBlockSize +
                                   8u * uint32_t_size + // Offset to the event record pool
                                   counter3NameOffset   // Offset to the name of the counter
                                  ] == counter3->m_Name.size() + 1); // The length of the name including the
                                                                     // null-terminator
    // The counter3 name
    CHECK(std::memcmp(categoryRecordPool + // The beginning of the category pool
                            categoryRecordBlockSize +
                            eventRecord2Offset + // Offset to the event record
                            8u * uint32_t_size + // Offset to the event record pool
                            counter3NameOffset + // Offset to the name of the counter
                            uint32_t_size,       // The length of the name
                            counter3->m_Name.data(),
                            counter3->m_Name.size()) == 0); // name
    // The null-terminator at the end of the counter3 name
    CHECK(categoryRecordPool[eventRecord2Offset +    // Offset to the event record
                                   categoryRecordBlockSize +
                                   8u * uint32_t_size +    // Offset to the event record pool
                                   counter3NameOffset +    // Offset to the name of the counter
                                   uint32_t_size +         // The length of the name
                                   counter3->m_Name.size() // The name of the counter
                                   ] == '\0');
}

TEST_CASE("CreateInvalidCategoryRecordTest1")
{
    MockBufferManager mockBuffer(0);
    SendCounterPacketTest sendCounterPacketTest(mockBuffer);

    // Create a category for testing
    const std::string categoryName = "some invalid category";
    const CategoryPtr category = std::make_unique<Category>(categoryName);
    CHECK(category);

    // Create a category record
    Counters counters;
    SendCounterPacket::CategoryRecord categoryRecord;
    std::string errorMessage;
    bool result = sendCounterPacketTest.CreateCategoryRecordTest(category, counters, categoryRecord, errorMessage);

    CHECK(!result);
    CHECK(!errorMessage.empty());
    CHECK(categoryRecord.empty());
}

TEST_CASE("CreateInvalidCategoryRecordTest2")
{
    MockBufferManager mockBuffer(0);
    SendCounterPacketTest sendCounterPacketTest(mockBuffer);

    // Create a category for testing
    const std::string categoryName = "some_category";
    const CategoryPtr category = std::make_unique<Category>(categoryName);
    CHECK(category);
    category->m_Counters = { 11u, 23u, 5670u };

    // Create a collection of counters
    Counters counters;
    counters.insert(std::make_pair<uint16_t, CounterPtr>(11,
                                                         CounterPtr(new Counter(armnn::profiling::BACKEND_ID,
                                                                                11,
                                                                                1234,
                                                                                0,
                                                                                1,
                                                                                534.0003f,
                                                                                "count€r1", // Invalid name
                                                                                "the first counter",
                                                                                "millipi2",
                                                                                0,
                                                                                0))));

    Counter* counter1 = counters.find(11)->second.get();
    CHECK(counter1);

    // Create a category record
    SendCounterPacket::CategoryRecord categoryRecord;
    std::string errorMessage;
    bool result = sendCounterPacketTest.CreateCategoryRecordTest(category, counters, categoryRecord, errorMessage);

    CHECK(!result);
    CHECK(!errorMessage.empty());
    CHECK(categoryRecord.empty());
}

TEST_CASE("SendCounterDirectoryPacketTest1")
{
    // The counter directory used for testing
    CounterDirectory counterDirectory;

    // Register a device
    const std::string device1Name = "device1";
    const Device* device1 = nullptr;
    CHECK_NOTHROW(device1 = counterDirectory.RegisterDevice(device1Name, 3));
    CHECK(counterDirectory.GetDeviceCount() == 1);
    CHECK(device1);

    // Register a device
    const std::string device2Name = "device2";
    const Device* device2 = nullptr;
    CHECK_NOTHROW(device2 = counterDirectory.RegisterDevice(device2Name));
    CHECK(counterDirectory.GetDeviceCount() == 2);
    CHECK(device2);

    // Buffer with not enough space
    MockBufferManager mockBuffer(10);
    SendCounterPacket sendCounterPacket(mockBuffer,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    CHECK_THROWS_AS(sendCounterPacket.SendCounterDirectoryPacket(counterDirectory),
                      BufferExhaustion);
}

TEST_CASE("SendCounterDirectoryPacketTest2")
{
    // The counter directory used for testing
    CounterDirectory counterDirectory;

    // Register a device
    const std::string device1Name = "device1";
    const Device* device1 = nullptr;
    CHECK_NOTHROW(device1 = counterDirectory.RegisterDevice(device1Name, 3));
    CHECK(counterDirectory.GetDeviceCount() == 1);
    CHECK(device1);

    // Register a device
    const std::string device2Name = "device2";
    const Device* device2 = nullptr;
    CHECK_NOTHROW(device2 = counterDirectory.RegisterDevice(device2Name));
    CHECK(counterDirectory.GetDeviceCount() == 2);
    CHECK(device2);

    // Register a counter set
    const std::string counterSet1Name = "counterset1";
    const CounterSet* counterSet1 = nullptr;
    CHECK_NOTHROW(counterSet1 = counterDirectory.RegisterCounterSet(counterSet1Name));
    CHECK(counterDirectory.GetCounterSetCount() == 1);
    CHECK(counterSet1);

    // Register a category associated to "device1" and "counterset1"
    const std::string category1Name = "category1";
    const Category* category1 = nullptr;
    CHECK_NOTHROW(category1 = counterDirectory.RegisterCategory(category1Name));
    CHECK(counterDirectory.GetCategoryCount() == 1);
    CHECK(category1);

    // Register a category not associated to "device2" but no counter set
    const std::string category2Name = "category2";
    const Category* category2 = nullptr;
    CHECK_NOTHROW(category2 = counterDirectory.RegisterCategory(category2Name));
    CHECK(counterDirectory.GetCategoryCount() == 2);
    CHECK(category2);

    uint16_t numberOfCores = 4;

    // Register a counter associated to "category1"
    const Counter* counter1 = nullptr;
    CHECK_NOTHROW(counter1 = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                              0,
                                                              category1Name,
                                                              0,
                                                              1,
                                                              123.45f,
                                                              "counter1",
                                                              "counter1description",
                                                              std::string("counter1units"),
                                                              numberOfCores));
    CHECK(counterDirectory.GetCounterCount() == 4);
    CHECK(counter1);

    // Register a counter associated to "category1"
    const Counter* counter2 = nullptr;
    CHECK_NOTHROW(counter2 = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                              4,
                                                              category1Name,
                                                              1,
                                                              0,
                                                              330.1245656765f,
                                                              "counter2",
                                                              "counter2description",
                                                              std::string("counter2units"),
                                                              arm::pipe::EmptyOptional(),
                                                              device2->m_Uid,
                                                              0));
    CHECK(counterDirectory.GetCounterCount() == 5);
    CHECK(counter2);

    // Register a counter associated to "category2"
    const Counter* counter3 = nullptr;
    CHECK_NOTHROW(counter3 = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                              5,
                                                              category2Name,
                                                              1,
                                                              1,
                                                              0.0000045399f,
                                                              "counter3",
                                                              "counter3description",
                                                              arm::pipe::EmptyOptional(),
                                                              numberOfCores,
                                                              device2->m_Uid,
                                                              counterSet1->m_Uid));
    CHECK(counterDirectory.GetCounterCount() == 9);
    CHECK(counter3);

    // Buffer with enough space
    MockBufferManager mockBuffer(1024);
    SendCounterPacket sendCounterPacket(mockBuffer,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    CHECK_NOTHROW(sendCounterPacket.SendCounterDirectoryPacket(counterDirectory));

    // Get the readable buffer
    auto readBuffer = mockBuffer.GetReadableBuffer();

    // Check the packet header
    const uint32_t packetHeaderWord0 = ReadUint32(readBuffer, 0);
    const uint32_t packetHeaderWord1 = ReadUint32(readBuffer, 4);
    CHECK(((packetHeaderWord0 >> 26) & 0x3F) == 0);  // packet_family
    CHECK(((packetHeaderWord0 >> 16) & 0x3FF) == 2); // packet_id
    CHECK(packetHeaderWord1 == 432);                 // data_length

    // Check the body header
    const uint32_t bodyHeaderWord0 = ReadUint32(readBuffer,  8);
    const uint32_t bodyHeaderWord1 = ReadUint32(readBuffer, 12);
    const uint32_t bodyHeaderWord2 = ReadUint32(readBuffer, 16);
    const uint32_t bodyHeaderWord3 = ReadUint32(readBuffer, 20);
    const uint32_t bodyHeaderWord4 = ReadUint32(readBuffer, 24);
    const uint32_t bodyHeaderWord5 = ReadUint32(readBuffer, 28);
    const uint16_t deviceRecordCount     = static_cast<uint16_t>(bodyHeaderWord0 >> 16);
    const uint16_t counterSetRecordCount = static_cast<uint16_t>(bodyHeaderWord2 >> 16);
    const uint16_t categoryRecordCount   = static_cast<uint16_t>(bodyHeaderWord4 >> 16);
    CHECK(deviceRecordCount == 2);                      // device_records_count
    CHECK(bodyHeaderWord1 == bodyHeaderSize * 4);           // device_records_pointer_table_offset
    CHECK(counterSetRecordCount == 1);                  // counter_set_count
    CHECK(bodyHeaderWord3 == 8 + bodyHeaderSize * 4);       // counter_set_pointer_table_offset
    CHECK(categoryRecordCount == 2);                    // categories_count
    CHECK(bodyHeaderWord5 == 12 + bodyHeaderSize * 4);      // categories_pointer_table_offset

    // Check the device records pointer table
    const uint32_t deviceRecordOffset0 = ReadUint32(readBuffer, 32);
    const uint32_t deviceRecordOffset1 = ReadUint32(readBuffer, 36);
    CHECK(deviceRecordOffset0 == 20); // Device record offset for "device1"
    CHECK(deviceRecordOffset1 == 40); // Device record offset for "device2"

    // Check the counter set pointer table
    const uint32_t counterSetRecordOffset0 = ReadUint32(readBuffer, 40);
    CHECK(counterSetRecordOffset0 == 52); // Counter set record offset for "counterset1"

    // Check the category pointer table
    const uint32_t categoryRecordOffset0 = ReadUint32(readBuffer, 44);
    const uint32_t categoryRecordOffset1 = ReadUint32(readBuffer, 48);
    CHECK(categoryRecordOffset0 ==  72); // Category record offset for "category1"
    CHECK(categoryRecordOffset1 == 176); // Category record offset for "category2"

    // Get the device record pool offset
    const uint32_t uint32_t_size = sizeof(uint32_t);
    const uint32_t packetHeaderSize = 2u * uint32_t_size;

    // Device record structure/collection used for testing
    struct DeviceRecord
    {
        uint16_t    uid;
        uint16_t    cores;
        uint32_t    name_offset;
        uint32_t    name_length;
        std::string name;
    };
    std::vector<DeviceRecord> deviceRecords;
    const uint32_t deviceRecordsPointerTableOffset = packetHeaderSize +
                                                     bodyHeaderWord1;     // device_records_pointer_table_offset

    const unsigned char* readData = readBuffer->GetReadableData();

    uint32_t offset = 0;
    std::vector<uint32_t> data(800);

    for (uint32_t i = 0; i < 800; i+=uint32_t_size)
    {
        data[i] = ReadUint32(readBuffer, offset);
        offset += uint32_t_size;
    }

    std::vector<uint32_t> deviceRecordOffsets(deviceRecordCount);
     offset = deviceRecordsPointerTableOffset;
    for (uint32_t i = 0; i < deviceRecordCount; ++i)
    {
        // deviceRecordOffset is relative to the start of the deviceRecordsPointerTable
        deviceRecordOffsets[i] = ReadUint32(readBuffer, offset) + deviceRecordsPointerTableOffset;
        offset += uint32_t_size;
    }

    for (uint32_t i = 0; i < deviceRecordCount; i++)
    {
        // Collect the data for the device record
        const uint32_t deviceRecordWord0 = ReadUint32(readBuffer, deviceRecordOffsets[i] + 0 * uint32_t_size);
        const uint32_t deviceRecordWord1 = ReadUint32(readBuffer, deviceRecordOffsets[i] + 1 * uint32_t_size);
        DeviceRecord deviceRecord;
        deviceRecord.uid = static_cast<uint16_t>(deviceRecordWord0 >> 16); // uid
        deviceRecord.cores = static_cast<uint16_t>(deviceRecordWord0);     // cores
        deviceRecord.name_offset = deviceRecordWord1;                      // name_offset

        uint32_t deviceRecordPoolOffset = deviceRecordOffsets[i] +                  // Packet body offset
                                          deviceRecord.name_offset; // Device name offset
        uint32_t deviceRecordNameLength = ReadUint32(readBuffer, deviceRecordPoolOffset);
        deviceRecord.name_length = deviceRecordNameLength; // name_length
        unsigned char deviceRecordNameNullTerminator = // name null-terminator
                ReadUint8(readBuffer, deviceRecordPoolOffset + uint32_t_size + deviceRecordNameLength - 1);
        CHECK(deviceRecordNameNullTerminator == '\0');
        std::vector<unsigned char> deviceRecordNameBuffer(deviceRecord.name_length - 1);
        std::memcpy(deviceRecordNameBuffer.data(),
                    readData + deviceRecordPoolOffset + uint32_t_size, deviceRecordNameBuffer.size());
        deviceRecord.name.assign(deviceRecordNameBuffer.begin(), deviceRecordNameBuffer.end()); // name

        deviceRecords.push_back(deviceRecord);
    }

    // Check that the device records are correct
    CHECK(deviceRecords.size() == 2);
    for (const DeviceRecord& deviceRecord : deviceRecords)
    {
        const Device* device = counterDirectory.GetDevice(deviceRecord.uid);
        CHECK(device);
        CHECK(device->m_Uid   == deviceRecord.uid);
        CHECK(device->m_Cores == deviceRecord.cores);
        CHECK(device->m_Name  == deviceRecord.name);
    }


    // Counter set record structure/collection used for testing
    struct CounterSetRecord
    {
        uint16_t    uid;
        uint16_t    count;
        uint32_t    name_offset;
        uint32_t    name_length;
        std::string name;
    };
    std::vector<CounterSetRecord> counterSetRecords;
    const uint32_t counterSetRecordsPointerTableOffset = 2u * uint32_t_size + // packet_header
                                                         bodyHeaderWord3;     // counter_set_pointer_table_offset

    offset = counterSetRecordsPointerTableOffset;
    std::vector<uint32_t> counterSetRecordOffsets(counterSetRecordCount);

    for (uint32_t i = 0; i < counterSetRecordCount; ++i)
    {
        // counterSetRecordOffset is relative to the start of the dcounterSetRecordsPointerTable
        counterSetRecordOffsets[i] = ReadUint32(readBuffer, offset) + counterSetRecordsPointerTableOffset;
        offset += uint32_t_size;
    }

    for (uint32_t i = 0; i < counterSetRecordCount; i++)
    {
        // Collect the data for the counter set record
        const uint32_t counterSetRecordWord0 = ReadUint32(readBuffer, counterSetRecordOffsets[i] + 0 * uint32_t_size);
        const uint32_t counterSetRecordWord1 = ReadUint32(readBuffer, counterSetRecordOffsets[i] + 1 * uint32_t_size);
        CounterSetRecord counterSetRecord;
        counterSetRecord.uid = static_cast<uint16_t>(counterSetRecordWord0 >> 16); // uid
        counterSetRecord.count = static_cast<uint16_t>(counterSetRecordWord0);     // count
        counterSetRecord.name_offset = counterSetRecordWord1;                      // name_offset

        uint32_t counterSetRecordPoolOffset = counterSetRecordOffsets[i]  +                 // Packet body offset
                                              counterSetRecord.name_offset; // Counter set name offset
        uint32_t counterSetRecordNameLength = ReadUint32(readBuffer, counterSetRecordPoolOffset);
        counterSetRecord.name_length = counterSetRecordNameLength; // name_length
        unsigned char counterSetRecordNameNullTerminator = // name null-terminator
                ReadUint8(readBuffer, counterSetRecordPoolOffset + uint32_t_size + counterSetRecordNameLength - 1);
        CHECK(counterSetRecordNameNullTerminator == '\0');
        std::vector<unsigned char> counterSetRecordNameBuffer(counterSetRecord.name_length - 1);
        std::memcpy(counterSetRecordNameBuffer.data(),
                    readData + counterSetRecordPoolOffset + uint32_t_size, counterSetRecordNameBuffer.size());
        counterSetRecord.name.assign(counterSetRecordNameBuffer.begin(), counterSetRecordNameBuffer.end()); // name

        counterSetRecords.push_back(counterSetRecord);
    }

    // Check that the counter set records are correct
    CHECK(counterSetRecords.size() == 1);
    for (const CounterSetRecord& counterSetRecord : counterSetRecords)
    {
        const CounterSet* counterSet = counterDirectory.GetCounterSet(counterSetRecord.uid);
        CHECK(counterSet);
        CHECK(counterSet->m_Uid   == counterSetRecord.uid);
        CHECK(counterSet->m_Count == counterSetRecord.count);
        CHECK(counterSet->m_Name  == counterSetRecord.name);
    }

    // Event record structure/collection used for testing
    struct EventRecord
    {
        uint16_t    counter_uid;
        uint16_t    max_counter_uid;
        uint16_t    device;
        uint16_t    counter_set;
        uint16_t    counter_class;
        uint16_t    interpolation;
        double      multiplier;
        uint32_t    name_offset;
        uint32_t    name_length;
        std::string name;
        uint32_t    description_offset;
        uint32_t    description_length;
        std::string description;
        uint32_t    units_offset;
        uint32_t    units_length;
        std::string units;
    };
    // Category record structure/collection used for testing
    struct CategoryRecord
    {
        uint16_t                 event_count;
        uint32_t                 event_pointer_table_offset;
        uint32_t                 name_offset;
        uint32_t                 name_length;
        std::string              name;
        std::vector<uint32_t>    event_pointer_table;
        std::vector<EventRecord> event_records;
    };
    std::vector<CategoryRecord> categoryRecords;
    const uint32_t categoryRecordsPointerTableOffset = 2u * uint32_t_size + // packet_header
                                                       bodyHeaderWord5;    // categories_pointer_table_offset

    offset = categoryRecordsPointerTableOffset;
    std::vector<uint32_t> categoryRecordOffsets(categoryRecordCount);
    for (uint32_t i = 0; i < categoryRecordCount; ++i)
    {
        // categoryRecordOffset is relative to the start of the categoryRecordsPointerTable
        categoryRecordOffsets[i] = ReadUint32(readBuffer, offset) + categoryRecordsPointerTableOffset;
        offset += uint32_t_size;
    }

    for (uint32_t i = 0; i < categoryRecordCount; i++)
    {
        // Collect the data for the category record
        const uint32_t categoryRecordWord1 = ReadUint32(readBuffer, categoryRecordOffsets[i] + 0 * uint32_t_size);
        const uint32_t categoryRecordWord2 = ReadUint32(readBuffer, categoryRecordOffsets[i] + 1 * uint32_t_size);
        const uint32_t categoryRecordWord3 = ReadUint32(readBuffer, categoryRecordOffsets[i] + 2 * uint32_t_size);
        CategoryRecord categoryRecord;
        categoryRecord.event_count = static_cast<uint16_t>(categoryRecordWord1 >> 16); // event_count
        categoryRecord.event_pointer_table_offset = categoryRecordWord2;               // event_pointer_table_offset
        categoryRecord.name_offset = categoryRecordWord3;                              // name_offset

        uint32_t categoryRecordNameLength = ReadUint32(readBuffer,
                                                       categoryRecordOffsets[i] + categoryRecord.name_offset);
        categoryRecord.name_length = categoryRecordNameLength; // name_length
        unsigned char categoryRecordNameNullTerminator =
                ReadUint8(readBuffer,
                          categoryRecordOffsets[i] +
                          categoryRecord.name_offset +
                          uint32_t_size +
                          categoryRecordNameLength - 1); // name null-terminator
        CHECK(categoryRecordNameNullTerminator == '\0');
        std::vector<unsigned char> categoryRecordNameBuffer(categoryRecord.name_length - 1);
        std::memcpy(categoryRecordNameBuffer.data(),
                    readData +
                    categoryRecordOffsets[i] +
                    categoryRecord.name_offset +
                    uint32_t_size,
                    categoryRecordNameBuffer.size());
        categoryRecord.name.assign(categoryRecordNameBuffer.begin(), categoryRecordNameBuffer.end()); // name

        categoryRecord.event_pointer_table.resize(categoryRecord.event_count);
        offset = categoryRecordOffsets[i] + categoryRecord.event_pointer_table_offset;
        for (uint32_t eventOffsetIndex = 0; eventOffsetIndex < categoryRecord.event_count; ++eventOffsetIndex)
        {
            // eventRecordOffset is relative to the start of the event pointer table
            categoryRecord.event_pointer_table[eventOffsetIndex] = ReadUint32(readBuffer, offset) +
                                                                   categoryRecordOffsets[i] +
                                                                   categoryRecord.event_pointer_table_offset;
            offset += uint32_t_size;
        }

        for (uint32_t eventIndex = 0; eventIndex < categoryRecord.event_count; eventIndex++)
        {
            const uint32_t eventOffset = categoryRecord.event_pointer_table[eventIndex];
            // Collect the data for the event record
            const uint32_t eventRecordWord0  = ReadUint32(readBuffer, eventOffset + 0 * uint32_t_size);
            const uint32_t eventRecordWord1  = ReadUint32(readBuffer, eventOffset + 1 * uint32_t_size);
            const uint32_t eventRecordWord2  = ReadUint32(readBuffer, eventOffset + 2 * uint32_t_size);
            const uint64_t eventRecordWord34 = ReadUint64(readBuffer, eventOffset + 3 * uint32_t_size);
            const uint32_t eventRecordWord5  = ReadUint32(readBuffer, eventOffset + 5 * uint32_t_size);
            const uint32_t eventRecordWord6  = ReadUint32(readBuffer, eventOffset + 6 * uint32_t_size);
            const uint32_t eventRecordWord7  = ReadUint32(readBuffer, eventOffset + 7 * uint32_t_size);

            EventRecord eventRecord;
            eventRecord.counter_uid = static_cast<uint16_t>(eventRecordWord0);                     // counter_uid
            eventRecord.max_counter_uid = static_cast<uint16_t>(eventRecordWord0 >> 16);           // max_counter_uid
            eventRecord.device = static_cast<uint16_t>(eventRecordWord1 >> 16);                    // device
            eventRecord.counter_set = static_cast<uint16_t>(eventRecordWord1);                     // counter_set
            eventRecord.counter_class = static_cast<uint16_t>(eventRecordWord2 >> 16);             // class
            eventRecord.interpolation = static_cast<uint16_t>(eventRecordWord2);                   // interpolation
            std::memcpy(&eventRecord.multiplier, &eventRecordWord34, sizeof(eventRecord.multiplier)); // multiplier
            eventRecord.name_offset = static_cast<uint32_t>(eventRecordWord5);                     // name_offset
            eventRecord.description_offset = static_cast<uint32_t>(eventRecordWord6);              // description_offset
            eventRecord.units_offset = static_cast<uint32_t>(eventRecordWord7);                    // units_offset

            uint32_t eventRecordNameLength = ReadUint32(readBuffer, eventOffset + eventRecord.name_offset);
            eventRecord.name_length = eventRecordNameLength; // name_length
            unsigned char eventRecordNameNullTerminator =
                    ReadUint8(readBuffer,
                              eventOffset +
                              eventRecord.name_offset +
                              uint32_t_size +
                              eventRecordNameLength - 1); // name null-terminator
            CHECK(eventRecordNameNullTerminator == '\0');
            std::vector<unsigned char> eventRecordNameBuffer(eventRecord.name_length - 1);
            std::memcpy(eventRecordNameBuffer.data(),
                        readData +
                        eventOffset +
                        eventRecord.name_offset +
                        uint32_t_size,
                        eventRecordNameBuffer.size());
            eventRecord.name.assign(eventRecordNameBuffer.begin(), eventRecordNameBuffer.end()); // name

            uint32_t eventRecordDescriptionLength = ReadUint32(readBuffer,
                                                               eventOffset + eventRecord.description_offset);
            eventRecord.description_length = eventRecordDescriptionLength; // description_length
            unsigned char eventRecordDescriptionNullTerminator =
                    ReadUint8(readBuffer,
                              eventOffset +
                              eventRecord.description_offset +
                              uint32_t_size +
                              eventRecordDescriptionLength - 1); // description null-terminator
            CHECK(eventRecordDescriptionNullTerminator == '\0');
            std::vector<unsigned char> eventRecordDescriptionBuffer(eventRecord.description_length - 1);
            std::memcpy(eventRecordDescriptionBuffer.data(),
                        readData +
                        eventOffset +
                        eventRecord.description_offset +
                        uint32_t_size,
                        eventRecordDescriptionBuffer.size());
            eventRecord.description.assign(eventRecordDescriptionBuffer.begin(),
                                           eventRecordDescriptionBuffer.end()); // description

            if (eventRecord.units_offset > 0)
            {
                uint32_t eventRecordUnitsLength = ReadUint32(readBuffer,
                                                             eventOffset + eventRecord.units_offset);
                eventRecord.units_length = eventRecordUnitsLength; // units_length
                unsigned char eventRecordUnitsNullTerminator =
                        ReadUint8(readBuffer,
                                  eventOffset +
                                  eventRecord.units_offset +
                                  uint32_t_size +
                                  eventRecordUnitsLength - 1); // units null-terminator
                CHECK(eventRecordUnitsNullTerminator == '\0');
                std::vector<unsigned char> eventRecordUnitsBuffer(eventRecord.units_length - 1);
                std::memcpy(eventRecordUnitsBuffer.data(),
                            readData +
                            eventOffset +
                            eventRecord.units_offset +
                            uint32_t_size,
                            eventRecordUnitsBuffer.size());
                eventRecord.units.assign(eventRecordUnitsBuffer.begin(), eventRecordUnitsBuffer.end()); // units
            }

            categoryRecord.event_records.push_back(eventRecord);
        }

        categoryRecords.push_back(categoryRecord);
    }

    // Check that the category records are correct
    CHECK(categoryRecords.size() == 2);
    for (const CategoryRecord& categoryRecord : categoryRecords)
    {
        const Category* category = counterDirectory.GetCategory(categoryRecord.name);
        CHECK(category);
        CHECK(category->m_Name == categoryRecord.name);
        CHECK(category->m_Counters.size() == categoryRecord.event_count + static_cast<size_t>(numberOfCores) -1);
        CHECK(category->m_Counters.size() == categoryRecord.event_count + static_cast<size_t>(numberOfCores) -1);

        // Check that the event records are correct
        for (const EventRecord& eventRecord : categoryRecord.event_records)
        {
            const Counter* counter = counterDirectory.GetCounter(eventRecord.counter_uid);
            CHECK(counter);
            CHECK(counter->m_MaxCounterUid == eventRecord.max_counter_uid);
            CHECK(counter->m_DeviceUid == eventRecord.device);
            CHECK(counter->m_CounterSetUid == eventRecord.counter_set);
            CHECK(counter->m_Class == eventRecord.counter_class);
            CHECK(counter->m_Interpolation == eventRecord.interpolation);
            CHECK(counter->m_Multiplier == eventRecord.multiplier);
            CHECK(counter->m_Name == eventRecord.name);
            CHECK(counter->m_Description == eventRecord.description);
            CHECK(counter->m_Units == eventRecord.units);
        }
    }
}

TEST_CASE("SendCounterDirectoryPacketTest3")
{
    // Using a mock counter directory that allows to register invalid objects
    MockCounterDirectory counterDirectory;

    // Register an invalid device
    const std::string deviceName = "inv@lid dev!c€";
    const Device* device = nullptr;
    CHECK_NOTHROW(device = counterDirectory.RegisterDevice(deviceName, 3));
    CHECK(counterDirectory.GetDeviceCount() == 1);
    CHECK(device);

    // Buffer with enough space
    MockBufferManager mockBuffer(1024);
    SendCounterPacket sendCounterPacket(mockBuffer,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    CHECK_THROWS_AS(sendCounterPacket.SendCounterDirectoryPacket(counterDirectory), arm::pipe::ProfilingException);
}

TEST_CASE("SendCounterDirectoryPacketTest4")
{
    // Using a mock counter directory that allows to register invalid objects
    MockCounterDirectory counterDirectory;

    // Register an invalid counter set
    const std::string counterSetName = "inv@lid count€rs€t";
    const CounterSet* counterSet = nullptr;
    CHECK_NOTHROW(counterSet = counterDirectory.RegisterCounterSet(counterSetName));
    CHECK(counterDirectory.GetCounterSetCount() == 1);
    CHECK(counterSet);

    // Buffer with enough space
    MockBufferManager mockBuffer(1024);
    SendCounterPacket sendCounterPacket(mockBuffer,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    CHECK_THROWS_AS(sendCounterPacket.SendCounterDirectoryPacket(counterDirectory), arm::pipe::ProfilingException);
}

TEST_CASE("SendCounterDirectoryPacketTest5")
{
    // Using a mock counter directory that allows to register invalid objects
    MockCounterDirectory counterDirectory;

    // Register an invalid category
    const std::string categoryName = "c@t€gory";
    const Category* category = nullptr;
    CHECK_NOTHROW(category = counterDirectory.RegisterCategory(categoryName));
    CHECK(counterDirectory.GetCategoryCount() == 1);
    CHECK(category);

    // Buffer with enough space
    MockBufferManager mockBuffer(1024);
    SendCounterPacket sendCounterPacket(mockBuffer,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    CHECK_THROWS_AS(sendCounterPacket.SendCounterDirectoryPacket(counterDirectory), arm::pipe::ProfilingException);
}

TEST_CASE("SendCounterDirectoryPacketTest6")
{
    // Using a mock counter directory that allows to register invalid objects
    MockCounterDirectory counterDirectory;

    // Register an invalid device
    const std::string deviceName = "inv@lid dev!c€";
    const Device* device = nullptr;
    CHECK_NOTHROW(device = counterDirectory.RegisterDevice(deviceName, 3));
    CHECK(counterDirectory.GetDeviceCount() == 1);
    CHECK(device);

    // Register an invalid counter set
    const std::string counterSetName = "inv@lid count€rs€t";
    const CounterSet* counterSet = nullptr;
    CHECK_NOTHROW(counterSet = counterDirectory.RegisterCounterSet(counterSetName));
    CHECK(counterDirectory.GetCounterSetCount() == 1);
    CHECK(counterSet);

    // Register an invalid category associated to an invalid device and an invalid counter set
    const std::string categoryName = "c@t€gory";
    const Category* category = nullptr;
    CHECK_NOTHROW(category = counterDirectory.RegisterCategory(categoryName));
    CHECK(counterDirectory.GetCategoryCount() == 1);
    CHECK(category);

    // Buffer with enough space
    MockBufferManager mockBuffer(1024);
    SendCounterPacket sendCounterPacket(mockBuffer,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    CHECK_THROWS_AS(sendCounterPacket.SendCounterDirectoryPacket(counterDirectory), arm::pipe::ProfilingException);
}

TEST_CASE("SendCounterDirectoryPacketTest7")
{
    // Using a mock counter directory that allows to register invalid objects
    MockCounterDirectory counterDirectory;

    // Register an valid device
    const std::string deviceName = "valid device";
    const Device* device = nullptr;
    CHECK_NOTHROW(device = counterDirectory.RegisterDevice(deviceName, 3));
    CHECK(counterDirectory.GetDeviceCount() == 1);
    CHECK(device);

    // Register an valid counter set
    const std::string counterSetName = "valid counterset";
    const CounterSet* counterSet = nullptr;
    CHECK_NOTHROW(counterSet = counterDirectory.RegisterCounterSet(counterSetName));
    CHECK(counterDirectory.GetCounterSetCount() == 1);
    CHECK(counterSet);

    // Register an valid category associated to a valid device and a valid counter set
    const std::string categoryName = "category";
    const Category* category = nullptr;
    CHECK_NOTHROW(category = counterDirectory.RegisterCategory(categoryName));
    CHECK(counterDirectory.GetCategoryCount() == 1);
    CHECK(category);

    // Register an invalid counter associated to a valid category
    const Counter* counter = nullptr;
    CHECK_NOTHROW(counter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                             0,
                                                             categoryName,
                                                             0,
                                                             1,
                                                             123.45f,
                                                             "counter",
                                                             "counter description",
                                                             std::string("invalid counter units"),
                                                             5,
                                                             device->m_Uid,
                                                             counterSet->m_Uid));
    CHECK(counterDirectory.GetCounterCount() == 5);
    CHECK(counter);

    // Buffer with enough space
    MockBufferManager mockBuffer(1024);
    SendCounterPacket sendCounterPacket(mockBuffer,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    CHECK_THROWS_AS(sendCounterPacket.SendCounterDirectoryPacket(counterDirectory), arm::pipe::ProfilingException);
}

TEST_CASE("SendThreadTest0")
{
    ProfilingStateMachine profilingStateMachine;
    SetActiveProfilingState(profilingStateMachine);

    MockProfilingConnection mockProfilingConnection;
    MockStreamCounterBuffer mockStreamCounterBuffer(0);
    SendCounterPacket sendCounterPacket(mockStreamCounterBuffer,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    SendThread sendThread(profilingStateMachine, mockStreamCounterBuffer, sendCounterPacket);

    // Try to start the send thread many times, it must only start once

    sendThread.Start(mockProfilingConnection);
    CHECK(sendThread.IsRunning());
    sendThread.Start(mockProfilingConnection);
    sendThread.Start(mockProfilingConnection);
    sendThread.Start(mockProfilingConnection);
    sendThread.Start(mockProfilingConnection);
    CHECK(sendThread.IsRunning());

    sendThread.Stop();
    CHECK(!sendThread.IsRunning());
}

TEST_CASE("SendThreadTest1")
{
    ProfilingStateMachine profilingStateMachine;
    SetActiveProfilingState(profilingStateMachine);

    unsigned int totalWrittenSize = 0;

    MockProfilingConnection mockProfilingConnection;
    MockStreamCounterBuffer mockStreamCounterBuffer(1024);
    SendCounterPacket sendCounterPacket(mockStreamCounterBuffer,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    SendThread sendThread(profilingStateMachine, mockStreamCounterBuffer, sendCounterPacket);
    sendThread.Start(mockProfilingConnection);

    // Interleaving writes and reads to/from the buffer with pauses to test that the send thread actually waits for
    // something to become available for reading

    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_UNTIL_READABLE_MS));

    CounterDirectory counterDirectory;
    sendCounterPacket.SendStreamMetaDataPacket();

    totalWrittenSize += GetStreamMetaDataPacketSize();

    sendThread.SetReadyToRead();

    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_UNTIL_READABLE_MS));

    sendCounterPacket.SendCounterDirectoryPacket(counterDirectory);

    // Get the size of the Counter Directory Packet
    unsigned int counterDirectoryPacketSize = 32;
    totalWrittenSize += counterDirectoryPacketSize;

    sendThread.SetReadyToRead();

    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_UNTIL_READABLE_MS));

    sendCounterPacket.SendPeriodicCounterCapturePacket(123u,
                                                       {
                                                           {   1u,      23u },
                                                           {  33u, 1207623u }
                                                       });

    // Get the size of the Periodic Counter Capture Packet
    unsigned int periodicCounterCapturePacketSize = 28;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendThread.SetReadyToRead();

    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_UNTIL_READABLE_MS));

    sendCounterPacket.SendPeriodicCounterCapturePacket(44u,
                                                       {
                                                           { 211u,     923u }
                                                       });

    // Get the size of the Periodic Counter Capture Packet
    periodicCounterCapturePacketSize = 22;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendCounterPacket.SendPeriodicCounterCapturePacket(1234u,
                                                       {
                                                           { 555u,      23u },
                                                           { 556u,       6u },
                                                           { 557u,  893454u },
                                                           { 558u, 1456623u },
                                                           { 559u,  571090u }
                                                       });

    // Get the size of the Periodic Counter Capture Packet
    periodicCounterCapturePacketSize = 46;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendCounterPacket.SendPeriodicCounterCapturePacket(997u,
                                                       {
                                                           {  88u,      11u },
                                                           {  96u,      22u },
                                                           {  97u,      33u },
                                                           { 999u,     444u }
                                                       });

    // Get the size of the Periodic Counter Capture Packet
    periodicCounterCapturePacketSize = 40;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendThread.SetReadyToRead();

    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_UNTIL_READABLE_MS));

    sendCounterPacket.SendPeriodicCounterSelectionPacket(1000u, { 1345u, 254u, 4536u, 408u, 54u, 6323u, 428u, 1u, 6u });

    // Get the size of the Periodic Counter Capture Packet
    periodicCounterCapturePacketSize = 30;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendThread.SetReadyToRead();

    // To test an exact value of the "read size" in the mock buffer, wait to allow the send thread to
    // read all what's remaining in the buffer
    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_UNTIL_READABLE_MS));

    sendThread.Stop();

    CHECK(mockStreamCounterBuffer.GetCommittedSize() == totalWrittenSize);
    CHECK(mockStreamCounterBuffer.GetReadableSize()  == totalWrittenSize);
    CHECK(mockStreamCounterBuffer.GetReadSize()      == totalWrittenSize);
}

TEST_CASE("SendThreadTest2")
{
    ProfilingStateMachine profilingStateMachine;
    SetActiveProfilingState(profilingStateMachine);

    unsigned int totalWrittenSize = 0;

    MockProfilingConnection mockProfilingConnection;
    MockStreamCounterBuffer mockStreamCounterBuffer(1024);
    SendCounterPacket sendCounterPacket(mockStreamCounterBuffer,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    SendThread sendThread(profilingStateMachine, mockStreamCounterBuffer, sendCounterPacket);
    sendThread.Start(mockProfilingConnection);

    // Adding many spurious "ready to read" signals throughout the test to check that the send thread is
    // capable of handling unnecessary read requests

    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_UNTIL_READABLE_MS));

    sendThread.SetReadyToRead();

    CounterDirectory counterDirectory;
    sendCounterPacket.SendStreamMetaDataPacket();

    totalWrittenSize += GetStreamMetaDataPacketSize();

    sendThread.SetReadyToRead();

    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_UNTIL_READABLE_MS));

    sendCounterPacket.SendCounterDirectoryPacket(counterDirectory);

    // Get the size of the Counter Directory Packet
    unsigned int counterDirectoryPacketSize = 32;
    totalWrittenSize += counterDirectoryPacketSize;

    sendThread.SetReadyToRead();
    sendThread.SetReadyToRead();

    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_UNTIL_READABLE_MS));

    sendCounterPacket.SendPeriodicCounterCapturePacket(123u,
                                                       {
                                                           {   1u,      23u },
                                                           {  33u, 1207623u }
                                                       });

    // Get the size of the Periodic Counter Capture Packet
    unsigned int periodicCounterCapturePacketSize = 28;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendThread.SetReadyToRead();

    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_UNTIL_READABLE_MS));

    sendThread.SetReadyToRead();
    sendThread.SetReadyToRead();
    sendThread.SetReadyToRead();

    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_UNTIL_READABLE_MS));

    sendThread.SetReadyToRead();
    sendCounterPacket.SendPeriodicCounterCapturePacket(44u,
                                                       {
                                                           { 211u,     923u }
                                                       });

    // Get the size of the Periodic Counter Capture Packet
    periodicCounterCapturePacketSize = 22;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendCounterPacket.SendPeriodicCounterCapturePacket(1234u,
                                                       {
                                                           { 555u,      23u },
                                                           { 556u,       6u },
                                                           { 557u,  893454u },
                                                           { 558u, 1456623u },
                                                           { 559u,  571090u }
                                                       });

    // Get the size of the Periodic Counter Capture Packet
    periodicCounterCapturePacketSize = 46;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendThread.SetReadyToRead();
    sendCounterPacket.SendPeriodicCounterCapturePacket(997u,
                                                       {
                                                           {  88u,      11u },
                                                           {  96u,      22u },
                                                           {  97u,      33u },
                                                           { 999u,     444u }
                                                       });

    // Get the size of the Periodic Counter Capture Packet
    periodicCounterCapturePacketSize = 40;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendThread.SetReadyToRead();
    sendThread.SetReadyToRead();

    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_UNTIL_READABLE_MS));

    sendCounterPacket.SendPeriodicCounterSelectionPacket(1000u, { 1345u, 254u, 4536u, 408u, 54u, 6323u, 428u, 1u, 6u });

    // Get the size of the Periodic Counter Capture Packet
    periodicCounterCapturePacketSize = 30;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendThread.SetReadyToRead();

    // To test an exact value of the "read size" in the mock buffer, wait to allow the send thread to
    // read all what's remaining in the buffer
    sendThread.Stop();

    CHECK(mockStreamCounterBuffer.GetCommittedSize() == totalWrittenSize);
    CHECK(mockStreamCounterBuffer.GetReadableSize()  == totalWrittenSize);
    CHECK(mockStreamCounterBuffer.GetReadSize()      == totalWrittenSize);
}

TEST_CASE("SendThreadTest3")
{
    ProfilingStateMachine profilingStateMachine;
    SetActiveProfilingState(profilingStateMachine);

    unsigned int totalWrittenSize = 0;

    MockProfilingConnection mockProfilingConnection;
    MockStreamCounterBuffer mockStreamCounterBuffer(1024);
    SendCounterPacket sendCounterPacket(mockStreamCounterBuffer,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    SendThread sendThread(profilingStateMachine, mockStreamCounterBuffer, sendCounterPacket);
    sendThread.Start(mockProfilingConnection);

    // Not using pauses or "grace periods" to stress test the send thread

    sendThread.SetReadyToRead();

    CounterDirectory counterDirectory;
    sendCounterPacket.SendStreamMetaDataPacket();

    totalWrittenSize += GetStreamMetaDataPacketSize();

    sendThread.SetReadyToRead();
    sendCounterPacket.SendCounterDirectoryPacket(counterDirectory);

    // Get the size of the Counter Directory Packet
    unsigned int counterDirectoryPacketSize =32;
    totalWrittenSize += counterDirectoryPacketSize;

    sendThread.SetReadyToRead();
    sendThread.SetReadyToRead();
    sendCounterPacket.SendPeriodicCounterCapturePacket(123u,
                                                       {
                                                           {   1u,      23u },
                                                           {  33u, 1207623u }
                                                       });

    // Get the size of the Periodic Counter Capture Packet
    unsigned int periodicCounterCapturePacketSize = 28;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendThread.SetReadyToRead();
    sendThread.SetReadyToRead();
    sendThread.SetReadyToRead();
    sendThread.SetReadyToRead();
    sendThread.SetReadyToRead();
    sendCounterPacket.SendPeriodicCounterCapturePacket(44u,
                                                       {
                                                           { 211u,     923u }
                                                       });

    // Get the size of the Periodic Counter Capture Packet
    periodicCounterCapturePacketSize = 22;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendCounterPacket.SendPeriodicCounterCapturePacket(1234u,
                                                       {
                                                           { 555u,      23u },
                                                           { 556u,       6u },
                                                           { 557u,  893454u },
                                                           { 558u, 1456623u },
                                                           { 559u,  571090u }
                                                       });

    // Get the size of the Periodic Counter Capture Packet
    periodicCounterCapturePacketSize = 46;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendThread.SetReadyToRead();
    sendThread.SetReadyToRead();
    sendCounterPacket.SendPeriodicCounterCapturePacket(997u,
                                                       {
                                                           {  88u,      11u },
                                                           {  96u,      22u },
                                                           {  97u,      33u },
                                                           { 999u,     444u }
                                                       });

    // Get the size of the Periodic Counter Capture Packet
    periodicCounterCapturePacketSize = 40;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendThread.SetReadyToRead();
    sendThread.SetReadyToRead();
    sendCounterPacket.SendPeriodicCounterSelectionPacket(1000u, { 1345u, 254u, 4536u, 408u, 54u, 6323u, 428u, 1u, 6u });

    // Get the size of the Periodic Counter Capture Packet
    periodicCounterCapturePacketSize = 30;
    totalWrittenSize += periodicCounterCapturePacketSize;

    sendThread.SetReadyToRead();

    // Abruptly terminating the send thread, the amount of data sent may be less that the amount written (the send
    // thread is not guaranteed to flush the buffer)
    sendThread.Stop();

    CHECK(mockStreamCounterBuffer.GetCommittedSize() == totalWrittenSize);
    CHECK(mockStreamCounterBuffer.GetReadableSize()  <= totalWrittenSize);
    CHECK(mockStreamCounterBuffer.GetReadSize()      <= totalWrittenSize);
    CHECK(mockStreamCounterBuffer.GetReadSize()      <= mockStreamCounterBuffer.GetReadableSize());
    CHECK(mockStreamCounterBuffer.GetReadSize()      <= mockStreamCounterBuffer.GetCommittedSize());
}

TEST_CASE("SendCounterPacketTestWithSendThread")
{
    ProfilingStateMachine profilingStateMachine;
    SetWaitingForAckProfilingState(profilingStateMachine);

    MockProfilingConnection mockProfilingConnection;
    BufferManager bufferManager(1, 1024);
    SendCounterPacket sendCounterPacket(bufferManager,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    SendThread sendThread(profilingStateMachine, bufferManager, sendCounterPacket, -1);
    sendThread.Start(mockProfilingConnection);

    unsigned int streamMetadataPacketsize = GetStreamMetaDataPacketSize();

    sendThread.Stop();

    // check for packet in ProfilingConnection
    CHECK(mockProfilingConnection.CheckForPacket({PacketType::StreamMetaData, streamMetadataPacketsize}) == 1);

    SetActiveProfilingState(profilingStateMachine);
    sendThread.Start(mockProfilingConnection);

    // SendCounterDirectoryPacket
    CounterDirectory counterDirectory;
    sendCounterPacket.SendCounterDirectoryPacket(counterDirectory);

    sendThread.Stop();
    unsigned int counterDirectoryPacketSize = 32;
    // check for packet in ProfilingConnection
    CHECK(mockProfilingConnection.CheckForPacket(
        {PacketType::CounterDirectory, counterDirectoryPacketSize}) == 1);

    sendThread.Start(mockProfilingConnection);

    // SendPeriodicCounterCapturePacket
    sendCounterPacket.SendPeriodicCounterCapturePacket(123u,
                                                       {
                                                           {   1u,      23u },
                                                           {  33u, 1207623u }
                                                       });

    sendThread.Stop();

    unsigned int periodicCounterCapturePacketSize = 28;
    CHECK(mockProfilingConnection.CheckForPacket(
        {PacketType::PeriodicCounterCapture, periodicCounterCapturePacketSize}) == 1);
}

TEST_CASE("SendThreadBufferTest")
{
    ProfilingStateMachine profilingStateMachine;
    SetActiveProfilingState(profilingStateMachine);

    MockProfilingConnection mockProfilingConnection;
    BufferManager bufferManager(3, 1024);
    SendCounterPacket sendCounterPacket(bufferManager,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    SendThread sendThread(profilingStateMachine, bufferManager, sendCounterPacket, -1);
    sendThread.Start(mockProfilingConnection);

    // SendStreamMetaDataPacket
    sendCounterPacket.SendStreamMetaDataPacket();

    // Read data from the buffer
    // Buffer should become readable after commit by SendStreamMetaDataPacket
    auto packetBuffer = bufferManager.GetReadableBuffer();
    CHECK(packetBuffer.get());

    unsigned int streamMetadataPacketsize = GetStreamMetaDataPacketSize();
    CHECK(packetBuffer->GetSize() == streamMetadataPacketsize);

    // Recommit to be read by sendCounterPacket
    bufferManager.Commit(packetBuffer, streamMetadataPacketsize);

    // SendCounterDirectoryPacket
    CounterDirectory counterDirectory;
    sendCounterPacket.SendCounterDirectoryPacket(counterDirectory);

    // SendPeriodicCounterCapturePacket
    sendCounterPacket.SendPeriodicCounterCapturePacket(123u,
                                                       {
                                                           {   1u,      23u },
                                                           {  33u, 1207623u }
                                                       });

    sendThread.Stop();

    // The buffer is read by the send thread so it should not be in the readable buffer.
    auto readBuffer = bufferManager.GetReadableBuffer();
    CHECK(!readBuffer);

    // Successfully reserved the buffer with requested size
    unsigned int reservedSize = 0;
    auto reservedBuffer = bufferManager.Reserve(512, reservedSize);
    CHECK(reservedSize == 512);
    CHECK(reservedBuffer.get());

    const auto writtenDataSize = mockProfilingConnection.GetWrittenDataSize();
    const auto metaDataPacketCount =
            mockProfilingConnection.CheckForPacket({PacketType::StreamMetaData, streamMetadataPacketsize});

    CHECK(metaDataPacketCount >= 1);
    CHECK(mockProfilingConnection.CheckForPacket({PacketType::CounterDirectory, 32}) == 1);
    CHECK(mockProfilingConnection.CheckForPacket({PacketType::PeriodicCounterCapture, 28}) == 1);
    // Check that we only received the packets we expected
    CHECK(metaDataPacketCount + 2 == writtenDataSize);
}

TEST_CASE("SendThreadSendStreamMetadataPacket1")
{
    ProfilingStateMachine profilingStateMachine;

    MockProfilingConnection mockProfilingConnection;
    BufferManager bufferManager(3, 1024);
    SendCounterPacket sendCounterPacket(bufferManager,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    SendThread sendThread(profilingStateMachine, bufferManager, sendCounterPacket);
    sendThread.Start(mockProfilingConnection);

    // The profiling state is set to "Uninitialized", so the send thread should throw an exception
    CHECK_THROWS_AS(sendThread.Stop(), arm::pipe::ProfilingException);
}

TEST_CASE("SendThreadSendStreamMetadataPacket2")
{
    ProfilingStateMachine profilingStateMachine;
    SetNotConnectedProfilingState(profilingStateMachine);

    MockProfilingConnection mockProfilingConnection;
    BufferManager bufferManager(3, 1024);
    SendCounterPacket sendCounterPacket(bufferManager,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    SendThread sendThread(profilingStateMachine, bufferManager, sendCounterPacket);
    sendThread.Start(mockProfilingConnection);

    // The profiling state is set to "NotConnected", so the send thread should throw an exception
    CHECK_THROWS_AS(sendThread.Stop(), arm::pipe::ProfilingException);
}

TEST_CASE("SendThreadSendStreamMetadataPacket3")
{
    ProfilingStateMachine profilingStateMachine;
    SetWaitingForAckProfilingState(profilingStateMachine);

    unsigned int streamMetadataPacketsize = GetStreamMetaDataPacketSize();

    MockProfilingConnection mockProfilingConnection;
    BufferManager bufferManager(3, 1024);
    SendCounterPacket sendCounterPacket(bufferManager,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    SendThread sendThread(profilingStateMachine, bufferManager, sendCounterPacket);
    sendThread.Start(mockProfilingConnection);

    // The profiling state is set to "WaitingForAck", so the send thread should send a Stream Metadata packet
    // Wait for sendThread to join
    CHECK_NOTHROW(sendThread.Stop());

    // Check that the buffer contains at least one Stream Metadata packet and no other packets
    const auto writtenDataSize = mockProfilingConnection.GetWrittenDataSize();

    CHECK(writtenDataSize >= 1u);
    CHECK(mockProfilingConnection.CheckForPacket(
                  {PacketType::StreamMetaData, streamMetadataPacketsize}) == writtenDataSize);
}

TEST_CASE("SendThreadSendStreamMetadataPacket4")
{
    ProfilingStateMachine profilingStateMachine;
    SetWaitingForAckProfilingState(profilingStateMachine);

    unsigned int streamMetadataPacketsize = GetStreamMetaDataPacketSize();

    MockProfilingConnection mockProfilingConnection;
    BufferManager bufferManager(3, 1024);
    SendCounterPacket sendCounterPacket(bufferManager,
                                        arm::pipe::ARMNN_SOFTWARE_INFO,
                                        arm::pipe::ARMNN_SOFTWARE_VERSION,
                                        arm::pipe::ARMNN_HARDWARE_VERSION);
    SendThread sendThread(profilingStateMachine, bufferManager, sendCounterPacket);
    sendThread.Start(mockProfilingConnection);

    // The profiling state is set to "WaitingForAck", so the send thread should send a Stream Metadata packet
    // Wait for sendThread to join
    sendThread.Stop();

    sendThread.Start(mockProfilingConnection);
    // Check that the profiling state is still "WaitingForAck"
    CHECK((profilingStateMachine.GetCurrentState() == ProfilingState::WaitingForAck));

    // Check that the buffer contains at least one Stream Metadata packet
    CHECK(mockProfilingConnection.CheckForPacket({PacketType::StreamMetaData, streamMetadataPacketsize}) >= 1);

    mockProfilingConnection.Clear();

    sendThread.Stop();
    sendThread.Start(mockProfilingConnection);

    // Try triggering a new buffer read
    sendThread.SetReadyToRead();

    // Wait for sendThread to join
    CHECK_NOTHROW(sendThread.Stop());

    // Check that the profiling state is still "WaitingForAck"
    CHECK((profilingStateMachine.GetCurrentState() == ProfilingState::WaitingForAck));

    // Check that the buffer contains at least one Stream Metadata packet and no other packets
    const auto writtenDataSize = mockProfilingConnection.GetWrittenDataSize();

    CHECK(writtenDataSize >= 1u);
    CHECK(mockProfilingConnection.CheckForPacket(
                  {PacketType::StreamMetaData, streamMetadataPacketsize}) == writtenDataSize);
}

}
