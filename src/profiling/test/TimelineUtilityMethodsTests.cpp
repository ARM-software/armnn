//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingMocks.hpp"
#include "ProfilingTestUtils.hpp"

#include <ArmNNProfilingServiceInitialiser.hpp>

#include <armnn/profiling/ArmNNProfiling.hpp>

#include <client/include/TimelineUtilityMethods.hpp>

#include <client/src/SendTimelinePacket.hpp>
#include <client/src/ProfilingService.hpp>

#include <common/include/LabelsAndEventClasses.hpp>

#include <memory>

#include <doctest/doctest.h>

using namespace armnn;
using namespace arm::pipe;

TEST_SUITE("TimelineUtilityMethodsTests")
{
TEST_CASE("CreateTypedLabelTest")
{
    MockBufferManager mockBufferManager(1024);
    armnn::ArmNNProfilingServiceInitialiser initialiser;
    ProfilingService profilingService(arm::pipe::MAX_ARMNN_COUNTER,
                                      initialiser,
                                      arm::pipe::ARMNN_SOFTWARE_INFO,
                                      arm::pipe::ARMNN_SOFTWARE_VERSION,
                                      arm::pipe::ARMNN_HARDWARE_VERSION);

    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);

    // Generate first guid to ensure that the named typed entity guid is not 0 on local single test.
    profilingService.NextGuid();

    ProfilingGuid entityGuid(123);
    const std::string entityName = "some entity";
    ProfilingStaticGuid labelTypeGuid(456);

    CHECK_NOTHROW(timelineUtilityMethods.MarkEntityWithLabel(entityGuid, entityName, labelTypeGuid));

    // Commit all packets at once
    timelineUtilityMethods.Commit();

    // Get the readable buffer
    auto readableBuffer = mockBufferManager.GetReadableBuffer();
    CHECK(readableBuffer != nullptr);
    unsigned int size = readableBuffer->GetSize();
    CHECK(size == 76);
    const unsigned char* readableData = readableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    // Utils
    unsigned int offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 68);

    // First dataset sent: TimelineLabelBinaryPacket
    VerifyTimelineLabelBinaryPacketData(arm::pipe::EmptyOptional(), entityName, readableData, offset);

    // Second dataset sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               entityGuid,
                                               arm::pipe::EmptyOptional(),
                                               labelTypeGuid,
                                               readableData,
                                               offset);

    // Mark the buffer as read
    mockBufferManager.MarkRead(readableBuffer);
}

TEST_CASE("SendWellKnownLabelsAndEventClassesTest")
{
    MockBufferManager mockBufferManager(1024);
    armnn::ArmNNProfilingServiceInitialiser initialiser;
    ProfilingService profilingService(arm::pipe::MAX_ARMNN_COUNTER,
                                      initialiser,
                                      arm::pipe::ARMNN_SOFTWARE_INFO,
                                      arm::pipe::ARMNN_SOFTWARE_VERSION,
                                      arm::pipe::ARMNN_HARDWARE_VERSION);
    SendTimelinePacket sendTimelinePacket(mockBufferManager);

    CHECK_NOTHROW(TimelineUtilityMethods::SendWellKnownLabelsAndEventClasses(sendTimelinePacket));

    // Get the readable buffer
    auto readableBuffer = mockBufferManager.GetReadableBuffer();
    CHECK(readableBuffer != nullptr);
    unsigned int size = readableBuffer->GetSize();
    CHECK(size == 460);
    const unsigned char* readableData = readableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    // Utils
    unsigned int offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 452);

    // First "well-known" label: NAME
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::NAME_GUID,
                                        LabelsAndEventClasses::NAME_LABEL,
                                        readableData,
                                        offset);

    // Second "well-known" label: TYPE
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::TYPE_GUID,
                                        LabelsAndEventClasses::TYPE_LABEL,
                                        readableData,
                                        offset);

    // Third "well-known" label: INDEX
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::INDEX_GUID,
                                        LabelsAndEventClasses::INDEX_LABEL,
                                        readableData,
                                        offset);

    // Forth "well-known" label: BACKENDID
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::BACKENDID_GUID,
                                        LabelsAndEventClasses::BACKENDID_LABEL,
                                        readableData,
                                        offset);

    // Fifth "well-known" label: CHILD
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::CHILD_GUID,
                                        LabelsAndEventClasses::CHILD_LABEL,
                                        readableData,
                                        offset);

    // Sixth "well-known" label: EXECUTION_OF
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::EXECUTION_OF_GUID,
                                        LabelsAndEventClasses::EXECUTION_OF_LABEL,
                                        readableData,
                                        offset);

    // Seventh "well-known" label: PROCESS_ID_LABEL
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::PROCESS_ID_GUID,
                                        LabelsAndEventClasses::PROCESS_ID_LABEL,
                                        readableData,
                                        offset);

    // Well-known types
    // Layer
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::LAYER_GUID,
                                        LabelsAndEventClasses::LAYER,
                                        readableData,
                                        offset);

    // Workload
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::WORKLOAD_GUID,
                                        LabelsAndEventClasses::WORKLOAD,
                                        readableData,
                                        offset);

    // Network
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::NETWORK_GUID,
                                        LabelsAndEventClasses::NETWORK,
                                        readableData,
                                        offset);

    // Connection
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::CONNECTION_GUID,
                                        LabelsAndEventClasses::CONNECTION,
                                        readableData,
                                        offset);

    // Inference
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::INFERENCE_GUID,
                                        LabelsAndEventClasses::INFERENCE,
                                        readableData,
                                        offset);

    // Workload Execution
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                        LabelsAndEventClasses::WORKLOAD_EXECUTION,
                                        readableData,
                                        offset);

    // First "well-known" event class: START OF LIFE
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS_NAME_GUID,
                                        LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS_NAME,
                                        readableData,
                                        offset);

    VerifyTimelineEventClassBinaryPacketData(LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                             LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS_NAME_GUID,
                                             readableData,
                                             offset);

    // Second "well-known" event class: END OF LIFE
    VerifyTimelineLabelBinaryPacketData(LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS_NAME_GUID,
                                        LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS_NAME,
                                        readableData,
                                        offset);

    VerifyTimelineEventClassBinaryPacketData(LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                             LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS_NAME_GUID,
                                             readableData,
                                             offset);

    // Mark the buffer as read
    mockBufferManager.MarkRead(readableBuffer);
}

TEST_CASE("CreateNamedTypedChildEntityTest")
{
    MockBufferManager mockBufferManager(1024);
    armnn::ArmNNProfilingServiceInitialiser initialiser;
    ProfilingService profilingService(arm::pipe::MAX_ARMNN_COUNTER,
                                      initialiser,
                                      arm::pipe::ARMNN_SOFTWARE_INFO,
                                      arm::pipe::ARMNN_SOFTWARE_VERSION,
                                      arm::pipe::ARMNN_HARDWARE_VERSION);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);

    ProfilingDynamicGuid childEntityGuid(0);
    ProfilingGuid parentEntityGuid(123);
    const std::string entityName = "some entity";
    const std::string entityType = "some type";

    // Generate first guid to ensure that the named typed entity guid is not 0 on local single test.
    profilingService.NextGuid();

    CHECK_THROWS_AS(timelineUtilityMethods.CreateNamedTypedChildEntity(parentEntityGuid, "", entityType),
                      arm::pipe::InvalidArgumentException);
    CHECK_THROWS_AS(timelineUtilityMethods.CreateNamedTypedChildEntity(parentEntityGuid, entityName, ""),
                    arm::pipe::InvalidArgumentException);
    CHECK_THROWS_AS(timelineUtilityMethods.CreateNamedTypedChildEntity(
        childEntityGuid, parentEntityGuid, "", entityType), arm::pipe::InvalidArgumentException);
    CHECK_THROWS_AS(timelineUtilityMethods.CreateNamedTypedChildEntity(
        childEntityGuid, parentEntityGuid, entityName, ""), arm::pipe::InvalidArgumentException);

    CHECK_NOTHROW(childEntityGuid = timelineUtilityMethods.CreateNamedTypedChildEntity(parentEntityGuid,
                                                                                              entityName,
                                                                                              entityType));
    CHECK(childEntityGuid != ProfilingGuid(0));

    // Commit all packets at onceTimelineUtilityMethodsTests.cpp
    timelineUtilityMethods.Commit();

    // Get the readable buffer
    auto readableBuffer = mockBufferManager.GetReadableBuffer();
    CHECK(readableBuffer != nullptr);
    unsigned int size = readableBuffer->GetSize();
    CHECK(size == 196);
    const unsigned char* readableData = readableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    // Utils
    unsigned int offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 188);

    // First dataset sent: TimelineEntityBinaryPacket
    VerifyTimelineEntityBinaryPacketData(arm::pipe::EmptyOptional(), readableData, offset);

    // Second dataset sent: TimelineLabelBinaryPacket
    VerifyTimelineLabelBinaryPacketData(arm::pipe::EmptyOptional(), entityName, readableData, offset);

    // Third dataset sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Fifth dataset sent: TimelineLabelBinaryPacket
    VerifyTimelineLabelBinaryPacketData(arm::pipe::EmptyOptional(), entityType, readableData, offset);

    // Sixth dataset sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);


    // Eighth dataset sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               parentEntityGuid,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               readableData,
                                               offset);

    // Mark the buffer as read
    mockBufferManager.MarkRead(readableBuffer);
}

TEST_CASE("DeclareLabelTest")
{
    MockBufferManager mockBufferManager(1024);
    armnn::ArmNNProfilingServiceInitialiser initialiser;
    ProfilingService profilingService(arm::pipe::MAX_ARMNN_COUNTER,
                                      initialiser,
                                      arm::pipe::ARMNN_SOFTWARE_INFO,
                                      arm::pipe::ARMNN_SOFTWARE_VERSION,
                                      arm::pipe::ARMNN_HARDWARE_VERSION);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);

    // Generate first guid to ensure that the named typed entity guid is not 0 on local single test.
    profilingService.NextGuid();

    // Try declaring an invalid (empty) label
    CHECK_THROWS_AS(timelineUtilityMethods.DeclareLabel(""), arm::pipe::InvalidArgumentException);

    // Try declaring an invalid (wrong SWTrace format) label
    CHECK_THROWS_AS(timelineUtilityMethods.DeclareLabel("inv@lid lab€l"), arm::pipe::ProfilingException);

    // Declare a valid label
    const std::string labelName = "valid label";
    ProfilingGuid labelGuid = 0;
    CHECK_NOTHROW(labelGuid = timelineUtilityMethods.DeclareLabel(labelName));
    CHECK(labelGuid != ProfilingGuid(0));

    // Try adding the same label as before
    ProfilingGuid newLabelGuid = 0;
    CHECK_NOTHROW(newLabelGuid = timelineUtilityMethods.DeclareLabel(labelName));
    CHECK(newLabelGuid != ProfilingGuid(0));
    CHECK(newLabelGuid == labelGuid);
}

TEST_CASE("CreateNameTypeEntityInvalidTest")
{
    MockBufferManager mockBufferManager(1024);
    armnn::ArmNNProfilingServiceInitialiser initialiser;
    ProfilingService profilingService(arm::pipe::MAX_ARMNN_COUNTER,
                                      initialiser,
                                      arm::pipe::ARMNN_SOFTWARE_INFO,
                                      arm::pipe::ARMNN_SOFTWARE_VERSION,
                                      arm::pipe::ARMNN_HARDWARE_VERSION);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);

    // Invalid name
    CHECK_THROWS_AS(timelineUtilityMethods.CreateNamedTypedEntity("", "Type"), arm::pipe::InvalidArgumentException);

    // Invalid type
    CHECK_THROWS_AS(timelineUtilityMethods.CreateNamedTypedEntity("Name", ""), arm::pipe::InvalidArgumentException);

    ProfilingDynamicGuid guid = profilingService.NextGuid();

    // CreatedNamedTypedEntity with Guid - Invalid name
    CHECK_THROWS_AS(timelineUtilityMethods.CreateNamedTypedEntity(guid, "", "Type"),
                    arm::pipe::InvalidArgumentException);

    // CreatedNamedTypedEntity with Guid - Invalid type
    CHECK_THROWS_AS(timelineUtilityMethods.CreateNamedTypedEntity(guid, "Name", ""),
                    arm::pipe::InvalidArgumentException);

}

TEST_CASE("CreateNameTypeEntityTest")
{
    MockBufferManager mockBufferManager(1024);
    armnn::ArmNNProfilingServiceInitialiser initialiser;
    ProfilingService profilingService(arm::pipe::MAX_ARMNN_COUNTER,
                                      initialiser,
                                      arm::pipe::ARMNN_SOFTWARE_INFO,
                                      arm::pipe::ARMNN_SOFTWARE_VERSION,
                                      arm::pipe::ARMNN_HARDWARE_VERSION);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);

    const std::string entityName = "Entity0";
    const std::string entityType = "Type0";

    // Generate first guid to ensure that the named typed entity guid is not 0 on local single test.
    profilingService.NextGuid();

    ProfilingDynamicGuid guid = timelineUtilityMethods.CreateNamedTypedEntity(entityName, entityType);
    CHECK(guid != ProfilingGuid(0));

    // Commit all packets at once
    timelineUtilityMethods.Commit();

    // Get the readable buffer
    auto readableBuffer = mockBufferManager.GetReadableBuffer();
    CHECK(readableBuffer != nullptr);
    unsigned int size = readableBuffer->GetSize();
    CHECK(size == 148);
    const unsigned char* readableData = readableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    // Utils
    unsigned int offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 140);

    // First dataset sent: TimelineEntityBinaryPacket
    VerifyTimelineEntityBinaryPacketData(guid, readableData, offset);

    // Packets for Name Entity
    // First dataset sent: TimelineLabelBinaryPacket
    VerifyTimelineLabelBinaryPacketData(arm::pipe::EmptyOptional(), entityName, readableData, offset);

    // Second dataset sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Packets for Type Entity
    // First dataset sent: TimelineLabelBinaryPacket
    VerifyTimelineLabelBinaryPacketData(arm::pipe::EmptyOptional(), entityType, readableData, offset);

    // Second dataset sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);


    // Mark the buffer as read
    mockBufferManager.MarkRead(readableBuffer);
}

TEST_CASE("RecordEventTest")
{
    MockBufferManager mockBufferManager(1024);
    armnn::ArmNNProfilingServiceInitialiser initialiser;
    ProfilingService profilingService(arm::pipe::MAX_ARMNN_COUNTER,
                                      initialiser,
                                      arm::pipe::ARMNN_SOFTWARE_INFO,
                                      arm::pipe::ARMNN_SOFTWARE_VERSION,
                                      arm::pipe::ARMNN_HARDWARE_VERSION);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);
    // Generate first guid to ensure that the named typed entity guid is not 0 on local single test.
    profilingService.NextGuid();

    ProfilingGuid entityGuid(123);
    ProfilingStaticGuid eventClassGuid(456);
    ProfilingDynamicGuid eventGuid(0);
    CHECK_NOTHROW(eventGuid = timelineUtilityMethods.RecordEvent(entityGuid, eventClassGuid));
    CHECK(eventGuid != ProfilingGuid(0));

    // Commit all packets at once
    timelineUtilityMethods.Commit();

    // Get the readable buffer
    auto readableBuffer = mockBufferManager.GetReadableBuffer();
    CHECK(readableBuffer != nullptr);
    unsigned int size = readableBuffer->GetSize();

    CHECK(size == 68 + ThreadIdSize);

    const unsigned char* readableData = readableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    // Utils
    unsigned int offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 60 + ThreadIdSize);

    // First dataset sent: TimelineEntityBinaryPacket
    VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Second dataset sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               entityGuid,
                                               eventGuid,
                                               eventClassGuid,
                                               readableData,
                                               offset);

    // Mark the buffer as read
    mockBufferManager.MarkRead(readableBuffer);
}

}
