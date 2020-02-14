//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingMocks.hpp"
#include "ProfilingTestUtils.hpp"

#include <SendTimelinePacket.hpp>
#include <TimelineUtilityMethods.hpp>
#include <LabelsAndEventClasses.hpp>
#include <ProfilingService.hpp>

#include <memory>

#include <boost/test/unit_test.hpp>

using namespace armnn;
using namespace armnn::profiling;

BOOST_AUTO_TEST_SUITE(TimelineUtilityMethodsTests)

BOOST_AUTO_TEST_CASE(CreateTypedLabelTest)
{
    MockBufferManager mockBufferManager(1024);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);

    // Generate first guid to ensure that the named typed entity guid is not 0 on local single test.
    ProfilingService::Instance().NextGuid();

    ProfilingGuid entityGuid(123);
    const std::string entityName = "some entity";
    ProfilingStaticGuid labelTypeGuid(456);

    BOOST_CHECK_NO_THROW(timelineUtilityMethods.MarkEntityWithLabel(entityGuid, entityName, labelTypeGuid));

    // Commit all packets at once
    timelineUtilityMethods.Commit();

    // Get the readable buffer
    auto readableBuffer = mockBufferManager.GetReadableBuffer();
    BOOST_CHECK(readableBuffer != nullptr);
    unsigned int size = readableBuffer->GetSize();
    BOOST_CHECK(size == 116);
    const unsigned char* readableData = readableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    // Utils
    unsigned int offset = 0;

    // First packet sent: TimelineLabelBinaryPacket
    VerifyTimelineLabelBinaryPacket(EmptyOptional(), entityName, readableData, offset);

    // Second packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           entityGuid,
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Third packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           labelTypeGuid,
                                           readableData,
                                           offset);

    // Mark the buffer as read
    mockBufferManager.MarkRead(readableBuffer);
}

BOOST_AUTO_TEST_CASE(SendWellKnownLabelsAndEventClassesTest)
{
    MockBufferManager mockBufferManager(1024);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);

    BOOST_CHECK_NO_THROW(timelineUtilityMethods.SendWellKnownLabelsAndEventClasses());

    // Commit all packets at once
    timelineUtilityMethods.Commit();

    // Get the readable buffer
    auto readableBuffer = mockBufferManager.GetReadableBuffer();
    BOOST_CHECK(readableBuffer != nullptr);
    unsigned int size = readableBuffer->GetSize();
    BOOST_TEST(size == 388);
    const unsigned char* readableData = readableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    // Utils
    unsigned int offset = 0;

    // First "well-known" label: NAME
    VerifyTimelineLabelBinaryPacket(LabelsAndEventClasses::NAME_GUID,
                                    LabelsAndEventClasses::NAME_LABEL,
                                    readableData,
                                    offset);

    // Second "well-known" label: TYPE
    VerifyTimelineLabelBinaryPacket(LabelsAndEventClasses::TYPE_GUID,
                                    LabelsAndEventClasses::TYPE_LABEL,
                                    readableData,
                                    offset);

    // Third "well-known" label: INDEX
    VerifyTimelineLabelBinaryPacket(LabelsAndEventClasses::INDEX_GUID,
                                    LabelsAndEventClasses::INDEX_LABEL,
                                    readableData,
                                    offset);

    // Forth "well-known" label: BACKENDID
    VerifyTimelineLabelBinaryPacket(LabelsAndEventClasses::BACKENDID_GUID,
                                    LabelsAndEventClasses::BACKENDID_LABEL,
                                    readableData,
                                    offset);

    // Well-known types
    // Layer
    VerifyTimelineLabelBinaryPacket(LabelsAndEventClasses::LAYER_GUID,
                                    LabelsAndEventClasses::LAYER,
                                    readableData,
                                    offset);

    // Workload
    VerifyTimelineLabelBinaryPacket(LabelsAndEventClasses::WORKLOAD_GUID,
                                    LabelsAndEventClasses::WORKLOAD,
                                    readableData,
                                    offset);

    // Network
    VerifyTimelineLabelBinaryPacket(LabelsAndEventClasses::NETWORK_GUID,
                                    LabelsAndEventClasses::NETWORK,
                                    readableData,
                                    offset);

    // Connection
    VerifyTimelineLabelBinaryPacket(LabelsAndEventClasses::CONNECTION_GUID,
                                    LabelsAndEventClasses::CONNECTION,
                                    readableData,
                                    offset);
    // Inference
    VerifyTimelineLabelBinaryPacket(LabelsAndEventClasses::INFERENCE_GUID,
                                    LabelsAndEventClasses::INFERENCE,
                                    readableData,
                                    offset);
    // Workload Execution
    VerifyTimelineLabelBinaryPacket(LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                    LabelsAndEventClasses::WORKLOAD_EXECUTION,
                                    readableData,
                                    offset);

    // First "well-known" event class: START OF LIFE
    VerifyTimelineEventClassBinaryPacket(LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                         readableData,
                                         offset);

    // Second "well-known" event class: END OF LIFE
    VerifyTimelineEventClassBinaryPacket(LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                         readableData,
                                         offset);

    // Mark the buffer as read
    mockBufferManager.MarkRead(readableBuffer);
}

BOOST_AUTO_TEST_CASE(CreateNamedTypedChildEntityTest)
{
    MockBufferManager mockBufferManager(1024);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);

    ProfilingDynamicGuid childEntityGuid(0);
    ProfilingGuid parentEntityGuid(123);
    const std::string entityName = "some entity";
    const std::string entityType = "some type";

    // Generate first guid to ensure that the named typed entity guid is not 0 on local single test.
    ProfilingService::Instance().NextGuid();

    BOOST_CHECK_THROW(timelineUtilityMethods.CreateNamedTypedChildEntity(parentEntityGuid, "", entityType),
                      InvalidArgumentException);
    BOOST_CHECK_THROW(timelineUtilityMethods.CreateNamedTypedChildEntity(parentEntityGuid, entityName, ""),
                      InvalidArgumentException);
    BOOST_CHECK_THROW(timelineUtilityMethods.CreateNamedTypedChildEntity(
        childEntityGuid, parentEntityGuid, "", entityType), InvalidArgumentException);
    BOOST_CHECK_THROW(timelineUtilityMethods.CreateNamedTypedChildEntity(
        childEntityGuid, parentEntityGuid, entityName, ""), InvalidArgumentException);

    BOOST_CHECK_NO_THROW(childEntityGuid = timelineUtilityMethods.CreateNamedTypedChildEntity(parentEntityGuid,
                                                                                              entityName,
                                                                                              entityType));
    BOOST_CHECK(childEntityGuid != ProfilingGuid(0));

    // Commit all packets at once
    timelineUtilityMethods.Commit();

    // Get the readable buffer
    auto readableBuffer = mockBufferManager.GetReadableBuffer();
    BOOST_CHECK(readableBuffer != nullptr);
    unsigned int size = readableBuffer->GetSize();
    BOOST_CHECK(size == 292);
    const unsigned char* readableData = readableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    // Utils
    unsigned int offset = 0;

    // First packet sent: TimelineEntityBinaryPacket
    VerifyTimelineEntityBinaryPacket(EmptyOptional(), readableData, offset);

    // Second packet sent: TimelineLabelBinaryPacket
    VerifyTimelineLabelBinaryPacket(EmptyOptional(), entityName, readableData, offset);

    // Third packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Fourth packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::NAME_GUID,
                                           readableData,
                                           offset);

    // Fifth packet sent: TimelineLabelBinaryPacket
    VerifyTimelineLabelBinaryPacket(EmptyOptional(), entityType, readableData, offset);

    // Sixth packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Seventh packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::TYPE_GUID,
                                           readableData,
                                           offset);

    // Eighth packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::RetentionLink,
                                           EmptyOptional(),
                                           parentEntityGuid,
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Mark the buffer as read
    mockBufferManager.MarkRead(readableBuffer);
}

BOOST_AUTO_TEST_CASE(DeclareLabelTest)
{
    MockBufferManager mockBufferManager(1024);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);

    // Generate first guid to ensure that the named typed entity guid is not 0 on local single test.
    ProfilingService::Instance().NextGuid();

    // Try declaring an invalid (empty) label
    BOOST_CHECK_THROW(timelineUtilityMethods.DeclareLabel(""), InvalidArgumentException);

    // Try declaring an invalid (wrong SWTrace format) label
    BOOST_CHECK_THROW(timelineUtilityMethods.DeclareLabel("inv@lid lab€l"), RuntimeException);

    // Declare a valid label
    const std::string labelName = "valid label";
    ProfilingGuid labelGuid = 0;
    BOOST_CHECK_NO_THROW(labelGuid = timelineUtilityMethods.DeclareLabel(labelName));
    BOOST_CHECK(labelGuid != ProfilingGuid(0));

    // Try adding the same label as before
    ProfilingGuid newLabelGuid = 0;
    BOOST_CHECK_NO_THROW(newLabelGuid = timelineUtilityMethods.DeclareLabel(labelName));
    BOOST_CHECK(newLabelGuid != ProfilingGuid(0));
    BOOST_CHECK(newLabelGuid == labelGuid);
}

BOOST_AUTO_TEST_CASE(CreateNameTypeEntityInvalidTest)
{
    MockBufferManager mockBufferManager(1024);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);

    // Invalid name
    BOOST_CHECK_THROW(timelineUtilityMethods.CreateNamedTypedEntity("", "Type"), InvalidArgumentException);

    // Invalid type
    BOOST_CHECK_THROW(timelineUtilityMethods.CreateNamedTypedEntity("Name", ""), InvalidArgumentException);

    ProfilingDynamicGuid guid = ProfilingService::Instance().NextGuid();

    // CreatedNamedTypedEntity with Guid - Invalid name
    BOOST_CHECK_THROW(timelineUtilityMethods.CreateNamedTypedEntity(guid, "", "Type"),
                      InvalidArgumentException);

    // CreatedNamedTypedEntity with Guid - Invalid type
    BOOST_CHECK_THROW(timelineUtilityMethods.CreateNamedTypedEntity(guid, "Name", ""),
                      InvalidArgumentException);

}

BOOST_AUTO_TEST_CASE(CreateNameTypeEntityTest)
{
    MockBufferManager mockBufferManager(1024);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);

    const std::string entityName = "Entity0";
    const std::string entityType = "Type0";

    // Generate first guid to ensure that the named typed entity guid is not 0 on local single test.
    ProfilingService::Instance().NextGuid();

    ProfilingDynamicGuid guid = timelineUtilityMethods.CreateNamedTypedEntity(entityName, entityType);
    BOOST_CHECK(guid != ProfilingGuid(0));

    // Commit all packets at once
    timelineUtilityMethods.Commit();

    // Get the readable buffer
    auto readableBuffer = mockBufferManager.GetReadableBuffer();
    BOOST_CHECK(readableBuffer != nullptr);
    unsigned int size = readableBuffer->GetSize();
    BOOST_CHECK(size == 244);
    const unsigned char* readableData = readableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    // Utils
    unsigned int offset = 0;

    // First packet sent: TimelineEntityBinaryPacket
    VerifyTimelineEntityBinaryPacket(guid, readableData, offset);

    // Packets for Name Entity
    // First packet sent: TimelineLabelBinaryPacket
    VerifyTimelineLabelBinaryPacket(EmptyOptional(), entityName, readableData, offset);

    // Second packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Third packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::NAME_GUID,
                                           readableData,
                                           offset);

    // Packets for Type Entity
    // First packet sent: TimelineLabelBinaryPacket
    VerifyTimelineLabelBinaryPacket(EmptyOptional(), entityType, readableData, offset);

    // Second packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Third packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::TYPE_GUID,
                                           readableData,
                                           offset);

    // Mark the buffer as read
    mockBufferManager.MarkRead(readableBuffer);
}

BOOST_AUTO_TEST_CASE(RecordEventTest)
{
    MockBufferManager mockBufferManager(1024);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = std::make_unique<SendTimelinePacket>(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);
    // Generate first guid to ensure that the named typed entity guid is not 0 on local single test.
    ProfilingService::Instance().NextGuid();

    ProfilingGuid entityGuid(123);
    ProfilingStaticGuid eventClassGuid(456);
    ProfilingDynamicGuid eventGuid(0);
    BOOST_CHECK_NO_THROW(eventGuid = timelineUtilityMethods.RecordEvent(entityGuid, eventClassGuid));
    BOOST_CHECK(eventGuid != ProfilingGuid(0));

    // Commit all packets at once
    timelineUtilityMethods.Commit();

    // Get the readable buffer
    auto readableBuffer = mockBufferManager.GetReadableBuffer();
    BOOST_CHECK(readableBuffer != nullptr);
    unsigned int size = readableBuffer->GetSize();
    BOOST_CHECK(size == 116);
    const unsigned char* readableData = readableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    // Utils
    unsigned int offset = 0;

    // First packet sent: TimelineEntityBinaryPacket
    VerifyTimelineEventBinaryPacket(EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Second packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::ExecutionLink,
                                           EmptyOptional(),
                                           entityGuid,
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Third packet sent: TimelineRelationshipBinaryPacket
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::DataLink,
                                           EmptyOptional(),
                                           eventGuid,
                                           eventClassGuid,
                                           readableData,
                                           offset);

    // Mark the buffer as read
    mockBufferManager.MarkRead(readableBuffer);
}

BOOST_AUTO_TEST_SUITE_END()
