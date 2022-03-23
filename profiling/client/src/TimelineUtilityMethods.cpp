//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ProfilingUtils.hpp"

#include <client/include/TimelineUtilityMethods.hpp>

#include <common/include/LabelsAndEventClasses.hpp>
#include <common/include/Threads.hpp>

namespace arm
{

namespace pipe
{

std::unique_ptr<TimelineUtilityMethods> TimelineUtilityMethods::GetTimelineUtils(IProfilingService& profilingService)
{
    if (profilingService.GetCurrentState() == ProfilingState::Active && profilingService.IsTimelineReportingEnabled())
    {
        std::unique_ptr<ISendTimelinePacket> sendTimelinepacket = profilingService.GetSendTimelinePacket();
        return std::make_unique<TimelineUtilityMethods>(sendTimelinepacket);
    }
    else
    {
        std::unique_ptr<TimelineUtilityMethods> empty;
        return empty;
    }
}


void TimelineUtilityMethods::SendWellKnownLabelsAndEventClasses(ISendTimelinePacket& timelinePacket)
{
    // Send the "name" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::NAME_GUID,
                                                 LabelsAndEventClasses::NAME_LABEL);

    // Send the "type" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::TYPE_GUID,
                                                 LabelsAndEventClasses::TYPE_LABEL);

    // Send the "index" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::INDEX_GUID,
                                                 LabelsAndEventClasses::INDEX_LABEL);

    // Send the "backendId" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::BACKENDID_GUID,
                                                 LabelsAndEventClasses::BACKENDID_LABEL);

    // Send the "child" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::CHILD_GUID,
                                                 LabelsAndEventClasses::CHILD_LABEL);

    // Send the "execution_of" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::EXECUTION_OF_GUID,
                                                 LabelsAndEventClasses::EXECUTION_OF_LABEL);

    // Send the "process_id" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::PROCESS_ID_GUID,
                                                 LabelsAndEventClasses::PROCESS_ID_LABEL);

    // Send the "layer" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::LAYER_GUID,
                                                 LabelsAndEventClasses::LAYER);

    // Send the "workload" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::WORKLOAD_GUID,
                                                 LabelsAndEventClasses::WORKLOAD);

    // Send the "network" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::NETWORK_GUID,
                                                 LabelsAndEventClasses::NETWORK);

    // Send the "connection" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::CONNECTION_GUID,
                                                 LabelsAndEventClasses::CONNECTION);

    // Send the "inference" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::INFERENCE_GUID,
                                                 LabelsAndEventClasses::INFERENCE);

    // Send the "workload_execution" label, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                                 LabelsAndEventClasses::WORKLOAD_EXECUTION);

    // Send the "start of life" event class, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS_NAME_GUID,
                                                 LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS_NAME);
    timelinePacket.SendTimelineEventClassBinaryPacket(LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                                      LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS_NAME_GUID);

    // Send the "end of life" event class, this call throws in case of error
    timelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS_NAME_GUID,
                                                 LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS_NAME);
    timelinePacket.SendTimelineEventClassBinaryPacket(LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                                      LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS_NAME_GUID);

    timelinePacket.Commit();
}

ProfilingDynamicGuid TimelineUtilityMethods::CreateNamedTypedEntity(const std::string& name, const std::string& type)
{
    // Check that the entity name is valid
    if (name.empty())
    {
        throw arm::pipe::InvalidArgumentException("Invalid entity name, the entity name cannot be empty");
    }

    // Check that the entity type is valid
    if (type.empty())
    {
        throw arm::pipe::InvalidArgumentException("Invalid entity type, the entity type cannot be empty");
    }

    // Generate dynamic GUID of the entity
    ProfilingDynamicGuid entityGuid = IProfilingService::GetNextGuid();

    CreateNamedTypedEntity(entityGuid, name, type);

    return entityGuid;
}

void TimelineUtilityMethods::CreateNamedTypedEntity(ProfilingGuid entityGuid,
                                                    const std::string& name,
                                                    const std::string& type)
{
    // Check that the entity name is valid
    if (name.empty())
    {
        throw arm::pipe::InvalidArgumentException("Invalid entity name, the entity name cannot be empty");
    }

    // Check that the entity type is valid
    if (type.empty())
    {
        throw arm::pipe::InvalidArgumentException("Invalid entity type, the entity type cannot be empty");
    }

    // Send Entity Binary Packet of the entity to the external profiling service
    m_SendTimelinePacket->SendTimelineEntityBinaryPacket(entityGuid);

    // Create name entity and send the relationship of the entity with the given name
    NameEntity(entityGuid, name);

    // Create type entity and send the relationship of the entity with the given type
    TypeEntity(entityGuid, type);
}

void TimelineUtilityMethods::CreateNamedTypedEntity(ProfilingGuid entityGuid,
                                                    const std::string& name,
                                                    ProfilingStaticGuid typeGuid)
{
    // Check that the entity name is valid
    if (name.empty())
    {
        throw arm::pipe::InvalidArgumentException("Invalid entity name, the entity name cannot be empty");
    }

    // Send Entity Binary Packet of the entity to the external profiling service
    m_SendTimelinePacket->SendTimelineEntityBinaryPacket(entityGuid);

    // Create name entity and send the relationship of the entity with the given name
    NameEntity(entityGuid, name);

    // Create type entity and send the relationship of the entity with the given type
    MarkEntityWithType(entityGuid, typeGuid);
}

ProfilingStaticGuid TimelineUtilityMethods::DeclareLabel(const std::string& labelName)
{
    // Check that the label name is valid
    if (labelName.empty())
    {
        // The label name is invalid
        throw arm::pipe::InvalidArgumentException("Invalid label name, the label name cannot be empty");
    }

    // Generate a static GUID for the given label name
    ProfilingStaticGuid labelGuid = IProfilingService::GetStaticId(labelName);

    // Send the new label to the external profiling service, this call throws in case of error
    m_SendTimelinePacket->SendTimelineLabelBinaryPacket(labelGuid, labelName);

    return labelGuid;
}

void TimelineUtilityMethods::MarkEntityWithLabel(ProfilingGuid entityGuid,
                                                 const std::string& labelName,
                                                 ProfilingStaticGuid labelTypeGuid)
{
    // Check that the label name is valid
    if (labelName.empty())
    {
        // The label name is invalid
        throw arm::pipe::InvalidArgumentException("Invalid entity name, the entity name cannot be empty");
    }

    // Declare a label with the label's name, this call throws in case of error
    ProfilingStaticGuid labelGuid = DeclareLabel(labelName);

    // Generate a GUID for the label relationship
    ProfilingDynamicGuid relationshipGuid = IProfilingService::GetNextGuid();

    // Send the new label link to the external profiling service, this call throws in case of error
    m_SendTimelinePacket->SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                                               relationshipGuid,
                                                               entityGuid,
                                                               labelGuid,
                                                               labelTypeGuid);
}

void TimelineUtilityMethods::MarkEntityWithType(ProfilingGuid entityGuid,
                                                ProfilingStaticGuid typeNameGuid)
{
    // Generate a GUID for the label relationship
    ProfilingDynamicGuid relationshipGuid = IProfilingService::GetNextGuid();

    // Send the new label link to the external profiling service, this call throws in case of error
    m_SendTimelinePacket->SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                                               relationshipGuid,
                                                               entityGuid,
                                                               typeNameGuid,
                                                               LabelsAndEventClasses::TYPE_GUID);
}

void TimelineUtilityMethods::NameEntity(ProfilingGuid entityGuid, const std::string& name)
{
    MarkEntityWithLabel(entityGuid, name, LabelsAndEventClasses::NAME_GUID);
}

void TimelineUtilityMethods::TypeEntity(ProfilingGuid entityGuid, const std::string& type)
{
    MarkEntityWithLabel(entityGuid, type, LabelsAndEventClasses::TYPE_GUID);
}

ProfilingDynamicGuid TimelineUtilityMethods::CreateNamedTypedChildEntity(ProfilingGuid parentEntityGuid,
                                                                         const std::string& entityName,
                                                                         const std::string& entityType)
{
    // Check that the entity name is valid
    if (entityName.empty())
    {
        // The entity name is invalid
        throw arm::pipe::InvalidArgumentException("Invalid entity name, the entity name cannot be empty");
    }

    // Check that the entity type is valid
    if (entityType.empty())
    {
        // The entity type is invalid
        throw arm::pipe::InvalidArgumentException("Invalid entity type, the entity type cannot be empty");
    }

    // Create a named type entity from the given name and type, this call throws in case of error
    ProfilingDynamicGuid childEntityGuid = CreateNamedTypedEntity(entityName, entityType);

    // Generate a GUID for the retention link relationship
    ProfilingDynamicGuid retentionLinkGuid = IProfilingService::GetNextGuid();

    // Send the new retention link to the external profiling service, this call throws in case of error
    m_SendTimelinePacket->SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType::RetentionLink,
                                                               retentionLinkGuid,
                                                               parentEntityGuid,
                                                               childEntityGuid,
                                                               LabelsAndEventClasses::EMPTY_GUID);

    return childEntityGuid;
}

void TimelineUtilityMethods::CreateNamedTypedChildEntity(ProfilingGuid childEntityGuid,
                                                         ProfilingGuid parentEntityGuid,
                                                         const std::string& entityName,
                                                         const std::string& entityType)
{
    // Check that the entity name is valid
    if (entityName.empty())
    {
        // The entity name is invalid
        throw arm::pipe::InvalidArgumentException("Invalid entity name, the entity name cannot be empty");
    }

    // Check that the entity type is valid
    if (entityType.empty())
    {
        // The entity type is invalid
        throw arm::pipe::InvalidArgumentException("Invalid entity type, the entity type cannot be empty");
    }

    // Create a named type entity from the given guid, name and type, this call throws in case of error
    CreateNamedTypedEntity(childEntityGuid, entityName, entityType);

    // Generate a GUID for the retention link relationship
    ProfilingDynamicGuid retentionLinkGuid = IProfilingService::GetNextGuid();

    // Send the new retention link to the external profiling service, this call throws in case of error
    m_SendTimelinePacket->SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType::RetentionLink,
                                                               retentionLinkGuid,
                                                               parentEntityGuid,
                                                               childEntityGuid,
                                                               LabelsAndEventClasses::CHILD_GUID);
}

void TimelineUtilityMethods::CreateNamedTypedChildEntity(ProfilingGuid childEntityGuid,
                                                         ProfilingGuid parentEntityGuid,
                                                         const std::string& entityName,
                                                         ProfilingStaticGuid typeGuid)
{
    // Check that the entity name is valid
    if (entityName.empty())
    {
        // The entity name is invalid
        throw arm::pipe::InvalidArgumentException("Invalid entity name, the entity name cannot be empty");
    }

    // Create a named type entity from the given guid, name and type, this call throws in case of error
    CreateNamedTypedEntity(childEntityGuid, entityName, typeGuid);

    // Generate a GUID for the retention link relationship
    ProfilingDynamicGuid retentionLinkGuid = IProfilingService::GetNextGuid();

    // Send the new retention link to the external profiling service, this call throws in case of error
    m_SendTimelinePacket->SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType::RetentionLink,
                                                               retentionLinkGuid,
                                                               parentEntityGuid,
                                                               childEntityGuid,
                                                               LabelsAndEventClasses::CHILD_GUID);
}

ProfilingDynamicGuid TimelineUtilityMethods::CreateRelationship(ProfilingRelationshipType relationshipType,
                                                                ProfilingGuid headGuid,
                                                                ProfilingGuid tailGuid,
                                                                ProfilingGuid relationshipCategory)
{
    // Generate a GUID for the relationship
    ProfilingDynamicGuid relationshipGuid = IProfilingService::GetNextGuid();

    // Send the new retention link to the external profiling service, this call throws in case of error
    m_SendTimelinePacket->SendTimelineRelationshipBinaryPacket(relationshipType,
                                                               relationshipGuid,
                                                               headGuid,
                                                               tailGuid,
                                                               relationshipCategory);
    return relationshipGuid;
}

ProfilingDynamicGuid TimelineUtilityMethods::CreateConnectionRelationship(ProfilingRelationshipType relationshipType,
                                                                          ProfilingGuid headGuid,
                                                                          ProfilingGuid tailGuid)
{
    // Generate a GUID for the relationship
    ProfilingDynamicGuid relationshipGuid = IProfilingService::GetNextGuid();

    // Send the new retention link to the external profiling service, this call throws in case of error
    m_SendTimelinePacket->SendTimelineRelationshipBinaryPacket(relationshipType,
                                                               relationshipGuid,
                                                               headGuid,
                                                               tailGuid,
                                                               LabelsAndEventClasses::CONNECTION_GUID);
    return relationshipGuid;
}

void TimelineUtilityMethods::CreateTypedEntity(ProfilingGuid entityGuid, ProfilingStaticGuid entityTypeGuid)
{
    // Send Entity Binary Packet of the entity to the external profiling service
    m_SendTimelinePacket->SendTimelineEntityBinaryPacket(entityGuid);

    // Create type entity and send the relationship of the entity with the given type
    MarkEntityWithType(entityGuid, entityTypeGuid);
}

ProfilingDynamicGuid TimelineUtilityMethods::RecordEvent(ProfilingGuid entityGuid, ProfilingStaticGuid eventClassGuid)
{
    // Take a timestamp
    uint64_t timestamp = GetTimestamp();

    // Get the thread id
    int threadId = arm::pipe::GetCurrentThreadId();

    // Generate a GUID for the event
    ProfilingDynamicGuid eventGuid = IProfilingService::GetNextGuid();

    // Send the new timeline event to the external profiling service, this call throws in case of error
    m_SendTimelinePacket->SendTimelineEventBinaryPacket(timestamp, threadId, eventGuid);

    // Generate a GUID for the execution link
    ProfilingDynamicGuid executionLinkId = IProfilingService::GetNextGuid();

    // Send the new execution link to the external profiling service, this call throws in case of error
    m_SendTimelinePacket->SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType::ExecutionLink,
                                                               executionLinkId,
                                                               entityGuid,
                                                               eventGuid,
                                                               eventClassGuid);

    return eventGuid;
}

ProfilingDynamicGuid TimelineUtilityMethods::RecordWorkloadInferenceAndStartOfLifeEvent(ProfilingGuid workloadGuid,
                                                                                        ProfilingGuid inferenceGuid)
{
    ProfilingDynamicGuid workloadInferenceGuid = IProfilingService::GetNextGuid();
    CreateTypedEntity(workloadInferenceGuid, LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID);
    CreateRelationship(ProfilingRelationshipType::RetentionLink,
                       inferenceGuid,
                       workloadInferenceGuid,
                       LabelsAndEventClasses::CHILD_GUID);
    CreateRelationship(ProfilingRelationshipType::RetentionLink,
                       workloadGuid,
                       workloadInferenceGuid,
                       LabelsAndEventClasses::EXECUTION_OF_GUID);
    RecordEvent(workloadInferenceGuid, LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS);
    return workloadInferenceGuid;
}

void TimelineUtilityMethods::RecordEndOfLifeEvent(ProfilingGuid entityGuid)
{
    RecordEvent(entityGuid, LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS);
}

} // namespace pipe

} // namespace arm
