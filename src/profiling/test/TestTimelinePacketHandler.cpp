//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestTimelinePacketHandler.hpp"
#include "IProfilingConnection.hpp"

#include <common/include/LabelsAndEventClasses.hpp>

#include <chrono>
#include <iostream>

namespace armnn
{

namespace profiling
{

std::vector<uint32_t> TestTimelinePacketHandler::GetHeadersAccepted()
{
    std::vector<uint32_t> headers;
    headers.push_back(m_DirectoryHeader); // message directory
    headers.push_back(m_MessageHeader); // message
    return headers;
}

void TestTimelinePacketHandler::HandlePacket(const arm::pipe::Packet& packet)
{
    if (packet.GetHeader() == m_DirectoryHeader)
    {
        ProcessDirectoryPacket(packet);
    }
    else if (packet.GetHeader() == m_MessageHeader)
    {
        ProcessMessagePacket(packet);
    }
    else
    {
        std::stringstream ss;
        ss << "Received a packet with unknown header [" << packet.GetHeader() << "]";
        throw armnn::Exception(ss.str());
    }
}

void TestTimelinePacketHandler::Stop()
{
    m_Connection->Close();
}

void TestTimelinePacketHandler::WaitOnInferenceCompletion(unsigned int timeout)
{
    std::unique_lock<std::mutex> lck(m_InferenceCompletedMutex);

    auto start = std::chrono::high_resolution_clock::now();
    // Here we we will go back to sleep after a spurious wake up if
    // m_InferenceCompleted is not yet true.
    if (!m_InferenceCompletedConditionVariable.wait_for(lck,
                                                        std::chrono::milliseconds(timeout),
                                                        [&]{return m_InferenceCompleted == true;}))
    {
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = finish - start;
        std::stringstream ss;
        ss << "Timed out waiting on inference completion for " << elapsed.count() << " ms";
        throw armnn::TimeoutException(ss.str());
    }
    return;
}

void TestTimelinePacketHandler::SetInferenceComplete()
{
    {   // only lock when we are updating the inference completed variable
        std::unique_lock<std::mutex> lck(m_InferenceCompletedMutex);
        m_InferenceCompleted = true;
    }
    m_InferenceCompletedConditionVariable.notify_one();
}

void TestTimelinePacketHandler::ProcessDirectoryPacket(const arm::pipe::Packet& packet)
{
    m_DirectoryDecoder(packet);
}

void TestTimelinePacketHandler::ProcessMessagePacket(const arm::pipe::Packet& packet)
{
    m_Decoder(packet);
}

// TimelineMessageDecoder functions
arm::pipe::ITimelineDecoder::TimelineStatus TimelineMessageDecoder::CreateEntity(const Entity& entity)
{
    m_TimelineModel.AddEntity(entity.m_Guid);
    return arm::pipe::ITimelineDecoder::TimelineStatus::TimelineStatus_Success;
}

arm::pipe::ITimelineDecoder::TimelineStatus TimelineMessageDecoder::CreateEventClass(
    const arm::pipe::ITimelineDecoder::EventClass& eventClass)
{
    m_TimelineModel.AddEventClass(eventClass);
    return arm::pipe::ITimelineDecoder::TimelineStatus::TimelineStatus_Success;
}

arm::pipe::ITimelineDecoder::TimelineStatus TimelineMessageDecoder::CreateEvent(
    const arm::pipe::ITimelineDecoder::Event& event)
{
    m_TimelineModel.AddEvent(event);
    return arm::pipe::ITimelineDecoder::TimelineStatus::TimelineStatus_Success;
}

arm::pipe::ITimelineDecoder::TimelineStatus TimelineMessageDecoder::CreateLabel(
    const arm::pipe::ITimelineDecoder::Label& label)
{
    m_TimelineModel.AddLabel(label);
    return arm::pipe::ITimelineDecoder::TimelineStatus::TimelineStatus_Success;
}

arm::pipe::ITimelineDecoder::TimelineStatus TimelineMessageDecoder::CreateRelationship(
    const arm::pipe::ITimelineDecoder::Relationship& relationship)
{
    m_TimelineModel.AddRelationship(relationship);
    // check to see if this is an execution link to an inference of event class end of life
    // if so the inference has completed so send out a notification...
    if (relationship.m_RelationshipType == RelationshipType::ExecutionLink &&
        m_TimelineModel.IsInferenceGuid(relationship.m_HeadGuid))
    {
        ProfilingStaticGuid attributeGuid(relationship.m_AttributeGuid);
        if (attributeGuid == armnn::profiling::LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS)
        {
            if (m_PacketHandler != nullptr)
            {
                m_PacketHandler->SetInferenceComplete();
            }
        }
    }
    return arm::pipe::ITimelineDecoder::TimelineStatus::TimelineStatus_Success;
}

} // namespace profiling

} // namespace armnn