//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestTimelinePacketHandler.hpp"
#include "IProfilingConnection.hpp"

#include <armnn/utility/IgnoreUnused.hpp>

#include <chrono>
#include <iostream>
#include <sstream>

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

void TestTimelinePacketHandler::HandlePacket(const Packet& packet)
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

void TestTimelinePacketHandler::ProcessDirectoryPacket(const Packet& packet)
{
    m_DirectoryDecoder(packet);
}

void TestTimelinePacketHandler::ProcessMessagePacket(const Packet& packet)
{
    m_Decoder(packet);
}

// TimelineMessageDecoder functions
ITimelineDecoder::TimelineStatus TimelineMessageDecoder::CreateEntity(const Entity& entity)
{
    m_TimelineModel.AddEntity(entity.m_Guid);
    return ITimelineDecoder::TimelineStatus::TimelineStatus_Success;
}

ITimelineDecoder::TimelineStatus TimelineMessageDecoder::CreateEventClass(
    const ITimelineDecoder::EventClass& eventClass)
{
    // for the moment terminate the run here so we can get this code
    // onto master prior to a major re-organisation
    if (m_PacketHandler != nullptr)
    {
        m_PacketHandler->SetInferenceComplete();
    }
    IgnoreUnused(eventClass);
    return ITimelineDecoder::TimelineStatus::TimelineStatus_Success;
}

ITimelineDecoder::TimelineStatus TimelineMessageDecoder::CreateEvent(const ITimelineDecoder::Event& event)
{
    IgnoreUnused(event);
    return ITimelineDecoder::TimelineStatus::TimelineStatus_Success;
}

ITimelineDecoder::TimelineStatus TimelineMessageDecoder::CreateLabel(const ITimelineDecoder::Label& label)
{
    m_TimelineModel.AddLabel(label);
    return ITimelineDecoder::TimelineStatus::TimelineStatus_Success;
}

ITimelineDecoder::TimelineStatus TimelineMessageDecoder::CreateRelationship(
    const ITimelineDecoder::Relationship& relationship)
{
    m_TimelineModel.AddRelationship(relationship);
    return ITimelineDecoder::TimelineStatus::TimelineStatus_Success;
}

} // namespace profiling

} // namespace armnn