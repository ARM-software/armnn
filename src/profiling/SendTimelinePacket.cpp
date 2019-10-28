//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SendTimelinePacket.hpp"

namespace armnn
{

namespace profiling
{

void SendTimelinePacket::Commit()
{
    if (m_WriteBuffer != nullptr)
    {
        // Commit the message
        m_BufferManager.Commit(m_WriteBuffer, m_Offset);
        m_WriteBuffer.reset(nullptr);
        m_Offset = 0;
        m_BufferSize = 0;
    }
}

void SendTimelinePacket::ReserveBuffer()
{
    if (m_WriteBuffer == nullptr)
    {
        uint32_t reserved = 0;

        // Reserve the buffer
        m_WriteBuffer = m_BufferManager.Reserve(MAX_METADATA_PACKET_LENGTH, reserved);

        // Check if there is enough space in the buffer
        if (m_WriteBuffer == nullptr || reserved < m_Offset)
        {
            throw BufferExhaustion("No space left on buffer", CHECK_LOCATION());
        }
        m_BufferSize = reserved;
    }
}

#define FORWARD_WRITE_BINARY_FUNC(func, ...) \
try \
{ \
   ReserveBuffer(); \
   unsigned int numberOfBytes = 0; \
   while (1) \
   { \
      TimelinePacketStatus result = func(__VA_ARGS__, numberOfBytes); \
      if (result == armnn::profiling::TimelinePacketStatus::BufferExhaustion) \
      { \
         Commit(); \
         ReserveBuffer(); \
      } \
      else if (result == armnn::profiling::TimelinePacketStatus::Error) \
      { \
         throw RuntimeException("Error processing while sending TimelineBinaryPacket.", CHECK_LOCATION()); \
      } \
      else \
      { \
         break; \
      } \
    } \
    m_Offset     += numberOfBytes; \
    m_BufferSize -= numberOfBytes; \
} \
catch(...) \
{ \
   throw RuntimeException("Error processing while sending TimelineBinaryPacket.", CHECK_LOCATION()); \
}

void SendTimelinePacket::SendTimelineEntityBinaryPacket(uint64_t profilingGuid)
{
    FORWARD_WRITE_BINARY_FUNC(WriteTimelineEntityBinaryPacket,
                              profilingGuid,
                              &m_WriteBuffer->GetWritableData()[m_Offset],
                              m_BufferSize);
}

void SendTimelinePacket::SendTimelineEventBinaryPacket(uint64_t timestamp, uint32_t threadId, uint64_t profilingGuid)
{
    FORWARD_WRITE_BINARY_FUNC(WriteTimelineEventBinaryPacket,
                              timestamp,
                              threadId,
                              profilingGuid,
                              &m_WriteBuffer->GetWritableData()[m_Offset],
                              m_BufferSize);
}

void SendTimelinePacket::SendTimelineEventClassBinaryPacket(uint64_t profilingGuid)
{
    FORWARD_WRITE_BINARY_FUNC(WriteTimelineEventClassBinaryPacket,
                              profilingGuid,
                              &m_WriteBuffer->GetWritableData()[m_Offset],
                              m_BufferSize);
}

void SendTimelinePacket::SendTimelineLabelBinaryPacket(uint64_t profilingGuid, const std::string& label)
{
    FORWARD_WRITE_BINARY_FUNC(WriteTimelineLabelBinaryPacket,
                              profilingGuid,
                              label,
                              &m_WriteBuffer->GetWritableData()[m_Offset],
                              m_BufferSize);
}

void SendTimelinePacket::SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType relationshipType,
                                                              uint64_t relationshipGuid,
                                                              uint64_t headGuid,
                                                              uint64_t tailGuid)
{
    FORWARD_WRITE_BINARY_FUNC(WriteTimelineRelationshipBinaryPacket,
                              relationshipType,
                              relationshipGuid,
                              headGuid,
                              tailGuid,
                              &m_WriteBuffer->GetWritableData()[m_Offset],
                              m_BufferSize);
}

void SendTimelinePacket::SendTimelineMessageDirectoryPackage()
{
    try
    {
        // Reserve buffer if hasn't already reserved
        ReserveBuffer();

        unsigned int numberOfBytes = 0;
        // Write to buffer
        TimelinePacketStatus result = WriteTimelineMessageDirectoryPackage(&m_WriteBuffer->GetWritableData()[m_Offset],
                                                                           m_BufferSize,
                                                                           numberOfBytes);

        if (result != armnn::profiling::TimelinePacketStatus::Ok)
        {
            throw RuntimeException("Error processing TimelineMessageDirectoryPackage.", CHECK_LOCATION());
        }

        // Commit the message
        m_Offset     += numberOfBytes;
        m_BufferSize -= numberOfBytes;
        m_BufferManager.Commit(m_WriteBuffer, m_Offset);
    }
    catch(...)
    {
        throw RuntimeException("Error processing TimelineMessageDirectoryPackage.", CHECK_LOCATION());
    }
}

} // namespace profiling

} // namespace armnn
