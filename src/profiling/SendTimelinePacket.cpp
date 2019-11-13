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
    if (m_WriteBuffer == nullptr)
    {
        // Can't commit from a null buffer
        return;
    }

    // Commit the message
    m_BufferManager.Commit(m_WriteBuffer, m_Offset);
    m_WriteBuffer.reset(nullptr);
    m_Offset = 0;
    m_BufferSize = 0;
}

void SendTimelinePacket::ReserveBuffer()
{
    if (m_WriteBuffer != nullptr)
    {
        // Buffer already reserved
        return;
    }

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

void SendTimelinePacket::SendTimelineEntityBinaryPacket(uint64_t profilingGuid)
{
    ForwardWriteBinaryFunction(WriteTimelineEntityBinaryPacket,
                               profilingGuid);
}

void SendTimelinePacket::SendTimelineEventBinaryPacket(uint64_t timestamp,
                                                       std::thread::id threadId,
                                                       uint64_t profilingGuid)
{
    ForwardWriteBinaryFunction(WriteTimelineEventBinaryPacket,
                               timestamp,
                               threadId,
                               profilingGuid);
}

void SendTimelinePacket::SendTimelineEventClassBinaryPacket(uint64_t profilingGuid)
{
    ForwardWriteBinaryFunction(WriteTimelineEventClassBinaryPacket,
                               profilingGuid);
}

void SendTimelinePacket::SendTimelineLabelBinaryPacket(uint64_t profilingGuid, const std::string& label)
{
    ForwardWriteBinaryFunction(WriteTimelineLabelBinaryPacket,
                               profilingGuid,
                               label);
}

void SendTimelinePacket::SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType relationshipType,
                                                              uint64_t relationshipGuid,
                                                              uint64_t headGuid,
                                                              uint64_t tailGuid)
{
    ForwardWriteBinaryFunction(WriteTimelineRelationshipBinaryPacket,
                               relationshipType,
                               relationshipGuid,
                               headGuid,
                               tailGuid);
}

void SendTimelinePacket::SendTimelineMessageDirectoryPackage()
{
    try
    {
        // Reserve buffer if it hasn't already been reserved
        ReserveBuffer();

        // Write to buffer
        unsigned int numberOfBytesWritten = 0;
        TimelinePacketStatus result = WriteTimelineMessageDirectoryPackage(&m_WriteBuffer->GetWritableData()[m_Offset],
                                                                           m_BufferSize,
                                                                           numberOfBytesWritten);
        if (result != armnn::profiling::TimelinePacketStatus::Ok)
        {
            throw RuntimeException("Error processing TimelineMessageDirectoryPackage", CHECK_LOCATION());
        }

        // Commit the message
        m_Offset     += numberOfBytesWritten;
        m_BufferSize -= numberOfBytesWritten;
        Commit();
    }
    catch (...)
    {
        throw RuntimeException("Error processing TimelineMessageDirectoryPackage", CHECK_LOCATION());
    }
}

} // namespace profiling

} // namespace armnn
