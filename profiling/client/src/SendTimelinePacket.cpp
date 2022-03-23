//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SendTimelinePacket.hpp"

namespace arm
{

namespace pipe
{

void SendTimelinePacket::Commit()
{
    if (m_WriteBuffer == nullptr)
    {
        // Can't commit from a null buffer
        return;
    }

    if (!m_DirectoryPackage)
    {
        // Datalength should be Offset minus the two header words
        m_PacketDataLength = m_Offset - m_uint32_t_size * 2;
        // Reset offset to prepend header with full packet datalength
        m_Offset = 0;

        // Add header before commit
        m_PacketHeader = CreateTimelinePacketHeader(1,0,1,0,0,m_PacketDataLength);

        // Write the timeline binary packet header to the buffer
        WriteUint32(m_WriteBuffer->GetWritableData(), m_Offset, m_PacketHeader.first);
        m_Offset += m_uint32_t_size;
        WriteUint32(m_WriteBuffer->GetWritableData(), m_Offset, m_PacketHeader.second);

        m_BufferManager.Commit(m_WriteBuffer, m_PacketDataLength + m_uint32_t_size * 2);

    }
    else
    {
        m_DirectoryPackage = false;
        m_BufferManager.Commit(m_WriteBuffer, m_Offset);
    }

    // Commit the message
    m_WriteBuffer.reset(nullptr);
    // Reset offset to start after prepended header
    m_Offset = 8;
    m_RemainingBufferSize = 0;
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
    if (m_WriteBuffer == nullptr)
    {
        throw arm::pipe::BufferExhaustion("No free buffers left", LOCATION());
    }
    if (reserved < m_Offset)
    {
        throw arm::pipe::BufferExhaustion("Reserved space too small for use", LOCATION());
    }

    if (m_DirectoryPackage)
    {
        m_RemainingBufferSize = reserved;
        return;
    }
    // Account for the header size which is added at Commit()
    m_RemainingBufferSize = reserved - 8;
}

void SendTimelinePacket::SendTimelineEntityBinaryPacket(uint64_t profilingGuid)
{
    ForwardWriteBinaryFunction(WriteTimelineEntityBinary,
                               profilingGuid);
}

void SendTimelinePacket::SendTimelineEventBinaryPacket(uint64_t timestamp,
                                                       int threadId,
                                                       uint64_t profilingGuid)
{
    ForwardWriteBinaryFunction(WriteTimelineEventBinary,
                               timestamp,
                               threadId,
                               profilingGuid);
}

void SendTimelinePacket::SendTimelineEventClassBinaryPacket(uint64_t profilingGuid, uint64_t nameGuid)
{
    ForwardWriteBinaryFunction(WriteTimelineEventClassBinary,
                               profilingGuid,
                               nameGuid);
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
                                                              uint64_t tailGuid,
                                                              uint64_t attributeGuid)
{
    ForwardWriteBinaryFunction(WriteTimelineRelationshipBinary,
                               relationshipType,
                               relationshipGuid,
                               headGuid,
                               tailGuid,
                               attributeGuid);
}

void SendTimelinePacket::SendTimelineMessageDirectoryPackage()
{
    try
    {
        // Flag to Reserve & Commit() that a DirectoryPackage is being sent
        m_DirectoryPackage = true;
        // Reserve buffer if it hasn't already been reserved
        ReserveBuffer();
        // Write to buffer
        unsigned int numberOfBytesWritten = 0;
        // Offset is initialised to 8
        m_Offset = 0;

        TimelinePacketStatus result = WriteTimelineMessageDirectoryPackage(&m_WriteBuffer->GetWritableData()[m_Offset],
                                                                           m_RemainingBufferSize,
                                                                           numberOfBytesWritten);
        if (result != TimelinePacketStatus::Ok)
        {
            throw arm::pipe::ProfilingException("Error processing TimelineMessageDirectoryPackage", LOCATION());
        }

        // Commit the message
        m_Offset     += numberOfBytesWritten;
        m_RemainingBufferSize -= numberOfBytesWritten;
        Commit();
    }
    catch (...)
    {
        throw arm::pipe::ProfilingException("Error processing TimelineMessageDirectoryPackage", LOCATION());
    }
}

} // namespace pipe

} // namespace arm
