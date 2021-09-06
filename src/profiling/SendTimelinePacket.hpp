//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBufferManager.hpp"
#include "armnn/profiling/ISendTimelinePacket.hpp"
#include "ProfilingUtils.hpp"

#include <armnn/utility/Assert.hpp>

#include <memory>

namespace armnn
{

namespace profiling
{

class SendTimelinePacket : public ISendTimelinePacket
{
public:
    SendTimelinePacket(IBufferManager& bufferManager)
      : m_BufferManager(bufferManager)
      , m_WriteBuffer(nullptr)
      , m_Offset(8u)
      , m_RemainingBufferSize(0u)
      , m_PacketDataLength(0u)
    {}

    /// Commits the current buffer and reset the member variables
    void Commit() override;

    /// Create and write a TimelineEntityBinaryPacket from the parameters to the buffer.
    void SendTimelineEntityBinaryPacket(uint64_t profilingGuid) override;

    /// Create and write a TimelineEventBinaryPacket from the parameters to the buffer.
    void SendTimelineEventBinaryPacket(uint64_t timestamp, int threadId, uint64_t profilingGuid) override;

    /// Create and write a TimelineEventClassBinaryPacket from the parameters to the buffer.
    void SendTimelineEventClassBinaryPacket(uint64_t profilingGuid, uint64_t nameGuid) override;

    /// Create and write a TimelineLabelBinaryPacket from the parameters to the buffer.
    void SendTimelineLabelBinaryPacket(uint64_t profilingGuid, const std::string& label) override;

    /// Create and write a TimelineMessageDirectoryPackage in the buffer
    void SendTimelineMessageDirectoryPackage() override;

    /// Create and write a TimelineRelationshipBinaryPacket from the parameters to the buffer.
    virtual void SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType relationshipType,
                                                      uint64_t relationshipGuid,
                                                      uint64_t headGuid,
                                                      uint64_t tailGuid,
                                                      uint64_t attributeGuid) override;
private:
    /// Reserves maximum packet size from buffer
    void ReserveBuffer();

    template <typename Func, typename ... Params>
    void ForwardWriteBinaryFunction(Func& func, Params&& ... params);

    IBufferManager&  m_BufferManager;
    IPacketBufferPtr m_WriteBuffer;
    unsigned int     m_Offset;
    unsigned int     m_RemainingBufferSize;

    const unsigned int m_uint32_t_size = sizeof(uint32_t);

    std::pair<uint32_t, uint32_t> m_PacketHeader;
    uint32_t                      m_PacketDataLength;

    bool m_DirectoryPackage = false;
};

template<typename Func, typename ... Params>
void SendTimelinePacket::ForwardWriteBinaryFunction(Func& func, Params&& ... params)
{
    try
    {
        ReserveBuffer();
        ARMNN_ASSERT(m_WriteBuffer);
        unsigned int numberOfBytesWritten = 0;
        // Header will be prepended to the buffer on Commit()
        while ( true )
        {
            TimelinePacketStatus result = func(std::forward<Params>(params)...,
                                               &m_WriteBuffer->GetWritableData()[m_Offset],
                                               m_RemainingBufferSize,
                                               numberOfBytesWritten);
            switch ( result )
            {
                case TimelinePacketStatus::BufferExhaustion:
                    Commit();
                    ReserveBuffer();
                    continue;

                case TimelinePacketStatus::Error:
                    throw RuntimeException("Error processing while sending TimelineBinaryPacket", CHECK_LOCATION());

                default:
                    m_Offset += numberOfBytesWritten;
                    m_RemainingBufferSize -= numberOfBytesWritten;
                    return;
            }
        }
    }
    catch (const RuntimeException& ex)
    {
        // don't swallow in the catch all block
        throw ex;
    }
    catch (const BufferExhaustion& ex)
    {
        // ditto
        throw ex;
    }
    catch (const Exception& ex)
    {
        throw ex;
    }
    catch ( ... )
    {
        throw RuntimeException("Unknown Exception thrown while sending TimelineBinaryPacket", CHECK_LOCATION());
    }
}

} // namespace profiling

} // namespace armnn
