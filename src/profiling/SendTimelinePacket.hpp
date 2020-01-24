//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBufferManager.hpp"
#include "armnn/profiling/ISendTimelinePacket.hpp"
#include "ProfilingUtils.hpp"

#include <boost/assert.hpp>

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
      , m_Offset(0u)
      , m_BufferSize(0u)
    {}

    /// Commits the current buffer and reset the member variables
    void Commit() override;

    /// Create and write a TimelineEntityBinaryPacket from the parameters to the buffer.
    void SendTimelineEntityBinaryPacket(uint64_t profilingGuid) override;

    /// Create and write a TimelineEventBinaryPacket from the parameters to the buffer.
    void SendTimelineEventBinaryPacket(uint64_t timestamp, std::thread::id threadId, uint64_t profilingGuid) override;

    /// Create and write a TimelineEventClassBinaryPacket from the parameters to the buffer.
    void SendTimelineEventClassBinaryPacket(uint64_t profilingGuid) override;

    /// Create and write a TimelineLabelBinaryPacket from the parameters to the buffer.
    void SendTimelineLabelBinaryPacket(uint64_t profilingGuid, const std::string& label) override;

    /// Create and write a TimelineMessageDirectoryPackage in the buffer
    void SendTimelineMessageDirectoryPackage() override;

    /// Create and write a TimelineRelationshipBinaryPacket from the parameters to the buffer.
    virtual void SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType relationshipType,
                                                      uint64_t relationshipGuid,
                                                      uint64_t headGuid,
                                                      uint64_t tailGuid) override;
private:
    /// Reserves maximum packet size from buffer
    void ReserveBuffer();

    template <typename Func, typename ... Params>
    void ForwardWriteBinaryFunction(Func& func, Params&& ... params);

    IBufferManager& m_BufferManager;
    IPacketBufferPtr m_WriteBuffer;
    unsigned int m_Offset;
    unsigned int m_BufferSize;
};

template <typename Func, typename ... Params>
void SendTimelinePacket::ForwardWriteBinaryFunction(Func& func, Params&& ... params)
{
    try
    {
        ReserveBuffer();
        BOOST_ASSERT(m_WriteBuffer);
        unsigned int numberOfBytesWritten = 0;
        while (true)
        {
            TimelinePacketStatus result = func(std::forward<Params>(params)...,
                                               &m_WriteBuffer->GetWritableData()[m_Offset],
                                               m_BufferSize,
                                               numberOfBytesWritten);
            switch (result)
            {
            case TimelinePacketStatus::BufferExhaustion:
                Commit();
                ReserveBuffer();
                continue;

            case TimelinePacketStatus::Error:
                throw RuntimeException("Error processing while sending TimelineBinaryPacket", CHECK_LOCATION());

            default:
                m_Offset     += numberOfBytesWritten;
                m_BufferSize -= numberOfBytesWritten;
                return;
            }
        }
    }
    catch (...)
    {
        throw RuntimeException("Error processing while sending TimelineBinaryPacket", CHECK_LOCATION());
    }
}

} // namespace profiling

} // namespace armnn
