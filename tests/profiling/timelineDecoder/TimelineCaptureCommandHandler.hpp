//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ITimelineDecoder.hpp"

#include <CommandHandlerFunctor.hpp>
#include <Packet.hpp>
#include <ProfilingUtils.hpp>

namespace armnn
{

namespace gatordmock
{

class TimelineCaptureCommandHandler : public profiling::CommandHandlerFunctor
{
    // Utils
    uint32_t uint32_t_size = sizeof(uint32_t);
    uint32_t uint64_t_size = sizeof(uint64_t);

    using ReadFunction = void (TimelineCaptureCommandHandler::*)(const unsigned char*, uint32_t);

public:
    TimelineCaptureCommandHandler(uint32_t familyId,
                                  uint32_t packetId,
                                  uint32_t version,
                                  ITimelineDecoder& timelineDecoder,
                                  uint32_t threadId_size)
            : CommandHandlerFunctor(familyId, packetId, version)
            , m_TimelineDecoder(timelineDecoder)
            , m_ThreadIdSize(threadId_size)
    {}

    void operator()(const armnn::profiling::Packet& packet) override;

    void ReadLabel(const unsigned char* data, uint32_t offset);
    void ReadEntity(const unsigned char* data, uint32_t offset);
    void ReadEventClass(const unsigned char* data, uint32_t offset);
    void ReadRelationship(const unsigned char* data, uint32_t offset);
    void ReadEvent(const unsigned char* data, uint32_t offset);

private:
    void ParseData(const armnn::profiling::Packet& packet);

    ITimelineDecoder& m_TimelineDecoder;
    const uint32_t m_ThreadIdSize;
    static const ReadFunction m_ReadFunctions[];
};

} //namespace gatordmock

} //namespace armnn
