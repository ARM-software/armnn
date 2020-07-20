//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ITimelineDecoder.hpp"

#include <common/include/CommandHandlerFunctor.hpp>
#include <common/include/Packet.hpp>

namespace arm
{

namespace pipe
{

class TimelineCaptureCommandHandler : public arm::pipe::CommandHandlerFunctor
{
    // Utils
    uint32_t uint32_t_size = sizeof(uint32_t);
    uint32_t uint64_t_size = sizeof(uint64_t);

    using ReadFunction = ITimelineDecoder::TimelineStatus (TimelineCaptureCommandHandler::*)(
        const unsigned char*, uint32_t&);

public:
    TimelineCaptureCommandHandler(uint32_t familyId,
                                  uint32_t packetId,
                                  uint32_t version,
                                  ITimelineDecoder& timelineDecoder,
                                  uint32_t threadIdSize = 0)
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_TimelineDecoder(timelineDecoder)
        , m_ThreadIdSize(threadIdSize)
        , m_PacketLength(0)
    {}

    void operator()(const arm::pipe::Packet& packet) override;


    void SetThreadIdSize(uint32_t size);

private:
    void ParseData(const arm::pipe::Packet& packet);

    ITimelineDecoder::TimelineStatus ReadLabel(const unsigned char* data, uint32_t& offset);
    ITimelineDecoder::TimelineStatus ReadEntity(const unsigned char* data, uint32_t& offset);
    ITimelineDecoder::TimelineStatus ReadEventClass(const unsigned char* data, uint32_t& offset);
    ITimelineDecoder::TimelineStatus ReadRelationship(const unsigned char* data, uint32_t& offset);
    ITimelineDecoder::TimelineStatus ReadEvent(const unsigned char* data, uint32_t& offset);

    ITimelineDecoder& m_TimelineDecoder;
    uint32_t m_ThreadIdSize;
    unsigned int              m_PacketLength;
    static const ReadFunction m_ReadFunctions[];

};

} //namespace pipe

} //namespace arm
