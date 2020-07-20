//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TimelineCaptureCommandHandler.hpp"

#include <common/include/SwTrace.hpp>

namespace arm
{

namespace pipe
{

class TimelineDirectoryCaptureCommandHandler : public arm::pipe::CommandHandlerFunctor
{
    // Utils
    uint32_t uint8_t_size  = sizeof(uint8_t);
    uint32_t uint32_t_size = sizeof(uint32_t);

public:
    TimelineDirectoryCaptureCommandHandler(uint32_t familyId,
                                           uint32_t packetId,
                                           uint32_t version,
                                           TimelineCaptureCommandHandler& timelineCaptureCommandHandler,
                                           bool quietOperation = false)
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_TimelineCaptureCommandHandler(timelineCaptureCommandHandler)
        , m_QuietOperation(quietOperation)
    {}

    void operator()(const arm::pipe::Packet& packet) override;

    arm::pipe::SwTraceHeader m_SwTraceHeader;
    std::vector<arm::pipe::SwTraceMessage> m_SwTraceMessages;

private:
    void ParseData(const arm::pipe::Packet& packet);
    void Print();

    TimelineCaptureCommandHandler& m_TimelineCaptureCommandHandler;
    bool m_QuietOperation;
};

} //namespace pipe

} //namespace arm
