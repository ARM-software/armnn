//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once


#include <CommandHandlerFunctor.hpp>
#include <Packet.hpp>
#include <PacketBuffer.hpp>
#include <ProfilingUtils.hpp>

namespace armnn
{

namespace gatordmock
{

class TimelineDirectoryCaptureCommandHandler : public profiling::CommandHandlerFunctor
{
    // Utils
    uint32_t uint8_t_size  = sizeof(uint8_t);
    uint32_t uint32_t_size = sizeof(uint32_t);

public:
    TimelineDirectoryCaptureCommandHandler(uint32_t familyId,
                                           uint32_t packetId,
                                           uint32_t version,
                                           bool quietOperation = false)
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_QuietOperation(quietOperation)
    {}

    void operator()(const armnn::profiling::Packet& packet) override;

    profiling::SwTraceHeader m_SwTraceHeader;
    std::vector<profiling::SwTraceMessage> m_SwTraceMessages;

private:
    void ParseData(const armnn::profiling::Packet& packet);
    void Print();

    bool m_QuietOperation;
};

} //namespace gatordmock

} //namespace armnn
