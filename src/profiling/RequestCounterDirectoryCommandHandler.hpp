//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingStateMachine.hpp"

#include <client/include/ISendCounterPacket.hpp>
#include <client/include/ISendTimelinePacket.hpp>

#include <common/include/CommandHandlerFunctor.hpp>
#include <common/include/Packet.hpp>

namespace arm
{

namespace pipe
{

class RequestCounterDirectoryCommandHandler : public arm::pipe::CommandHandlerFunctor
{

public:
    RequestCounterDirectoryCommandHandler(uint32_t familyId,
                                          uint32_t packetId,
                                          uint32_t version,
                                          ICounterDirectory& counterDirectory,
                                          ISendCounterPacket& sendCounterPacket,
                                          ISendTimelinePacket& sendTimelinePacket,
                                          ProfilingStateMachine& profilingStateMachine)
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_CounterDirectory(counterDirectory)
        , m_SendCounterPacket(sendCounterPacket)
        , m_SendTimelinePacket(sendTimelinePacket)
        , m_StateMachine(profilingStateMachine)
    {}

    void operator()(const arm::pipe::Packet& packet) override;

private:
    const ICounterDirectory& m_CounterDirectory;
    ISendCounterPacket& m_SendCounterPacket;
    ISendTimelinePacket& m_SendTimelinePacket;
    const ProfilingStateMachine& m_StateMachine;
};

} // namespace pipe

} // namespace arm
