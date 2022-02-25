//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingStateMachine.hpp"
#include "INotifyBackends.hpp"
#include "ProfilingStateMachine.hpp"

#include <common/include/CommandHandlerFunctor.hpp>
#include <common/include/Packet.hpp>

namespace arm
{

namespace pipe
{

class DeactivateTimelineReportingCommandHandler : public arm::pipe::CommandHandlerFunctor
{

public:
    DeactivateTimelineReportingCommandHandler(uint32_t familyId,
                                              uint32_t packetId,
                                              uint32_t version,
                                              std::atomic<bool>& timelineReporting,
                                              ProfilingStateMachine& profilingStateMachine,
                                              INotifyBackends& notifyBackends)
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_TimelineReporting(timelineReporting)
        , m_StateMachine(profilingStateMachine)
        , m_BackendNotifier(notifyBackends)
    {}

    void operator()(const arm::pipe::Packet& packet) override;

private:
    std::atomic<bool>&     m_TimelineReporting;
    ProfilingStateMachine& m_StateMachine;
    INotifyBackends&       m_BackendNotifier;
};

} // namespace pipe

} // namespace arm