//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CommandHandlerFunctor.hpp"
#include "Packet.hpp"
#include "ProfilingStateMachine.hpp"
#include "INotifyBackends.hpp"

namespace armnn
{

namespace profiling
{

class DeactivateTimelineReportingCommandHandler : public CommandHandlerFunctor
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

    void operator()(const Packet& packet) override;

private:
    std::atomic<bool>&     m_TimelineReporting;
    ProfilingStateMachine& m_StateMachine;
    INotifyBackends&       m_BackendNotifier;
};

} // namespace profiling

} // namespace armnn