//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingStateMachine.hpp"
#include "SendTimelinePacket.hpp"
#include "IReportStructure.hpp"
#include "INotifyBackends.hpp"

#include "armnn/Optional.hpp"

#include <common/include/CommandHandlerFunctor.hpp>
#include <common/include/Packet.hpp>


namespace armnn
{

namespace profiling
{

class ActivateTimelineReportingCommandHandler : public arm::pipe::CommandHandlerFunctor
{
public:
    ActivateTimelineReportingCommandHandler(uint32_t familyId,
                                            uint32_t packetId,
                                            uint32_t version,
                                            SendTimelinePacket& sendTimelinePacket,
                                            ProfilingStateMachine& profilingStateMachine,
                                            Optional<IReportStructure&> reportStructure,
                                            std::atomic<bool>& timelineReporting,
                                            INotifyBackends& notifyBackends)
        : CommandHandlerFunctor(familyId, packetId, version),
          m_SendTimelinePacket(sendTimelinePacket),
          m_StateMachine(profilingStateMachine),
          m_TimelineReporting(timelineReporting),
          m_BackendNotifier(notifyBackends),
          m_ReportStructure(reportStructure)
    {}

    void operator()(const arm::pipe::Packet& packet) override;

private:
    SendTimelinePacket&    m_SendTimelinePacket;
    ProfilingStateMachine& m_StateMachine;
    std::atomic<bool>&     m_TimelineReporting;
    INotifyBackends&       m_BackendNotifier;

    Optional<IReportStructure&> m_ReportStructure;
};

} // namespace profiling

} // namespace armnn