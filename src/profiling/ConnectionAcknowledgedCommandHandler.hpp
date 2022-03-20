//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingStateMachine.hpp"

#include <client/include/IProfilingServiceStatus.hpp>
#include <client/include/ISendCounterPacket.hpp>
#include <client/include/ISendTimelinePacket.hpp>

#include <client/include/backends/IBackendProfilingContext.hpp>

#include <common/include/CommandHandlerFunctor.hpp>
#include <common/include/Packet.hpp>

namespace arm
{

namespace pipe
{

class ConnectionAcknowledgedCommandHandler final : public arm::pipe::CommandHandlerFunctor
{

typedef const std::unordered_map<std::string, std::shared_ptr<IBackendProfilingContext>>&
    BackendProfilingContexts;

public:
    ConnectionAcknowledgedCommandHandler(uint32_t familyId,
                                         uint32_t packetId,
                                         uint32_t version,
                                         ICounterDirectory& counterDirectory,
                                         ISendCounterPacket& sendCounterPacket,
                                         ISendTimelinePacket& sendTimelinePacket,
                                         ProfilingStateMachine& profilingStateMachine,
                                         IProfilingServiceStatus& profilingServiceStatus,
                                         arm::pipe::Optional<BackendProfilingContexts> backendProfilingContexts =
                                         arm::pipe::EmptyOptional())
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_CounterDirectory(counterDirectory)
        , m_SendCounterPacket(sendCounterPacket)
        , m_SendTimelinePacket(sendTimelinePacket)
        , m_StateMachine(profilingStateMachine)
        , m_ProfilingServiceStatus(profilingServiceStatus)
        , m_BackendProfilingContext(backendProfilingContexts)
        , m_TimelineEnabled(false)
    {}

    void operator()(const arm::pipe::Packet& packet) override;

    void setTimelineEnabled(bool timelineEnabled)
    {
        m_TimelineEnabled = timelineEnabled;
    }

private:
    const ICounterDirectory& m_CounterDirectory;
    ISendCounterPacket&      m_SendCounterPacket;
    ISendTimelinePacket&     m_SendTimelinePacket;
    ProfilingStateMachine&   m_StateMachine;
    IProfilingServiceStatus& m_ProfilingServiceStatus;
    arm::pipe::Optional<BackendProfilingContexts> m_BackendProfilingContext;
    std::atomic<bool> m_TimelineEnabled;
};

} // namespace pipe

} // namespace arm
