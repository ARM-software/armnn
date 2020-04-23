//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/profiling/IBackendProfilingContext.hpp>
#include "CommandHandlerFunctor.hpp"
#include "ISendCounterPacket.hpp"
#include "armnn/profiling/ISendTimelinePacket.hpp"
#include "Packet.hpp"
#include "ProfilingStateMachine.hpp"
#include <future>

namespace armnn
{

namespace profiling
{

class ConnectionAcknowledgedCommandHandler final : public CommandHandlerFunctor
{

typedef const std::unordered_map<BackendId, std::shared_ptr<armnn::profiling::IBackendProfilingContext>>&
    BackendProfilingContexts;

public:
    ConnectionAcknowledgedCommandHandler(uint32_t familyId,
                                         uint32_t packetId,
                                         uint32_t version,
                                         ICounterDirectory& counterDirectory,
                                         ISendCounterPacket& sendCounterPacket,
                                         ISendTimelinePacket& sendTimelinePacket,
                                         ProfilingStateMachine& profilingStateMachine,
                                         Optional<BackendProfilingContexts> backendProfilingContexts = EmptyOptional())
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_CounterDirectory(counterDirectory)
        , m_SendCounterPacket(sendCounterPacket)
        , m_SendTimelinePacket(sendTimelinePacket)
        , m_StateMachine(profilingStateMachine)
        , m_BackendProfilingContext(backendProfilingContexts)
    {}

    void operator()(const Packet& packet) override;

    void setTimelineEnabled(bool timelineEnabled)
    {
        m_TimelineEnabled = timelineEnabled;
    }

private:
    const ICounterDirectory& m_CounterDirectory;
    ISendCounterPacket&      m_SendCounterPacket;
    ISendTimelinePacket&     m_SendTimelinePacket;
    ProfilingStateMachine&   m_StateMachine;
    Optional<BackendProfilingContexts> m_BackendProfilingContext;
    bool m_TimelineEnabled = false;
};

} // namespace profiling

} // namespace armnn

