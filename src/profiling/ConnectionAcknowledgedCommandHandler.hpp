//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CommandHandlerFunctor.hpp"
#include "Packet.hpp"
#include "ProfilingStateMachine.hpp"

namespace armnn
{

namespace profiling
{

class ConnectionAcknowledgedCommandHandler final : public CommandHandlerFunctor
{

public:
    ConnectionAcknowledgedCommandHandler(uint32_t packetId,
                                         uint32_t version,
                                         ProfilingStateMachine& profilingStateMachine)
        : CommandHandlerFunctor(packetId, version)
        , m_StateMachine(profilingStateMachine)
    {}

    void operator()(const Packet& packet) override;

private:
    ProfilingStateMachine& m_StateMachine;
};

} // namespace profiling

} // namespace armnn

