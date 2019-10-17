//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Packet.hpp"
#include "CommandHandlerFunctor.hpp"
#include "ProfilingStateMachine.hpp"

namespace armnn
{

namespace profiling
{

class PerJobCounterSelectionCommandHandler : public CommandHandlerFunctor
{

public:
    PerJobCounterSelectionCommandHandler(uint32_t familyId,
                                         uint32_t packetId,
                                         uint32_t version,
                                         const ProfilingStateMachine& profilingStateMachine)
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_StateMachine(profilingStateMachine)
    {}

    void operator()(const Packet& packet) override;

private:
    const ProfilingStateMachine& m_StateMachine;
};

} // namespace profiling

} // namespace armnn

