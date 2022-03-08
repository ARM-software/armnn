//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PerJobCounterSelectionCommandHandler.hpp"

#include <common/include/CommonProfilingUtils.hpp>

#include <fmt/format.h>

namespace arm
{

namespace pipe
{

void PerJobCounterSelectionCommandHandler::operator()(const arm::pipe::Packet& packet)
{
    ProfilingState currentState = m_StateMachine.GetCurrentState();
    switch (currentState)
    {
    case ProfilingState::Uninitialised:
    case ProfilingState::NotConnected:
    case ProfilingState::WaitingForAck:
        throw arm::pipe::ProfilingException(fmt::format(
            "Per-Job Counter Selection Command Handler invoked while in an incorrect state: {}",
            GetProfilingStateName(currentState)));
    case ProfilingState::Active:
        // Process the packet
        if (!(packet.GetPacketFamily() == 0u && packet.GetPacketId() == 5u))
        {
            throw arm::pipe::InvalidArgumentException(fmt::format("Expected Packet family = 0, id = 5 but "
                                                                  "received family = {}, id = {}",
                                                                  packet.GetPacketFamily(),
                                                                  packet.GetPacketId()));
        }

        // Silently drop the packet

        break;
    default:
        throw arm::pipe::ProfilingException(fmt::format("Unknown profiling service state: {}",
                                                        static_cast<int>(currentState)));
    }
}

} // namespace pipe

} // namespace arm
