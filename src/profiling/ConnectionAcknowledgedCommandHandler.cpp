//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConnectionAcknowledgedCommandHandler.hpp"

#include <client/include/TimelineUtilityMethods.hpp>

#include <common/include/ProfilingException.hpp>

#include <fmt/format.h>

namespace arm
{

namespace pipe
{

void ConnectionAcknowledgedCommandHandler::operator()(const arm::pipe::Packet& packet)
{
    ProfilingState currentState = m_StateMachine.GetCurrentState();
    switch (currentState)
    {
    case ProfilingState::Uninitialised:
    case ProfilingState::NotConnected:
        throw arm::pipe::ProfilingException(fmt::format("Connection Acknowledged Command Handler invoked while in an "
                                            "wrong state: {}",
                                            GetProfilingStateName(currentState)));
    case ProfilingState::WaitingForAck:
        // Process the packet
        if (!(packet.GetPacketFamily() == 0u && packet.GetPacketId() == 1u))
        {
            throw arm::pipe::InvalidArgumentException(fmt::format("Expected Packet family = 0, id = 1 but "
                                                                  "received family = {}, id = {}",
                                                                  packet.GetPacketFamily(),
                                                                  packet.GetPacketId()));
        }

        // Once a Connection Acknowledged packet has been received, move to the Active state immediately
        m_StateMachine.TransitionToState(ProfilingState::Active);
        // Send the counter directory packet.
        m_SendCounterPacket.SendCounterDirectoryPacket(m_CounterDirectory);

        if (m_TimelineEnabled)
        {
            m_SendTimelinePacket.SendTimelineMessageDirectoryPackage();
            TimelineUtilityMethods::SendWellKnownLabelsAndEventClasses(m_SendTimelinePacket);
        }

        if (m_BackendProfilingContext.has_value())
        {
            for (auto backendContext : m_BackendProfilingContext.value())
            {
                // Enable profiling on the backend and assert that it returns true
                if(!backendContext.second->EnableProfiling(true))
                {
                    throw arm::pipe::BackendProfilingException(
                            "Unable to enable profiling on Backend Id: " + backendContext.first);
                }
            }
        }

        // At this point signal any external processes waiting on the profiling system
        // to come up that profiling is fully active
        m_ProfilingServiceStatus.NotifyProfilingServiceActive();
        break;
    case ProfilingState::Active:
        return; // NOP
    default:
        throw arm::pipe::ProfilingException(fmt::format("Unknown profiling service state: {}",
                                            static_cast<int>(currentState)));
    }
}

} // namespace pipe

} // namespace arm
