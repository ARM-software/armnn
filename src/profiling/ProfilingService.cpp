//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingService.hpp"

namespace armnn
{

namespace profiling
{

ProfilingService::ProfilingService(const Runtime::CreationOptions::ExternalProfilingOptions& options)
    : m_Options(options)
{
    Initialise();
}

void ProfilingService::Initialise()
{
    if (m_Options.m_EnableProfiling == true)
    {
        // Setup Counter Directory - this should only be created if profiling is enabled
        // Setup Counter meta

        // For now until CounterDirectory setup is implemented, change m_State once everything initialised
        m_State.TransitionToState(ProfilingState::NotConnected);
    }
}

void ProfilingService::Run()
{
    if (m_State.GetCurrentState() == ProfilingState::NotConnected)
    {
        //  Since GetProfilingConnection is not implemented, if !NULL,
        //  then change to WaitingForAck. This will need to change once there is implementation
        //  for the IProfilingConnection
        if (!m_Factory.GetProfilingConnection(m_Options))
        {
            m_State.TransitionToState(ProfilingState::WaitingForAck);
        }
    } else if (m_State.GetCurrentState() == ProfilingState::Uninitialised && m_Options.m_EnableProfiling == true)
    {
        Initialise();
    }
}

ProfilingState ProfilingService::GetCurrentState() const
{
    return m_State.GetCurrentState();
}
} // namespace profiling

} // namespace armnn