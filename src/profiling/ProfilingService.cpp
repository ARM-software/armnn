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
        // Setup provisional Counter Directory example - this should only be created if profiling is enabled
        // Setup provisional Counter meta example
        const std::string categoryName = "Category";

        m_CounterDirectory.RegisterCategory(categoryName);
        m_CounterDirectory.RegisterDevice("device name", 0, categoryName);
        m_CounterDirectory.RegisterCounterSet("counterSet_name", 2, categoryName);

        m_CounterDirectory.RegisterCounter(categoryName,
                                           0,
                                           1,
                                           123.45f,
                                           "counter name 1",
                                           "counter description");

        m_CounterDirectory.RegisterCounter(categoryName,
                                           0,
                                           1,
                                           123.45f,
                                           "counter name 2",
                                           "counter description");

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

const ICounterDirectory& ProfilingService::GetCounterDirectory() const
{
    return m_CounterDirectory;
}

ProfilingState ProfilingService::GetCurrentState() const
{
    return m_State.GetCurrentState();
}

void ProfilingService::ResetExternalProfilingOptions(const Runtime::CreationOptions::ExternalProfilingOptions& options)
{
    if(!m_Options.m_EnableProfiling)
    {
        m_Options = options;
        Initialise();
        return;
    }
    m_Options = options;
}
} // namespace profiling

} // namespace armnn