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

        for (unsigned short i = 0; i < m_CounterDirectory.GetCounterCount(); ++i)
        {
            m_CounterIdToValue[i] = 0;
        }

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

void ProfilingService::SetCounterValue(uint16_t counterIndex, uint32_t value)
{
    CheckIndexSize(counterIndex);
    m_CounterIdToValue.at(counterIndex).store(value, std::memory_order::memory_order_relaxed);
}

void ProfilingService::GetCounterValue(uint16_t counterIndex, uint32_t& value) const
{
    CheckIndexSize(counterIndex);
    value = m_CounterIdToValue.at(counterIndex).load(std::memory_order::memory_order_relaxed);
}

void ProfilingService::AddCounterValue(uint16_t counterIndex, uint32_t value)
{
    CheckIndexSize(counterIndex);
    m_CounterIdToValue.at(counterIndex).fetch_add(value, std::memory_order::memory_order_relaxed);
}

void ProfilingService::SubtractCounterValue(uint16_t counterIndex, uint32_t value)
{
    CheckIndexSize(counterIndex);
    m_CounterIdToValue.at(counterIndex).fetch_sub(value, std::memory_order::memory_order_relaxed);
}

void ProfilingService::IncrementCounterValue(uint16_t counterIndex)
{
    CheckIndexSize(counterIndex);
    m_CounterIdToValue.at(counterIndex).operator++(std::memory_order::memory_order_relaxed);
}

void ProfilingService::DecrementCounterValue(uint16_t counterIndex)
{
    CheckIndexSize(counterIndex);
    m_CounterIdToValue.at(counterIndex).operator--(std::memory_order::memory_order_relaxed);
}

uint16_t ProfilingService::GetCounterCount() const
{
    return m_CounterDirectory.GetCounterCount();
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

inline void ProfilingService::CheckIndexSize(uint16_t counterIndex) const
{
    if (counterIndex >= m_CounterDirectory.GetCounterCount())
    {
        throw InvalidArgumentException("Counter index is out of range");
    }
}

} // namespace profiling

} // namespace armnn