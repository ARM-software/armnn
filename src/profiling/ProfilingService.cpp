//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingService.hpp"

#include <boost/log/trivial.hpp>
#include <boost/format.hpp>

namespace armnn
{

namespace profiling
{

void ProfilingService::ResetExternalProfilingOptions(const ExternalProfilingOptions& options,
                                                     bool resetProfilingService)
{
    // Update the profiling options
    m_Options = options;

    if (resetProfilingService)
    {
        // Reset the profiling service
        m_CounterDirectory.Clear();
        m_ProfilingConnection.reset();
        m_StateMachine.Reset();
        m_CounterIndex.clear();
        m_CounterValues.clear();
    }

    // Re-initialize the profiling service
    Initialize();
}

void ProfilingService::Run()
{
    if (m_StateMachine.GetCurrentState() == ProfilingState::Uninitialised)
    {
        Initialize();
    }
    else if (m_StateMachine.GetCurrentState() == ProfilingState::NotConnected)
    {
        try
        {
            m_ProfilingConnectionFactory.GetProfilingConnection(m_Options);
            m_StateMachine.TransitionToState(ProfilingState::WaitingForAck);
        }
        catch (const armnn::Exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }
}

const ICounterDirectory& ProfilingService::GetCounterDirectory() const
{
    return m_CounterDirectory;
}

ProfilingState ProfilingService::GetCurrentState() const
{
    return m_StateMachine.GetCurrentState();
}

uint16_t ProfilingService::GetCounterCount() const
{
    return m_CounterDirectory.GetCounterCount();
}

uint32_t ProfilingService::GetCounterValue(uint16_t counterUid) const
{
    BOOST_ASSERT(counterUid < m_CounterIndex.size());
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    BOOST_ASSERT(counterValuePtr);
    return counterValuePtr->load(std::memory_order::memory_order_relaxed);
}

void ProfilingService::SetCounterValue(uint16_t counterUid, uint32_t value)
{
    BOOST_ASSERT(counterUid < m_CounterIndex.size());
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    BOOST_ASSERT(counterValuePtr);
    counterValuePtr->store(value, std::memory_order::memory_order_relaxed);
}

uint32_t ProfilingService::AddCounterValue(uint16_t counterUid, uint32_t value)
{
    BOOST_ASSERT(counterUid < m_CounterIndex.size());
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    BOOST_ASSERT(counterValuePtr);
    return counterValuePtr->fetch_add(value, std::memory_order::memory_order_relaxed);
}

uint32_t ProfilingService::SubtractCounterValue(uint16_t counterUid, uint32_t value)
{
    BOOST_ASSERT(counterUid < m_CounterIndex.size());
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    BOOST_ASSERT(counterValuePtr);
    return counterValuePtr->fetch_sub(value, std::memory_order::memory_order_relaxed);
}

uint32_t ProfilingService::IncrementCounterValue(uint16_t counterUid)
{
    BOOST_ASSERT(counterUid < m_CounterIndex.size());
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    BOOST_ASSERT(counterValuePtr);
    return counterValuePtr->operator++(std::memory_order::memory_order_relaxed);
}

uint32_t ProfilingService::DecrementCounterValue(uint16_t counterUid)
{
    BOOST_ASSERT(counterUid < m_CounterIndex.size());
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    BOOST_ASSERT(counterValuePtr);
    return counterValuePtr->operator--(std::memory_order::memory_order_relaxed);
}

void ProfilingService::Initialize()
{
    if (!m_Options.m_EnableProfiling)
    {
        // Skip the initialization if profiling is disabled
        return;
    }

    // Register a category for the basic runtime counters
    if (!m_CounterDirectory.IsCategoryRegistered("ArmNN_Runtime"))
    {
        m_CounterDirectory.RegisterCategory("ArmNN_Runtime");
    }

    // Register a counter for the number of loaded networks
    if (!m_CounterDirectory.IsCounterRegistered("Loaded networks"))
    {
        const Counter* loadedNetworksCounter =
                m_CounterDirectory.RegisterCounter("ArmNN_Runtime",
                                                   0,
                                                   0,
                                                   1.f,
                                                   "Loaded networks",
                                                   "The number of networks loaded at runtime",
                                                   std::string("networks"));
        BOOST_ASSERT(loadedNetworksCounter);
        InitializeCounterValue(loadedNetworksCounter->m_Uid);
    }

    // Register a counter for the number of registered backends
    if (!m_CounterDirectory.IsCounterRegistered("Registered backends"))
    {
        const Counter* registeredBackendsCounter =
                m_CounterDirectory.RegisterCounter("ArmNN_Runtime",
                                                   0,
                                                   0,
                                                   1.f,
                                                   "Registered backends",
                                                   "The number of registered backends",
                                                   std::string("backends"));
        BOOST_ASSERT(registeredBackendsCounter);
        InitializeCounterValue(registeredBackendsCounter->m_Uid);
    }

    // Register a counter for the number of inferences run
    if (!m_CounterDirectory.IsCounterRegistered("Inferences run"))
    {
        const Counter* inferencesRunCounter =
                m_CounterDirectory.RegisterCounter("ArmNN_Runtime",
                                                   0,
                                                   0,
                                                   1.f,
                                                   "Inferences run",
                                                   "The number of inferences run",
                                                   std::string("inferences"));
        BOOST_ASSERT(inferencesRunCounter);
        InitializeCounterValue(inferencesRunCounter->m_Uid);
    }

    // Initialization is done, update the profiling service state
    m_StateMachine.TransitionToState(ProfilingState::NotConnected);
}

void ProfilingService::InitializeCounterValue(uint16_t counterUid)
{
    // Increase the size of the counter index if necessary
    if (counterUid >= m_CounterIndex.size())
    {
        m_CounterIndex.resize(boost::numeric_cast<size_t>(counterUid) + 1);
    }

    // Create a new atomic counter and add it to the list
    m_CounterValues.emplace_back(0);

    // Register the new counter to the counter index for quick access
    std::atomic<uint32_t>* counterValuePtr = &(m_CounterValues.back());
    m_CounterIndex.at(counterUid) = counterValuePtr;
}

} // namespace profiling

} // namespace armnn
