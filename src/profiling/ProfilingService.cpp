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

    // Check if the profiling service needs to be reset
    if (resetProfilingService)
    {
        // Reset the profiling service
        Reset();
    }
}

void ProfilingService::Update()
{
    if (!m_Options.m_EnableProfiling)
    {
        // Don't run if profiling is disabled
        return;
    }

    ProfilingState currentState = m_StateMachine.GetCurrentState();
    switch (currentState)
    {
    case ProfilingState::Uninitialised:
        // Initialize the profiling service
        Initialize();

        // Move to the next state
        m_StateMachine.TransitionToState(ProfilingState::NotConnected);
        break;
    case ProfilingState::NotConnected:
        // Stop the command thread (if running)
        m_CommandHandler.Stop();

        // Stop the send thread (if running)
        m_SendCounterPacket.Stop(false);

        // Reset any existing profiling connection
        m_ProfilingConnection.reset();

        try
        {
            // Setup the profiling connection
            BOOST_ASSERT(m_ProfilingConnectionFactory);
            m_ProfilingConnection = m_ProfilingConnectionFactory->GetProfilingConnection(m_Options);
        }
        catch (const Exception& e)
        {
            BOOST_LOG_TRIVIAL(warning) << "An error has occurred when creating the profiling connection: "
                                       << e.what() << std::endl;
        }

        // Move to the next state
        m_StateMachine.TransitionToState(m_ProfilingConnection
                                         ? ProfilingState::WaitingForAck  // Profiling connection obtained, wait for ack
                                         : ProfilingState::NotConnected); // Profiling connection failed, stay in the
                                                                          // "NotConnected" state
        break;
    case ProfilingState::WaitingForAck:
        BOOST_ASSERT(m_ProfilingConnection);

        // Start the command thread
        m_CommandHandler.Start(*m_ProfilingConnection);

        // Start the send thread, while in "WaitingForAck" state it'll send out a "Stream MetaData" packet waiting for
        // a valid "Connection Acknowledged" packet confirming the connection
        m_SendCounterPacket.Start(*m_ProfilingConnection);

        // The connection acknowledged command handler will automatically transition the state to "Active" once a
        // valid "Connection Acknowledged" packet has been received

        break;
    case ProfilingState::Active:

        break;
    default:
        throw RuntimeException(boost::str(boost::format("Unknown profiling service state: %1")
                                          % static_cast<int>(currentState)));
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

void ProfilingService::Reset()
{
    // Reset the profiling service

    // The order in which we reset/stop the components is not trivial!

    // First stop the threads (Command Handler first)...
    m_CommandHandler.Stop();
    m_SendCounterPacket.Stop(false);

    // ...then destroy the profiling connection...
    m_ProfilingConnection.reset();

    // ...then delete all the counter data and configuration...
    m_CounterIndex.clear();
    m_CounterValues.clear();
    m_CounterDirectory.Clear();

    // ...finally reset the profiling state machine
    m_StateMachine.Reset();
}

} // namespace profiling

} // namespace armnn
