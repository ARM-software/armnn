//
// Copyright © 2019, 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingService.hpp"

#include <common/include/Logging.hpp>
#include <common/include/NumericCast.hpp>
#include <common/include/ProfilingGuid.hpp>
#include <common/include/SocketConnectionException.hpp>

#if defined(ARMNN_BUILD_BARE_METAL) || defined(ARMNN_EXECUTE_NETWORK_STATIC)
#include <common/include/IgnoreUnused.hpp>
#endif


#include <fmt/format.h>

namespace arm
{

namespace pipe
{

void ProfilingService::ResetExternalProfilingOptions(const arm::pipe::ProfilingOptions& options,
                                                     bool resetProfilingService)
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    // Update the profiling options
    m_Options = options;
    m_TimelineReporting = options.m_TimelineEnabled;
    m_ConnectionAcknowledgedCommandHandler.setTimelineEnabled(options.m_TimelineEnabled);

    // Check if the profiling service needs to be reset
    if (resetProfilingService)
    {
        // Reset the profiling service
        Reset();
    }
#else
    IgnoreUnused(options);
    IgnoreUnused(resetProfilingService);
#endif // ARMNN_BUILD_BARE_METAL || ARMNN_EXECUTE_NETWORK_STATIC
}

bool ProfilingService::IsProfilingEnabled() const
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    return m_Options.m_EnableProfiling;
#else
    return false;
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

ProfilingState ProfilingService::ConfigureProfilingService(
        const ProfilingOptions& options,
        bool resetProfilingService)
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    ResetExternalProfilingOptions(options, resetProfilingService);
    ProfilingState currentState = m_StateMachine.GetCurrentState();
    if (options.m_EnableProfiling)
    {
        switch (currentState)
        {
            case ProfilingState::Uninitialised:
                Update(); // should transition to NotConnected
                Update(); // will either stay in NotConnected because there is no server
                          // or will enter WaitingForAck.
                currentState = m_StateMachine.GetCurrentState();
                if (currentState == ProfilingState::WaitingForAck)
                {
                    Update(); // poke it again to send out the metadata packet
                }
                currentState = m_StateMachine.GetCurrentState();
                return currentState;
            case ProfilingState::NotConnected:
                Update(); // will either stay in NotConnected because there is no server
                          // or will enter WaitingForAck
                currentState = m_StateMachine.GetCurrentState();
                if (currentState == ProfilingState::WaitingForAck)
                {
                    Update(); // poke it again to send out the metadata packet
                }
                currentState = m_StateMachine.GetCurrentState();
                return currentState;
            default:
                return currentState;
        }
    }
    else
    {
        // Make sure profiling is shutdown
        switch (currentState)
        {
            case ProfilingState::Uninitialised:
            case ProfilingState::NotConnected:
                return currentState;
            default:
                Stop();
                return m_StateMachine.GetCurrentState();
        }
    }
#else
    IgnoreUnused(options);
    IgnoreUnused(resetProfilingService);
    return ProfilingState::Uninitialised;
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

void ProfilingService::Update()
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
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
        m_SendThread.Stop(false);

        // Stop the periodic counter capture thread (if running)
        m_PeriodicCounterCapture.Stop();

        // Reset any existing profiling connection
        m_ProfilingConnection.reset();

        try
        {
            // Setup the profiling connection
            ARM_PIPE_ASSERT(m_ProfilingConnectionFactory);
            m_ProfilingConnection = m_ProfilingConnectionFactory->GetProfilingConnection(m_Options);
        }
        catch (const arm::pipe::ProfilingException& e)
        {
            ARM_PIPE_LOG(warning) << "An error has occurred when creating the profiling connection: "
                                       << e.what();
        }
        catch (const arm::pipe::SocketConnectionException& e)
        {
            ARM_PIPE_LOG(warning) << "An error has occurred when creating the profiling connection ["
                                       << e.what() << "] on socket [" << e.GetSocketFd() << "].";
        }

        // Move to the next state
        m_StateMachine.TransitionToState(m_ProfilingConnection
                                         ? ProfilingState::WaitingForAck  // Profiling connection obtained, wait for ack
                                         : ProfilingState::NotConnected); // Profiling connection failed, stay in the
                                                                          // "NotConnected" state
        break;
    case ProfilingState::WaitingForAck:
        ARM_PIPE_ASSERT(m_ProfilingConnection);

        // Start the command thread
        m_CommandHandler.Start(*m_ProfilingConnection);

        // Start the send thread, while in "WaitingForAck" state it'll send out a "Stream MetaData" packet waiting for
        // a valid "Connection Acknowledged" packet confirming the connection
        m_SendThread.Start(*m_ProfilingConnection);

        // The connection acknowledged command handler will automatically transition the state to "Active" once a
        // valid "Connection Acknowledged" packet has been received

        break;
    case ProfilingState::Active:

        // The period counter capture thread is started by the Periodic Counter Selection command handler upon
        // request by an external profiling service

        break;
    default:
        throw arm::pipe::ProfilingException(fmt::format("Unknown profiling service state: {}",
                                            static_cast<int>(currentState)));
    }
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

void ProfilingService::Disconnect()
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    ProfilingState currentState = m_StateMachine.GetCurrentState();
    switch (currentState)
    {
    case ProfilingState::Uninitialised:
    case ProfilingState::NotConnected:
    case ProfilingState::WaitingForAck:
        return; // NOP
    case ProfilingState::Active:
        // Stop the command thread (if running)
        Stop();

        break;
    default:
        throw arm::pipe::ProfilingException(fmt::format("Unknown profiling service state: {}",
                                                        static_cast<int>(currentState)));
    }
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

// Store a profiling context returned from a backend that support profiling, and register its counters
void ProfilingService::AddBackendProfilingContext(
    const std::string& backendId,
    std::shared_ptr<IBackendProfilingContext> profilingContext)
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    ARM_PIPE_ASSERT(profilingContext != nullptr);
    // Register the backend counters
    m_MaxGlobalCounterId = profilingContext->RegisterCounters(m_MaxGlobalCounterId);
    m_BackendProfilingContexts.emplace(backendId, std::move(profilingContext));
#else
    IgnoreUnused(backendId);
    IgnoreUnused(profilingContext);
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}
const ICounterDirectory& ProfilingService::GetCounterDirectory() const
{
    return m_CounterDirectory;
}

ICounterRegistry& ProfilingService::GetCounterRegistry()
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

bool ProfilingService::IsCounterRegistered(uint16_t counterUid) const
{
    return m_CounterDirectory.IsCounterRegistered(counterUid);
}

uint32_t ProfilingService::GetAbsoluteCounterValue(uint16_t counterUid) const
{
    CheckCounterUid(counterUid);
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    ARM_PIPE_ASSERT(counterValuePtr);
    return counterValuePtr->load(std::memory_order::memory_order_relaxed);
}

uint32_t ProfilingService::GetDeltaCounterValue(uint16_t counterUid)
{
    CheckCounterUid(counterUid);
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    ARM_PIPE_ASSERT(counterValuePtr);
    const uint32_t counterValue = counterValuePtr->load(std::memory_order::memory_order_relaxed);
    SubtractCounterValue(counterUid, counterValue);
    return counterValue;
}

const ICounterMappings& ProfilingService::GetCounterMappings() const
{
    return m_CounterIdMap;
}

IRegisterCounterMapping& ProfilingService::GetCounterMappingRegistry()
{
    return m_CounterIdMap;
}

bool ProfilingService::IsCategoryRegistered(const std::string& categoryName) const
{
    return m_CounterDirectory.IsCategoryRegistered(categoryName);
}

bool ProfilingService::IsCounterRegistered(const std::string& counterName) const
{
    return m_CounterDirectory.IsCounterRegistered(counterName);
}

CaptureData ProfilingService::GetCaptureData()
{
    return m_Holder.GetCaptureData();
}

void ProfilingService::SetCaptureData(uint32_t capturePeriod,
                                      const std::vector<uint16_t>& counterIds,
                                      const std::set<std::string>& activeBackends)
{
    m_Holder.SetCaptureData(capturePeriod, counterIds, activeBackends);
}

void ProfilingService::SetCounterValue(uint16_t counterUid, uint32_t value)
{
    CheckCounterUid(counterUid);
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    ARM_PIPE_ASSERT(counterValuePtr);
    counterValuePtr->store(value, std::memory_order::memory_order_relaxed);
}

uint32_t ProfilingService::AddCounterValue(uint16_t counterUid, uint32_t value)
{
    CheckCounterUid(counterUid);
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    ARM_PIPE_ASSERT(counterValuePtr);
    return counterValuePtr->fetch_add(value, std::memory_order::memory_order_relaxed);
}

uint32_t ProfilingService::SubtractCounterValue(uint16_t counterUid, uint32_t value)
{
    CheckCounterUid(counterUid);
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    ARM_PIPE_ASSERT(counterValuePtr);
    return counterValuePtr->fetch_sub(value, std::memory_order::memory_order_relaxed);
}

uint32_t ProfilingService::IncrementCounterValue(uint16_t counterUid)
{
    CheckCounterUid(counterUid);
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    ARM_PIPE_ASSERT(counterValuePtr);
    return counterValuePtr->operator++(std::memory_order::memory_order_relaxed);
}

std::unique_ptr<ISendTimelinePacket> ProfilingService::GetSendTimelinePacket() const
{
    return m_TimelinePacketWriterFactory.GetSendTimelinePacket();
}

void ProfilingService::Initialize()
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    m_Initialiser.InitialiseProfilingService(*this);
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

void ProfilingService::InitializeCounterValue(uint16_t counterUid)
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    // Increase the size of the counter index if necessary
    if (counterUid >= m_CounterIndex.size())
    {
        m_CounterIndex.resize(arm::pipe::numeric_cast<size_t>(counterUid) + 1);
    }

    // Create a new atomic counter and add it to the list
    m_CounterValues.emplace_back(0);

    // Register the new counter to the counter index for quick access
    std::atomic<uint32_t>* counterValuePtr = &(m_CounterValues.back());
    m_CounterIndex.at(counterUid) = counterValuePtr;
#else
    IgnoreUnused(counterUid);
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

void ProfilingService::Reset()
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    // Stop the profiling service...
    Stop();

    // ...then delete all the counter data and configuration...
    m_CounterIndex.clear();
    m_CounterValues.clear();
    m_CounterDirectory.Clear();
    m_CounterIdMap.Reset();
    m_BufferManager.Reset();

    // ...finally reset the profiling state machine
    m_StateMachine.Reset();
    m_BackendProfilingContexts.clear();
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

void ProfilingService::Stop()
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    {   // only lock when we are updating the inference completed variable
        std::unique_lock<std::mutex> lck(m_ServiceActiveMutex);
        m_ServiceActive = false;
    }
    // The order in which we reset/stop the components is not trivial!
    // First stop the producing threads
    // Command Handler first as it is responsible for launching then Periodic Counter capture thread
    m_CommandHandler.Stop();
    m_PeriodicCounterCapture.Stop();
    // The the consuming thread
    m_SendThread.Stop(false);

    // ...then close and destroy the profiling connection...
    if (m_ProfilingConnection != nullptr && m_ProfilingConnection->IsOpen())
    {
        m_ProfilingConnection->Close();
    }
    m_ProfilingConnection.reset();

    // ...then move to the "NotConnected" state
    m_StateMachine.TransitionToState(ProfilingState::NotConnected);
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

inline void ProfilingService::CheckCounterUid(uint16_t counterUid) const
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    if (!IsCounterRegistered(counterUid))
    {
        throw arm::pipe::InvalidArgumentException(fmt::format("Counter UID {} is not registered", counterUid));
    }
#else
    IgnoreUnused(counterUid);
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

void ProfilingService::NotifyBackendsForTimelineReporting()
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    BackendProfilingContext::iterator it = m_BackendProfilingContexts.begin();
    while (it != m_BackendProfilingContexts.end())
    {
        auto& backendProfilingContext = it->second;
        backendProfilingContext->EnableTimelineReporting(m_TimelineReporting);
        // Increment the Iterator to point to next entry
        it++;
    }
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

void ProfilingService::NotifyProfilingServiceActive()
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    {   // only lock when we are updating the inference completed variable
        std::unique_lock<std::mutex> lck(m_ServiceActiveMutex);
        m_ServiceActive = true;
    }
    m_ServiceActiveConditionVariable.notify_one();
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

void ProfilingService::WaitForProfilingServiceActivation(unsigned int timeout)
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    std::unique_lock<std::mutex> lck(m_ServiceActiveMutex);

    auto start = std::chrono::high_resolution_clock::now();
    // Here we we will go back to sleep after a spurious wake up if
    // m_InferenceCompleted is not yet true.
    if (!m_ServiceActiveConditionVariable.wait_for(lck,
                                                   std::chrono::milliseconds(timeout),
                                                   [&]{return m_ServiceActive == true;}))
    {
        if (m_ServiceActive == true)
        {
            return;
        }
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = finish - start;
        std::stringstream ss;
        ss << "Timed out waiting on profiling service activation for " << elapsed.count() << " ms";
        ARM_PIPE_LOG(warning) << ss.str();
    }
#else
    IgnoreUnused(timeout);
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

ProfilingService::~ProfilingService()
{
#if !defined(ARMNN_BUILD_BARE_METAL) && !defined(ARMNN_EXECUTE_NETWORK_STATIC)
    Stop();
#endif // ARMNN_BUILD_BARE_METAL && ARMNN_EXECUTE_NETWORK_STATIC
}

} // namespace pipe

} // namespace arm
