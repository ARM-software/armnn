//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingService.hpp"

#include <armnn/BackendId.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <common/include/ProfilingGuid.hpp>

#include <common/include/SocketConnectionException.hpp>

#include <fmt/format.h>

namespace armnn
{

namespace profiling
{

ProfilingGuidGenerator ProfilingService::m_GuidGenerator;

ProfilingDynamicGuid ProfilingService::GetNextGuid()
{
    return m_GuidGenerator.NextGuid();
}

ProfilingStaticGuid ProfilingService::GetStaticId(const std::string& str)
{
    return m_GuidGenerator.GenerateStaticId(str);
}

void ProfilingService::ResetGuidGenerator()
{
    m_GuidGenerator.Reset();
}

void ProfilingService::ResetExternalProfilingOptions(const ExternalProfilingOptions& options,
                                                     bool resetProfilingService)
{
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
}

bool ProfilingService::IsProfilingEnabled() const
{
    return m_Options.m_EnableProfiling;
}

ProfilingState ProfilingService::ConfigureProfilingService(
        const ExternalProfilingOptions& options,
        bool resetProfilingService)
{
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
        m_SendThread.Stop(false);

        // Stop the periodic counter capture thread (if running)
        m_PeriodicCounterCapture.Stop();

        // Reset any existing profiling connection
        m_ProfilingConnection.reset();

        try
        {
            // Setup the profiling connection
            ARMNN_ASSERT(m_ProfilingConnectionFactory);
            m_ProfilingConnection = m_ProfilingConnectionFactory->GetProfilingConnection(m_Options);
        }
        catch (const Exception& e)
        {
            ARMNN_LOG(warning) << "An error has occurred when creating the profiling connection: "
                                       << e.what();
        }
        catch (const arm::pipe::SocketConnectionException& e)
        {
            ARMNN_LOG(warning) << "An error has occurred when creating the profiling connection ["
                                       << e.what() << "] on socket [" << e.GetSocketFd() << "].";
        }

        // Move to the next state
        m_StateMachine.TransitionToState(m_ProfilingConnection
                                         ? ProfilingState::WaitingForAck  // Profiling connection obtained, wait for ack
                                         : ProfilingState::NotConnected); // Profiling connection failed, stay in the
                                                                          // "NotConnected" state
        break;
    case ProfilingState::WaitingForAck:
        ARMNN_ASSERT(m_ProfilingConnection);

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
        throw RuntimeException(fmt::format("Unknown profiling service state: {}",
                                           static_cast<int>(currentState)));
    }
}

void ProfilingService::Disconnect()
{
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
        throw RuntimeException(fmt::format("Unknown profiling service state: {}",
                                           static_cast<int>(currentState)));
    }
}

// Store a profiling context returned from a backend that support profiling, and register its counters
void ProfilingService::AddBackendProfilingContext(const BackendId backendId,
    std::shared_ptr<armnn::profiling::IBackendProfilingContext> profilingContext)
{
    ARMNN_ASSERT(profilingContext != nullptr);
    // Register the backend counters
    m_MaxGlobalCounterId = profilingContext->RegisterCounters(m_MaxGlobalCounterId);
    m_BackendProfilingContexts.emplace(backendId, std::move(profilingContext));
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
    ARMNN_ASSERT(counterValuePtr);
    return counterValuePtr->load(std::memory_order::memory_order_relaxed);
}

uint32_t ProfilingService::GetDeltaCounterValue(uint16_t counterUid)
{
    CheckCounterUid(counterUid);
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    ARMNN_ASSERT(counterValuePtr);
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

CaptureData ProfilingService::GetCaptureData()
{
    return m_Holder.GetCaptureData();
}

void ProfilingService::SetCaptureData(uint32_t capturePeriod,
                                      const std::vector<uint16_t>& counterIds,
                                      const std::set<BackendId>& activeBackends)
{
    m_Holder.SetCaptureData(capturePeriod, counterIds, activeBackends);
}

void ProfilingService::SetCounterValue(uint16_t counterUid, uint32_t value)
{
    CheckCounterUid(counterUid);
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    ARMNN_ASSERT(counterValuePtr);
    counterValuePtr->store(value, std::memory_order::memory_order_relaxed);
}

uint32_t ProfilingService::AddCounterValue(uint16_t counterUid, uint32_t value)
{
    CheckCounterUid(counterUid);
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    ARMNN_ASSERT(counterValuePtr);
    return counterValuePtr->fetch_add(value, std::memory_order::memory_order_relaxed);
}

uint32_t ProfilingService::SubtractCounterValue(uint16_t counterUid, uint32_t value)
{
    CheckCounterUid(counterUid);
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    ARMNN_ASSERT(counterValuePtr);
    return counterValuePtr->fetch_sub(value, std::memory_order::memory_order_relaxed);
}

uint32_t ProfilingService::IncrementCounterValue(uint16_t counterUid)
{
    CheckCounterUid(counterUid);
    std::atomic<uint32_t>* counterValuePtr = m_CounterIndex.at(counterUid);
    ARMNN_ASSERT(counterValuePtr);
    return counterValuePtr->operator++(std::memory_order::memory_order_relaxed);
}

ProfilingDynamicGuid ProfilingService::NextGuid()
{
    return ProfilingService::GetNextGuid();
}

ProfilingStaticGuid ProfilingService::GenerateStaticId(const std::string& str)
{
    return ProfilingService::GetStaticId(str);
}

std::unique_ptr<ISendTimelinePacket> ProfilingService::GetSendTimelinePacket() const
{
    return m_TimelinePacketWriterFactory.GetSendTimelinePacket();
}

void ProfilingService::Initialize()
{
    // Register a category for the basic runtime counters
    if (!m_CounterDirectory.IsCategoryRegistered("ArmNN_Runtime"))
    {
        m_CounterDirectory.RegisterCategory("ArmNN_Runtime");
    }

    // Register a counter for the number of Network loads
    if (!m_CounterDirectory.IsCounterRegistered("Network loads"))
    {
        const Counter* loadedNetworksCounter =
                m_CounterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                   armnn::profiling::NETWORK_LOADS,
                                                   "ArmNN_Runtime",
                                                   0,
                                                   0,
                                                   1.f,
                                                   "Network loads",
                                                   "The number of networks loaded at runtime",
                                                   std::string("networks"));
        ARMNN_ASSERT(loadedNetworksCounter);
        InitializeCounterValue(loadedNetworksCounter->m_Uid);
    }
    // Register a counter for the number of unloaded networks
    if (!m_CounterDirectory.IsCounterRegistered("Network unloads"))
    {
        const Counter* unloadedNetworksCounter =
                m_CounterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                   armnn::profiling::NETWORK_UNLOADS,
                                                   "ArmNN_Runtime",
                                                   0,
                                                   0,
                                                   1.f,
                                                   "Network unloads",
                                                   "The number of networks unloaded at runtime",
                                                   std::string("networks"));
        ARMNN_ASSERT(unloadedNetworksCounter);
        InitializeCounterValue(unloadedNetworksCounter->m_Uid);
    }
    // Register a counter for the number of registered backends
    if (!m_CounterDirectory.IsCounterRegistered("Backends registered"))
    {
        const Counter* registeredBackendsCounter =
                m_CounterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                   armnn::profiling::REGISTERED_BACKENDS,
                                                   "ArmNN_Runtime",
                                                   0,
                                                   0,
                                                   1.f,
                                                   "Backends registered",
                                                   "The number of registered backends",
                                                   std::string("backends"));
        ARMNN_ASSERT(registeredBackendsCounter);
        InitializeCounterValue(registeredBackendsCounter->m_Uid);

        // Due to backends being registered before the profiling service becomes active,
        // we need to set the counter to the correct value here
        SetCounterValue(armnn::profiling::REGISTERED_BACKENDS, static_cast<uint32_t>(BackendRegistryInstance().Size()));
    }
    // Register a counter for the number of registered backends
    if (!m_CounterDirectory.IsCounterRegistered("Backends unregistered"))
    {
        const Counter* unregisteredBackendsCounter =
                m_CounterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                   armnn::profiling::UNREGISTERED_BACKENDS,
                                                   "ArmNN_Runtime",
                                                   0,
                                                   0,
                                                   1.f,
                                                   "Backends unregistered",
                                                   "The number of unregistered backends",
                                                   std::string("backends"));
        ARMNN_ASSERT(unregisteredBackendsCounter);
        InitializeCounterValue(unregisteredBackendsCounter->m_Uid);
    }
    // Register a counter for the number of inferences run
    if (!m_CounterDirectory.IsCounterRegistered("Inferences run"))
    {
        const Counter* inferencesRunCounter =
                m_CounterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                   armnn::profiling::INFERENCES_RUN,
                                                   "ArmNN_Runtime",
                                                   0,
                                                   0,
                                                   1.f,
                                                   "Inferences run",
                                                   "The number of inferences run",
                                                   std::string("inferences"));
        ARMNN_ASSERT(inferencesRunCounter);
        InitializeCounterValue(inferencesRunCounter->m_Uid);
    }
}

void ProfilingService::InitializeCounterValue(uint16_t counterUid)
{
    // Increase the size of the counter index if necessary
    if (counterUid >= m_CounterIndex.size())
    {
        m_CounterIndex.resize(armnn::numeric_cast<size_t>(counterUid) + 1);
    }

    // Create a new atomic counter and add it to the list
    m_CounterValues.emplace_back(0);

    // Register the new counter to the counter index for quick access
    std::atomic<uint32_t>* counterValuePtr = &(m_CounterValues.back());
    m_CounterIndex.at(counterUid) = counterValuePtr;
}

void ProfilingService::Reset()
{
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
    m_MaxGlobalCounterId = armnn::profiling::MAX_ARMNN_COUNTER;
}

void ProfilingService::Stop()
{
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
}

inline void ProfilingService::CheckCounterUid(uint16_t counterUid) const
{
    if (!IsCounterRegistered(counterUid))
    {
        throw InvalidArgumentException(fmt::format("Counter UID {} is not registered", counterUid));
    }
}

void ProfilingService::NotifyBackendsForTimelineReporting()
{
    BackendProfilingContext::iterator it = m_BackendProfilingContexts.begin();
    while (it != m_BackendProfilingContexts.end())
    {
        auto& backendProfilingContext = it->second;
        backendProfilingContext->EnableTimelineReporting(m_TimelineReporting);
        // Increment the Iterator to point to next entry
        it++;
    }
}

void ProfilingService::NotifyProfilingServiceActive()
{
    {   // only lock when we are updating the inference completed variable
        std::unique_lock<std::mutex> lck(m_ServiceActiveMutex);
        m_ServiceActive = true;
    }
    m_ServiceActiveConditionVariable.notify_one();
}

void ProfilingService::WaitForProfilingServiceActivation(unsigned int timeout)
{
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
        ARMNN_LOG(warning) << ss.str();
    }
    return;
}

ProfilingService::~ProfilingService()
{
    Stop();
}
} // namespace profiling

} // namespace armnn
