//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BufferManager.hpp"
#include "CommandHandler.hpp"
#include "ConnectionAcknowledgedCommandHandler.hpp"
#include "CounterDirectory.hpp"
#include "CounterIdMap.hpp"
#include "ICounterRegistry.hpp"
#include "ICounterValues.hpp"
#include "IProfilingService.hpp"
#include "PeriodicCounterCapture.hpp"
#include "PeriodicCounterSelectionCommandHandler.hpp"
#include "PerJobCounterSelectionCommandHandler.hpp"
#include "ProfilingConnectionFactory.hpp"
#include "ProfilingGuidGenerator.hpp"
#include "ProfilingStateMachine.hpp"
#include "RequestCounterDirectoryCommandHandler.hpp"
#include "SendCounterPacket.hpp"
#include "SendThread.hpp"
#include "SendTimelinePacket.hpp"
#include "TimelinePacketWriterFactory.hpp"
#include <armnn/backends/profiling/IBackendProfilingContext.hpp>

namespace armnn
{

namespace profiling
{
// Static constants describing ArmNN's counter UID's
static const uint16_t NETWORK_LOADS         =   0;
static const uint16_t NETWORK_UNLOADS       =   1;
static const uint16_t REGISTERED_BACKENDS   =   2;
static const uint16_t UNREGISTERED_BACKENDS =   3;
static const uint16_t INFERENCES_RUN        =   4;
static const uint16_t MAX_ARMNN_COUNTER = INFERENCES_RUN;

class ProfilingService : public IReadWriteCounterValues, public IProfilingService
{
public:
    using ExternalProfilingOptions = IRuntime::CreationOptions::ExternalProfilingOptions;
    using IProfilingConnectionFactoryPtr = std::unique_ptr<IProfilingConnectionFactory>;
    using IProfilingConnectionPtr = std::unique_ptr<IProfilingConnection>;
    using CounterIndices = std::vector<std::atomic<uint32_t>*>;
    using CounterValues = std::list<std::atomic<uint32_t>>;

    // Getter for the singleton instance
    static ProfilingService& Instance()
    {
        static ProfilingService instance;
        return instance;
    }

    // Resets the profiling options, optionally clears the profiling service entirely
    void ResetExternalProfilingOptions(const ExternalProfilingOptions& options, bool resetProfilingService = false);
    ProfilingState ConfigureProfilingService(const ExternalProfilingOptions& options,
                                             bool resetProfilingService = false);


    // Updates the profiling service, making it transition to a new state if necessary
    void Update();

    // Disconnects the profiling service from the external server
    void Disconnect();

    // Store a profiling context returned from a backend that support profiling.
    void AddBackendProfilingContext(const BackendId backendId,
        std::shared_ptr<armnn::profiling::IBackendProfilingContext> profilingContext);

    const ICounterDirectory& GetCounterDirectory() const;
    ICounterRegistry& GetCounterRegistry();
    ProfilingState GetCurrentState() const;
    bool IsCounterRegistered(uint16_t counterUid) const override;
    uint32_t GetCounterValue(uint16_t counterUid) const override;
    uint16_t GetCounterCount() const override;
    // counter global/backend mapping functions
    const ICounterMappings& GetCounterMappings() const override;
    IRegisterCounterMapping& GetCounterMappingRegistry();

    // Getters for the profiling service state
    bool IsProfilingEnabled() const override;

    CaptureData GetCaptureData() override;
    void SetCaptureData(uint32_t capturePeriod,
                        const std::vector<uint16_t>& counterIds,
                        const std::set<BackendId>& activeBackends);

    // Setters for the profiling service state
    void SetCounterValue(uint16_t counterUid, uint32_t value) override;
    uint32_t AddCounterValue(uint16_t counterUid, uint32_t value) override;
    uint32_t SubtractCounterValue(uint16_t counterUid, uint32_t value) override;
    uint32_t IncrementCounterValue(uint16_t counterUid) override;

    // IProfilingGuidGenerator functions
    /// Return the next random Guid in the sequence
    ProfilingDynamicGuid NextGuid() override;
    /// Create a ProfilingStaticGuid based on a hash of the string
    ProfilingStaticGuid GenerateStaticId(const std::string& str) override;

    std::unique_ptr<ISendTimelinePacket> GetSendTimelinePacket() const override;

    ISendCounterPacket& GetSendCounterPacket() override
    {
        return m_SendCounterPacket;
    }

    /// Check if the profiling is enabled
    bool IsEnabled() { return m_Options.m_EnableProfiling; }

private:
    // Copy/move constructors/destructors and copy/move assignment operators are deleted
    ProfilingService(const ProfilingService&) = delete;
    ProfilingService(ProfilingService&&) = delete;
    ProfilingService& operator=(const ProfilingService&) = delete;
    ProfilingService& operator=(ProfilingService&&) = delete;

    // Initialization/reset functions
    void Initialize();
    void InitializeCounterValue(uint16_t counterUid);
    void Reset();
    void Stop();

    // Helper function
    void CheckCounterUid(uint16_t counterUid) const;

    // Profiling service components
    ExternalProfilingOptions m_Options;
    CounterDirectory m_CounterDirectory;
    CounterIdMap m_CounterIdMap;
    IProfilingConnectionFactoryPtr m_ProfilingConnectionFactory;
    IProfilingConnectionPtr m_ProfilingConnection;
    ProfilingStateMachine m_StateMachine;
    CounterIndices m_CounterIndex;
    CounterValues m_CounterValues;
    CommandHandlerRegistry m_CommandHandlerRegistry;
    PacketVersionResolver m_PacketVersionResolver;
    CommandHandler m_CommandHandler;
    BufferManager m_BufferManager;
    SendCounterPacket m_SendCounterPacket;
    SendThread m_SendThread;
    SendTimelinePacket m_SendTimelinePacket;
    Holder m_Holder;
    PeriodicCounterCapture m_PeriodicCounterCapture;
    ConnectionAcknowledgedCommandHandler m_ConnectionAcknowledgedCommandHandler;
    RequestCounterDirectoryCommandHandler m_RequestCounterDirectoryCommandHandler;
    PeriodicCounterSelectionCommandHandler m_PeriodicCounterSelectionCommandHandler;
    PerJobCounterSelectionCommandHandler m_PerJobCounterSelectionCommandHandler;
    ProfilingGuidGenerator m_GuidGenerator;
    TimelinePacketWriterFactory m_TimelinePacketWriterFactory;
    std::unordered_map<BackendId,
    std::shared_ptr<armnn::profiling::IBackendProfilingContext>> m_BackendProfilingContexts;
    uint16_t m_MaxGlobalCounterId;

protected:
    // Default constructor/destructor kept protected for testing
    ProfilingService()
        : m_Options()
        , m_CounterDirectory()
        , m_ProfilingConnectionFactory(new ProfilingConnectionFactory())
        , m_ProfilingConnection()
        , m_StateMachine()
        , m_CounterIndex()
        , m_CounterValues()
        , m_CommandHandlerRegistry()
        , m_PacketVersionResolver()
        , m_CommandHandler(1000,
                           false,
                           m_CommandHandlerRegistry,
                           m_PacketVersionResolver)
        , m_BufferManager()
        , m_SendCounterPacket(m_BufferManager)
        , m_SendThread(m_StateMachine, m_BufferManager, m_SendCounterPacket)
        , m_SendTimelinePacket(m_BufferManager)
        , m_PeriodicCounterCapture(m_Holder, m_SendCounterPacket, *this, m_CounterIdMap, m_BackendProfilingContexts)
        , m_ConnectionAcknowledgedCommandHandler(0,
                                                 1,
                                                 m_PacketVersionResolver.ResolvePacketVersion(0, 1).GetEncodedValue(),
                                                 m_CounterDirectory,
                                                 m_SendCounterPacket,
                                                 m_SendTimelinePacket,
                                                 m_StateMachine)
        , m_RequestCounterDirectoryCommandHandler(0,
                                                  3,
                                                  m_PacketVersionResolver.ResolvePacketVersion(0, 3).GetEncodedValue(),
                                                  m_CounterDirectory,
                                                  m_SendCounterPacket,
                                                  m_SendTimelinePacket,
                                                  m_StateMachine)
        , m_PeriodicCounterSelectionCommandHandler(0,
                                                   4,
                                                   m_PacketVersionResolver.ResolvePacketVersion(0, 4).GetEncodedValue(),
                                                   m_BackendProfilingContexts,
                                                   m_CounterIdMap,
                                                   m_Holder,
                                                   MAX_ARMNN_COUNTER,
                                                   m_PeriodicCounterCapture,
                                                   *this,
                                                   m_SendCounterPacket,
                                                   m_StateMachine)
        , m_PerJobCounterSelectionCommandHandler(0,
                                                 5,
                                                 m_PacketVersionResolver.ResolvePacketVersion(0, 5).GetEncodedValue(),
                                                 m_StateMachine)
        , m_TimelinePacketWriterFactory(m_BufferManager)
        , m_MaxGlobalCounterId(armnn::profiling::INFERENCES_RUN)
    {
        // Register the "Connection Acknowledged" command handler
        m_CommandHandlerRegistry.RegisterFunctor(&m_ConnectionAcknowledgedCommandHandler);

        // Register the "Request Counter Directory" command handler
        m_CommandHandlerRegistry.RegisterFunctor(&m_RequestCounterDirectoryCommandHandler);

        // Register the "Periodic Counter Selection" command handler
        m_CommandHandlerRegistry.RegisterFunctor(&m_PeriodicCounterSelectionCommandHandler);

        // Register the "Per-Job Counter Selection" command handler
        m_CommandHandlerRegistry.RegisterFunctor(&m_PerJobCounterSelectionCommandHandler);
    }
    ~ProfilingService();

    // Protected methods for testing
    void SwapProfilingConnectionFactory(ProfilingService& instance,
                                        IProfilingConnectionFactory* other,
                                        IProfilingConnectionFactory*& backup)
    {
        BOOST_ASSERT(instance.m_ProfilingConnectionFactory);
        BOOST_ASSERT(other);

        backup = instance.m_ProfilingConnectionFactory.release();
        instance.m_ProfilingConnectionFactory.reset(other);
    }
    IProfilingConnection* GetProfilingConnection(ProfilingService& instance)
    {
        return instance.m_ProfilingConnection.get();
    }
    void TransitionToState(ProfilingService& instance, ProfilingState newState)
    {
        instance.m_StateMachine.TransitionToState(newState);
    }
    bool WaitForPacketSent(ProfilingService& instance, uint32_t timeout = 1000)
    {
        return instance.m_SendThread.WaitForPacketSent(timeout);
    }

    BufferManager& GetBufferManager(ProfilingService& instance)
    {
        return instance.m_BufferManager;
    }
};

} // namespace profiling

} // namespace armnn
