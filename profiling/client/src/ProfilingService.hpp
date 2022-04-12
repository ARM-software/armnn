//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ActivateTimelineReportingCommandHandler.hpp"
#include "BufferManager.hpp"
#include "CommandHandler.hpp"
#include "ConnectionAcknowledgedCommandHandler.hpp"
#include "DeactivateTimelineReportingCommandHandler.hpp"
#include "INotifyBackends.hpp"
#include "PeriodicCounterCapture.hpp"
#include "PeriodicCounterSelectionCommandHandler.hpp"
#include "PerJobCounterSelectionCommandHandler.hpp"
#include "ProfilingConnectionFactory.hpp"
#include "ProfilingStateMachine.hpp"
#include "RequestCounterDirectoryCommandHandler.hpp"
#include "SendCounterPacket.hpp"
#include "SendThread.hpp"
#include "SendTimelinePacket.hpp"
#include "TimelinePacketWriterFactory.hpp"

#include <client/include/CounterIdMap.hpp>
#include <client/include/ICounterValues.hpp>
#include <client/include/ILocalPacketHandler.hpp>
#include <client/include/IProfilingService.hpp>
#include <client/include/IReportStructure.hpp>

#include <client/include/backends/IBackendProfilingContext.hpp>

#include <common/include/CounterDirectory.hpp>

#include <list>

namespace arm
{

namespace pipe
{

class ProfilingService : public IProfilingService, public INotifyBackends
{
public:
    using IProfilingConnectionFactoryPtr = std::unique_ptr<IProfilingConnectionFactory>;
    using IProfilingConnectionPtr = std::unique_ptr<IProfilingConnection>;
    using CounterIndices = std::vector<std::atomic<uint32_t>*>;
    using CounterValues = std::list<std::atomic<uint32_t>>;
    using BackendProfilingContext = std::unordered_map<std::string,
                                                       std::shared_ptr<IBackendProfilingContext>>;

    ProfilingService(uint16_t maxGlobalCounterId,
                     IInitialiseProfilingService& initialiser,
                     const std::string& softwareInfo,
                     const std::string& softwareVersion,
                     const std::string& hardwareVersion,
                     arm::pipe::Optional<IReportStructure&> reportStructure = arm::pipe::EmptyOptional())
        : m_Options()
        , m_TimelineReporting(false)
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
        , m_SendCounterPacket(m_BufferManager, softwareInfo, softwareVersion, hardwareVersion)
        , m_SendThread(m_StateMachine, m_BufferManager, m_SendCounterPacket)
        , m_SendTimelinePacket(m_BufferManager)
        , m_PeriodicCounterCapture(m_Holder, m_SendCounterPacket, *this, m_CounterIdMap, m_BackendProfilingContexts)
        , m_ConnectionAcknowledgedCommandHandler(0,
                                                 1,
                                                 m_PacketVersionResolver.ResolvePacketVersion(0, 1).GetEncodedValue(),
                                                 m_CounterDirectory,
                                                 m_SendCounterPacket,
                                                 m_SendTimelinePacket,
                                                 m_StateMachine,
                                                 *this,
                                                 m_BackendProfilingContexts)
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
                                                   maxGlobalCounterId,
                                                   m_PeriodicCounterCapture,
                                                   *this,
                                                   m_SendCounterPacket,
                                                   m_StateMachine)
        , m_PerJobCounterSelectionCommandHandler(0,
                                                 5,
                                                 m_PacketVersionResolver.ResolvePacketVersion(0, 5).GetEncodedValue(),
                                                 m_StateMachine)
        , m_ActivateTimelineReportingCommandHandler(0,
                                                    6,
                                                    m_PacketVersionResolver.ResolvePacketVersion(0, 6)
                                                                           .GetEncodedValue(),
                                                    m_SendTimelinePacket,
                                                    m_StateMachine,
                                                    reportStructure,
                                                    m_TimelineReporting,
                                                    *this,
                                                    *this)
        , m_DeactivateTimelineReportingCommandHandler(0,
                                                      7,
                                                      m_PacketVersionResolver.ResolvePacketVersion(0, 7)
                                                                             .GetEncodedValue(),
                                                      m_TimelineReporting,
                                                      m_StateMachine,
                                                      *this)
        , m_TimelinePacketWriterFactory(m_BufferManager)
        , m_MaxGlobalCounterId(maxGlobalCounterId)
        , m_ServiceActive(false)
        , m_Initialiser(initialiser)
    {
        // Register the "Connection Acknowledged" command handler
        m_CommandHandlerRegistry.RegisterFunctor(&m_ConnectionAcknowledgedCommandHandler);

        // Register the "Request Counter Directory" command handler
        m_CommandHandlerRegistry.RegisterFunctor(&m_RequestCounterDirectoryCommandHandler);

        // Register the "Periodic Counter Selection" command handler
        m_CommandHandlerRegistry.RegisterFunctor(&m_PeriodicCounterSelectionCommandHandler);

        // Register the "Per-Job Counter Selection" command handler
        m_CommandHandlerRegistry.RegisterFunctor(&m_PerJobCounterSelectionCommandHandler);

        m_CommandHandlerRegistry.RegisterFunctor(&m_ActivateTimelineReportingCommandHandler);

        m_CommandHandlerRegistry.RegisterFunctor(&m_DeactivateTimelineReportingCommandHandler);
    }

    ~ProfilingService();

    // Resets the profiling options, optionally clears the profiling service entirely
    void ResetExternalProfilingOptions(const ProfilingOptions& options,
                                       bool resetProfilingService = false) override;
    ProfilingState ConfigureProfilingService(const ProfilingOptions& options,
                                             bool resetProfilingService = false) override;


    // Updates the profiling service, making it transition to a new state if necessary
    void Update();

    // Disconnects the profiling service from the external server
    void Disconnect() override;

    // Store a profiling context returned from a backend that support profiling.
    void AddBackendProfilingContext(const std::string& backendId,
        std::shared_ptr<IBackendProfilingContext> profilingContext) override;

    // Enable the recording of timeline events and entities
    void NotifyBackendsForTimelineReporting() override;

    const ICounterDirectory& GetCounterDirectory() const;
    ICounterRegistry& GetCounterRegistry() override;
    ProfilingState GetCurrentState() const override;
    bool IsCounterRegistered(uint16_t counterUid) const override;
    uint32_t GetAbsoluteCounterValue(uint16_t counterUid) const override;
    uint32_t GetDeltaCounterValue(uint16_t counterUid) override;
    uint16_t GetCounterCount() const override;
    // counter global/backend mapping functions
    const ICounterMappings& GetCounterMappings() const override;
    IRegisterCounterMapping& GetCounterMappingRegistry() override;
    bool IsCategoryRegistered(const std::string& categoryName) const override;
    bool IsCounterRegistered(const std::string& counterName) const override;

    // Getters for the profiling service state
    bool IsProfilingEnabled() const override;

    CaptureData GetCaptureData() override;
    void SetCaptureData(uint32_t capturePeriod,
                        const std::vector<uint16_t>& counterIds,
                        const std::set<std::string>& activeBackends);

    // Setters for the profiling service state
    void SetCounterValue(uint16_t counterUid, uint32_t value) override;
    uint32_t AddCounterValue(uint16_t counterUid, uint32_t value) override;
    uint32_t SubtractCounterValue(uint16_t counterUid, uint32_t value) override;
    uint32_t IncrementCounterValue(uint16_t counterUid) override;

    void InitializeCounterValue(uint16_t counterUid) override;

    std::unique_ptr<ISendTimelinePacket> GetSendTimelinePacket() const override;

    ISendCounterPacket& GetSendCounterPacket() override
    {
        return m_SendCounterPacket;
    }

    bool IsTimelineReportingEnabled() const override
    {
        return m_TimelineReporting;
    }

    void AddLocalPacketHandler(ILocalPacketHandlerSharedPtr localPacketHandler);

    void NotifyProfilingServiceActive() override; // IProfilingServiceStatus
    void WaitForProfilingServiceActivation(unsigned int timeout) override; // IProfilingServiceStatus

private:
    //Copy/move constructors/destructors and copy/move assignment operators are deleted
    ProfilingService(const ProfilingService&) = delete;
    ProfilingService(ProfilingService&&) = delete;
    ProfilingService& operator=(const ProfilingService&) = delete;
    ProfilingService& operator=(ProfilingService&&) = delete;

    // Initialization/reset functions
    void Initialize();
    void Reset();
    void Stop();

    // Helper function
    void CheckCounterUid(uint16_t counterUid) const;

    // Profiling service components
    ProfilingOptions                   m_Options;
    std::atomic<bool>                  m_TimelineReporting;
    CounterDirectory                   m_CounterDirectory;
    CounterIdMap                       m_CounterIdMap;
    IProfilingConnectionFactoryPtr     m_ProfilingConnectionFactory;
    IProfilingConnectionPtr            m_ProfilingConnection;
    ProfilingStateMachine              m_StateMachine;
    CounterIndices                     m_CounterIndex;
    CounterValues                      m_CounterValues;
    arm::pipe::CommandHandlerRegistry  m_CommandHandlerRegistry;
    arm::pipe::PacketVersionResolver   m_PacketVersionResolver;
    CommandHandler                     m_CommandHandler;
    BufferManager                      m_BufferManager;
    SendCounterPacket                  m_SendCounterPacket;
    SendThread                         m_SendThread;
    SendTimelinePacket                 m_SendTimelinePacket;

    Holder m_Holder;

    PeriodicCounterCapture m_PeriodicCounterCapture;

    ConnectionAcknowledgedCommandHandler      m_ConnectionAcknowledgedCommandHandler;
    RequestCounterDirectoryCommandHandler     m_RequestCounterDirectoryCommandHandler;
    PeriodicCounterSelectionCommandHandler    m_PeriodicCounterSelectionCommandHandler;
    PerJobCounterSelectionCommandHandler      m_PerJobCounterSelectionCommandHandler;
    ActivateTimelineReportingCommandHandler   m_ActivateTimelineReportingCommandHandler;
    DeactivateTimelineReportingCommandHandler m_DeactivateTimelineReportingCommandHandler;

    TimelinePacketWriterFactory m_TimelinePacketWriterFactory;
    BackendProfilingContext     m_BackendProfilingContexts;
    uint16_t                    m_MaxGlobalCounterId;

    // Signalling to let external actors know when service is active or not
#if !defined(ARMNN_DISABLE_THREADS)
    std::mutex m_ServiceActiveMutex;
    std::condition_variable m_ServiceActiveConditionVariable;
#endif
    bool m_ServiceActive;

    IInitialiseProfilingService& m_Initialiser;

protected:

    // Protected methods for testing
    void SwapProfilingConnectionFactory(ProfilingService& instance,
                                        IProfilingConnectionFactory* other,
                                        IProfilingConnectionFactory*& backup)
    {
        ARM_PIPE_ASSERT(instance.m_ProfilingConnectionFactory);
        ARM_PIPE_ASSERT(other);

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

} // namespace pipe

} // namespace arm
