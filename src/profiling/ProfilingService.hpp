//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingStateMachine.hpp"
#include "ProfilingConnectionFactory.hpp"
#include "CounterDirectory.hpp"
#include "ICounterValues.hpp"
#include "CommandHandler.hpp"
#include "BufferManager.hpp"
#include "SendCounterPacket.hpp"
#include "PeriodicCounterCapture.hpp"
#include "ConnectionAcknowledgedCommandHandler.hpp"
#include "RequestCounterDirectoryCommandHandler.hpp"
#include "PeriodicCounterSelectionCommandHandler.hpp"
#include "PerJobCounterSelectionCommandHandler.hpp"

namespace armnn
{

namespace profiling
{

class ProfilingService : public IReadWriteCounterValues
{
public:
    using ExternalProfilingOptions = Runtime::CreationOptions::ExternalProfilingOptions;
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

    // Getters for the profiling service state
    const ICounterDirectory& GetCounterDirectory() const;
    ProfilingState GetCurrentState() const;
    bool IsCounterRegistered(uint16_t counterUid) const override;
    uint16_t GetCounterCount() const override;
    uint32_t GetCounterValue(uint16_t counterUid) const override;

    // Setters for the profiling service state
    void SetCounterValue(uint16_t counterUid, uint32_t value) override;
    uint32_t AddCounterValue(uint16_t counterUid, uint32_t value) override;
    uint32_t SubtractCounterValue(uint16_t counterUid, uint32_t value) override;
    uint32_t IncrementCounterValue(uint16_t counterUid) override;
    uint32_t DecrementCounterValue(uint16_t counterUid) override;

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
    Holder m_Holder;
    PeriodicCounterCapture m_PeriodicCounterCapture;
    ConnectionAcknowledgedCommandHandler m_ConnectionAcknowledgedCommandHandler;
    RequestCounterDirectoryCommandHandler m_RequestCounterDirectoryCommandHandler;
    PeriodicCounterSelectionCommandHandler m_PeriodicCounterSelectionCommandHandler;
    PerJobCounterSelectionCommandHandler m_PerJobCounterSelectionCommandHandler;

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
        , m_SendCounterPacket(m_StateMachine, m_BufferManager)
        , m_PeriodicCounterCapture(m_Holder, m_SendCounterPacket, *this)
        , m_ConnectionAcknowledgedCommandHandler(1,
                                                 m_PacketVersionResolver.ResolvePacketVersion(1).GetEncodedValue(),
                                                 m_StateMachine)
        , m_RequestCounterDirectoryCommandHandler(3,
                                                  m_PacketVersionResolver.ResolvePacketVersion(3).GetEncodedValue(),
                                                  m_CounterDirectory,
                                                  m_SendCounterPacket,
                                                  m_StateMachine)
        , m_PeriodicCounterSelectionCommandHandler(4,
                                                   m_PacketVersionResolver.ResolvePacketVersion(4).GetEncodedValue(),
                                                   m_Holder,
                                                   m_PeriodicCounterCapture,
                                                   *this,
                                                   m_SendCounterPacket,
                                                   m_StateMachine)
        , m_PerJobCounterSelectionCommandHandler(5,
                                                 m_PacketVersionResolver.ResolvePacketVersion(4).GetEncodedValue(),
                                                 m_StateMachine)
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
    ~ProfilingService() = default;

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
    void WaitForPacketSent(ProfilingService& instance)
    {
        return instance.m_SendCounterPacket.WaitForPacketSent();
    }
};

} // namespace profiling

} // namespace armnn
