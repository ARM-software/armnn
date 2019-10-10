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
#include "ConnectionAcknowledgedCommandHandler.hpp"
#include "RequestCounterDirectoryCommandHandler.hpp"

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

    // Updates the profiling service, making it transition to a new state if necessary
    void Update();

    // Getters for the profiling service state
    const ICounterDirectory& GetCounterDirectory() const;
    ProfilingState GetCurrentState() const;
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
    ConnectionAcknowledgedCommandHandler m_ConnectionAcknowledgedCommandHandler;
    RequestCounterDirectoryCommandHandler m_RequestCounterDirectoryCommandHandler;

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
        , m_ConnectionAcknowledgedCommandHandler(1,
                                                 m_PacketVersionResolver.ResolvePacketVersion(1).GetEncodedValue(),
                                                 m_StateMachine)
        , m_RequestCounterDirectoryCommandHandler(3,
                                                  m_PacketVersionResolver.ResolvePacketVersion(3).GetEncodedValue(),
                                                  m_CounterDirectory,
                                                  m_SendCounterPacket,
                                                  m_StateMachine)
    {
        // Register the "Connection Acknowledged" command handler
        m_CommandHandlerRegistry.RegisterFunctor(&m_ConnectionAcknowledgedCommandHandler);

        // Register the "Request Counter Directory" command handler
        m_CommandHandlerRegistry.RegisterFunctor(&m_RequestCounterDirectoryCommandHandler);
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
};

} // namespace profiling

} // namespace armnn
