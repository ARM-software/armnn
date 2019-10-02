//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingStateMachine.hpp"
#include "ProfilingConnectionFactory.hpp"
#include "CounterDirectory.hpp"
#include "CounterValues.hpp"

namespace armnn
{

namespace profiling
{

class ProfilingService final : public IReadWriteCounterValues
{
public:
    using ExternalProfilingOptions = Runtime::CreationOptions::ExternalProfilingOptions;
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

    // Runs the profiling service
    void Run();

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
    // Default/copy/move constructors/destructors and copy/move assignment operators are kept private
    ProfilingService() = default;
    ProfilingService(const ProfilingService&) = delete;
    ProfilingService(ProfilingService&&) = delete;
    ProfilingService& operator=(const ProfilingService&) = delete;
    ProfilingService& operator=(ProfilingService&&) = delete;
    ~ProfilingService() = default;

    // Initialization functions
    void Initialize();
    void InitializeCounterValue(uint16_t counterUid);

    // Profiling service state variables
    ExternalProfilingOptions m_Options;
    CounterDirectory m_CounterDirectory;
    ProfilingConnectionFactory m_ProfilingConnectionFactory;
    IProfilingConnectionPtr m_ProfilingConnection;
    ProfilingStateMachine m_StateMachine;
    CounterIndices m_CounterIndex;
    CounterValues m_CounterValues;
};

} // namespace profiling

} // namespace armnn
