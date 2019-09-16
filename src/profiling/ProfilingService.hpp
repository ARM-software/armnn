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

class ProfilingService : IWriteCounterValues
{
public:
    ProfilingService(const Runtime::CreationOptions::ExternalProfilingOptions& options);
    ~ProfilingService() = default;

    void Run();

    const ICounterDirectory& GetCounterDirectory() const;
    ProfilingState GetCurrentState() const;
    void ResetExternalProfilingOptions(const Runtime::CreationOptions::ExternalProfilingOptions& options);

    uint16_t GetCounterCount() const;
    void GetCounterValue(uint16_t index, uint32_t& value) const;
    void SetCounterValue(uint16_t index, uint32_t value);
    void AddCounterValue(uint16_t index, uint32_t value);
    void SubtractCounterValue(uint16_t index, uint32_t value);
    void IncrementCounterValue(uint16_t index);
    void DecrementCounterValue(uint16_t index);

private:
    void Initialise();
    void CheckIndexSize(uint16_t counterIndex) const;

    CounterDirectory m_CounterDirectory;
    ProfilingConnectionFactory m_Factory;
    Runtime::CreationOptions::ExternalProfilingOptions m_Options;
    ProfilingStateMachine m_State;

    std::unordered_map<uint16_t, std::atomic<uint32_t>> m_CounterIdToValue;
};

} // namespace profiling

} // namespace armnn