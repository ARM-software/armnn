//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingStateMachine.hpp"
#include "ProfilingConnectionFactory.hpp"
#include "CounterDirectory.hpp"

namespace armnn
{

namespace profiling
{

class ProfilingService
{
public:
    ProfilingService(const Runtime::CreationOptions::ExternalProfilingOptions& options);
    ~ProfilingService() = default;

    void Run();

    const ICounterDirectory& GetCounterDirectory() const;
    ProfilingState GetCurrentState() const;
    void ResetExternalProfilingOptions(const Runtime::CreationOptions::ExternalProfilingOptions& options);

private:
    void Initialise();

    CounterDirectory m_CounterDirectory;
    ProfilingConnectionFactory m_Factory;
    Runtime::CreationOptions::ExternalProfilingOptions m_Options;
    ProfilingStateMachine m_State;
};

} // namespace profiling

} // namespace armnn