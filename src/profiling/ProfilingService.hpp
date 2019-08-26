//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingStateMachine.hpp"
#include "ProfilingConnectionFactory.hpp"

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

    ProfilingState GetCurrentState() const;

    // Options are public to allow profiling to be turned on at runtime
    Runtime::CreationOptions::ExternalProfilingOptions m_Options;

private:
    void Initialise();

    ProfilingStateMachine m_State;
    ProfilingConnectionFactory m_Factory;
};

} // namespace profiling

} // namespace armnn