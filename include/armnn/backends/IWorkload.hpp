//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>

namespace armnn
{

/// Workload interface to enqueue a layer computation.
class IWorkload {
public:
    virtual ~IWorkload() {}

    virtual void PostAllocationConfigure() = 0;

    virtual void Execute() const = 0;

    virtual profiling::ProfilingGuid GetGuid() const = 0;

    virtual void RegisterDebugCallback(const DebugCallbackFunction & /*func*/) {}
};

} //namespace armnn
