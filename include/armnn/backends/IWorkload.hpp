//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>

namespace armnn
{

namespace experimental
{

struct WorkingMemDescriptor;

} // end experimental namespace

using namespace armnn::experimental;

/// Workload interface to enqueue a layer computation.
class IWorkload {
public:
    virtual ~IWorkload() {}

    virtual void PostAllocationConfigure() = 0;

    virtual void Execute() const = 0;

    virtual void ExecuteAsync(WorkingMemDescriptor& desc) = 0;

    virtual profiling::ProfilingGuid GetGuid() const = 0;

    virtual void RegisterDebugCallback(const DebugCallbackFunction& /*func*/) {}
};

} //namespace armnn
