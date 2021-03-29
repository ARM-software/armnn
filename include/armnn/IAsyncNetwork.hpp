//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/NetworkFwd.hpp>

#include "INetwork.hpp"
#include "IProfiler.hpp"
#include "IWorkingMemHandle.hpp"
#include "Tensor.hpp"
#include "Types.hpp"

#include <mutex>

namespace armnn
{

namespace experimental
{

class IAsyncNetwork
{
public:
    virtual ~IAsyncNetwork() {};

    virtual TensorInfo GetInputTensorInfo(LayerBindingId layerId) const = 0;
    virtual TensorInfo GetOutputTensorInfo(LayerBindingId layerId) const = 0;

    /// Thread safe execution of the network. Returns once execution is complete.
    /// Will block until this and any other thread using the same workingMem object completes.
    virtual Status Execute(const InputTensors& inputTensors,
                           const OutputTensors& outputTensors,
                           IWorkingMemHandle& workingMemHandle) = 0;

    /// Create a new unique WorkingMemHandle object. Create multiple handles if you wish to have
    /// overlapped Execution by calling this function from different threads.
    virtual std::unique_ptr<IWorkingMemHandle> CreateWorkingMemHandle() = 0;

    /// Get the profiler used for this network
    virtual std::shared_ptr<IProfiler> GetProfiler() const = 0;

    /// Register a debug callback function to be used with this network
    virtual void RegisterDebugCallback(const DebugCallbackFunction& func) = 0;
};

} // end experimental namespace

} // end armnn namespace
