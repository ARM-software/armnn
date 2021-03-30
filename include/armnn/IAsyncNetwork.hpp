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
struct INetworkProperties;

namespace profiling
{
class ProfilingService;
}

namespace experimental
{
class AsyncNetworkImpl;

class IAsyncNetwork
{
public:
    IAsyncNetwork(std::unique_ptr<IOptimizedNetwork> net,
                  const INetworkProperties& networkProperties,
                  profiling::ProfilingService& profilingService);
    ~IAsyncNetwork();

    TensorInfo GetInputTensorInfo(LayerBindingId layerId) const;
    TensorInfo GetOutputTensorInfo(LayerBindingId layerId) const;

    /// Thread safe execution of the network. Returns once execution is complete.
    /// Will block until this and any other thread using the same workingMem object completes.
    Status Execute(const InputTensors& inputTensors,
                   const OutputTensors& outputTensors,
                   IWorkingMemHandle& workingMemHandle);

    /// Create a new unique WorkingMemHandle object. Create multiple handles if you wish to have
    /// overlapped Execution by calling this function from different threads.
    std::unique_ptr<IWorkingMemHandle> CreateWorkingMemHandle();

    /// Get the profiler used for this network
    std::shared_ptr<IProfiler> GetProfiler() const;

    /// Register a debug callback function to be used with this network
    void RegisterDebugCallback(const DebugCallbackFunction& func);

private:
    std::unique_ptr<AsyncNetworkImpl> pAsyncNetworkImpl;
};

} // end experimental namespace

} // end armnn namespace
