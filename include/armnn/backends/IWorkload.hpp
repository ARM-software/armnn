//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <armnn/backends/WorkloadInfo.hpp>

namespace armnn
{

namespace experimental
{

struct ExecutionData;

} // end experimental namespace

using namespace armnn::experimental;

/// Workload interface to enqueue a layer computation.
class IWorkload {
public:
    virtual ~IWorkload() {}

    // Note: do not call for async networks via ExecuteAsync or otherwise,
    // as async networks memory is allocated outside the workload.
    virtual void PostAllocationConfigure() = 0;

    virtual void Execute() const = 0;

    virtual void ExecuteAsync(ExecutionData& executionData) = 0;

    virtual arm::pipe::ProfilingGuid GetGuid() const = 0;

    // SupportsTensorHandleReplacement signals that a given workload is capable of
    // replacing any of its I/O tensors via ReplaceInput/OutputTensorHandle
    virtual bool SupportsTensorHandleReplacement() const = 0;

    // Replace input tensor handle with the given TensorHandle
    virtual void ReplaceInputTensorHandle(ITensorHandle* /*input*/, unsigned int /*slot*/) = 0;

    // Replace output tensor handle with the given TensorHandle
    virtual void ReplaceOutputTensorHandle(ITensorHandle* /*output*/, unsigned int /*slot*/) = 0;

    virtual void RegisterDebugCallback(const DebugCallbackFunction& /*func*/) {}

    virtual armnn::Optional<armnn::MemoryRequirements> GetMemoryRequirements()
    {
        return armnn::EmptyOptional();
    }
};

} //namespace armnn
