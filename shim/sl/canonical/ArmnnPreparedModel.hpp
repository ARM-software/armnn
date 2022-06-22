//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ArmnnDriver.hpp"
#include "ArmnnDriverImpl.hpp"
#include "ModelToINetworkTransformer.hpp"

#include <armnn/ArmNN.hpp>

#include <BufferTracker.h>
#include <CpuExecutor.h>
#include <nnapi/IExecution.h>
#include <nnapi/IPreparedModel.h>
#include <nnapi/Result.h>
#include <nnapi/Types.h>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <string>

namespace armnn_driver
{
    struct CanonicalExecutionContext
    {
        ::android::nn::MeasureTiming    measureTimings =
                ::android::nn::MeasureTiming::NO;
        android::nn::TimePoint driverStart;
        android::nn::TimePoint driverEnd;
        android::nn::TimePoint deviceStart;
        android::nn::TimePoint deviceEnd;
    };
class ArmnnPreparedModel final : public IPreparedModel,
                                 public std::enable_shared_from_this<ArmnnPreparedModel>
{
public:
    ArmnnPreparedModel(armnn::NetworkId networkId,
                       armnn::IRuntime* runtime,
                       const Model& model,
                       const std::string& requestInputsAndOutputsDumpDir,
                       const bool gpuProfilingEnabled,
                       Priority priority = Priority::MEDIUM);

    ArmnnPreparedModel(armnn::NetworkId networkId,
                       armnn::IRuntime* runtime,
                       const std::string& requestInputsAndOutputsDumpDir,
                       const bool gpuProfilingEnabled,
                       Priority priority = Priority::MEDIUM,
                       const bool prepareModelFromCache = false);

    virtual ~ArmnnPreparedModel();

    ExecutionResult<std::pair<std::vector<OutputShape>, Timing>> execute(
        const Request& request,
        MeasureTiming measureTiming,
        const OptionalTimePoint& deadline,
        const OptionalDuration& loopTimeoutDuration,
        const std::vector<android::nn::TokenValuePair>& hints,
        const std::vector<android::nn::ExtensionNameAndPrefix>& extensionNameToPrefix) const override;

    GeneralResult<std::pair<SyncFence, ExecuteFencedInfoCallback>> executeFenced(
        const Request& request,
        const std::vector<SyncFence>& waitFor,
        MeasureTiming measureTiming,
        const OptionalTimePoint& deadline,
        const OptionalDuration& loopTimeoutDuration,
        const OptionalDuration& timeoutDurationAfterFence,
        const std::vector<android::nn::TokenValuePair>& hints,
        const std::vector<android::nn::ExtensionNameAndPrefix>& extensionNameToPrefix) const override;

    GeneralResult<android::nn::SharedExecution> createReusableExecution(
        const Request& request,
        MeasureTiming measureTiming,
        const OptionalDuration& loopTimeoutDuration,
        const std::vector<android::nn::TokenValuePair>& hints,
        const std::vector<android::nn::ExtensionNameAndPrefix>& extensionNameToPrefix) const override;

    GeneralResult<SharedBurst> configureExecutionBurst() const override;

    std::any getUnderlyingResource() const override;

    /// execute the graph prepared from the request
    ErrorStatus ExecuteGraph(
        std::shared_ptr<std::vector<android::nn::RunTimePoolInfo>>& pMemPools,
        armnn::InputTensors& inputTensors,
        armnn::OutputTensors& outputTensors,
        CanonicalExecutionContext  callback,
        const bool pointerMemory = false) const;

    Priority GetModelPriority() const;

    /// Executes this model with dummy inputs (e.g. all zeroes).
    /// \return false on failure, otherwise true
    bool ExecuteWithDummyInputs(unsigned int numInputs, unsigned int numOutputs) const;

private:
    void Init();
    ErrorStatus PrepareMemoryForInputs(
        armnn::InputTensors& inputs,
        const Request& request,
        const std::vector<android::nn::RunTimePoolInfo>& memPools) const;

    ErrorStatus PrepareMemoryForOutputs(
        armnn::OutputTensors& outputs,
        std::vector<OutputShape> &outputShapes,
        const Request& request,
        const std::vector<android::nn::RunTimePoolInfo>& memPools) const;

    ErrorStatus PrepareMemoryForIO(armnn::InputTensors& inputs,
                                   armnn::OutputTensors& outputs,
                                   std::vector<android::nn::RunTimePoolInfo>& memPools,
                                   const Request& request,
                                   const bool pointerMemory = false) const;

    template <typename TensorBindingCollection>
    void DumpTensorsIfRequired(char const* tensorNamePrefix, const TensorBindingCollection& tensorBindings) const;

    /// schedule the graph prepared from the request for execution
    armnn::NetworkId                        m_NetworkId;
    armnn::IRuntime*                        m_Runtime;

    const Model                             m_Model;
    const std::string&                      m_RequestInputsAndOutputsDumpDir;
    const bool                              m_GpuProfilingEnabled;
    Priority                                m_ModelPriority;
    const bool                              m_PrepareFromCache;
};

}
