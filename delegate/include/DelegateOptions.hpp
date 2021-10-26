//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>
#include <armnn/Logging.hpp>
#include <armnn/Optional.hpp>

#include <set>
#include <string>
#include <vector>

namespace armnnDelegate
{

class DelegateOptions
{
public:
    DelegateOptions(armnn::Compute computeDevice,
                    const std::vector<armnn::BackendOptions>& backendOptions = {},
                    armnn::Optional<armnn::LogSeverity> logSeverityLevel = armnn::EmptyOptional());

    DelegateOptions(const std::vector<armnn::BackendId>& backends,
                    const std::vector<armnn::BackendOptions>& backendOptions = {},
                    armnn::Optional<armnn::LogSeverity> logSeverityLevel = armnn::EmptyOptional());

    DelegateOptions(armnn::Compute computeDevice,
                    const armnn::OptimizerOptions& optimizerOptions,
                    const armnn::Optional<armnn::LogSeverity>& logSeverityLevel = armnn::EmptyOptional(),
                    const armnn::Optional<armnn::DebugCallbackFunction>& func = armnn::EmptyOptional());

    DelegateOptions(const std::vector<armnn::BackendId>& backends,
                    const armnn::OptimizerOptions& optimizerOptions,
                    const armnn::Optional<armnn::LogSeverity>& logSeverityLevel = armnn::EmptyOptional(),
                    const armnn::Optional<armnn::DebugCallbackFunction>& func = armnn::EmptyOptional());

    const std::vector<armnn::BackendId>& GetBackends() const { return m_Backends; }

    void SetBackends(const std::vector<armnn::BackendId>& backends) { m_Backends = backends; }

    void SetDynamicBackendsPath(const std::string& dynamicBackendsPath)
    {
        m_RuntimeOptions.m_DynamicBackendsPath = dynamicBackendsPath;
    }
    const std::string& GetDynamicBackendsPath() const
    {
        return m_RuntimeOptions.m_DynamicBackendsPath;
    }

    void SetGpuProfilingState(bool gpuProfilingState)
    {
        m_RuntimeOptions.m_EnableGpuProfiling = gpuProfilingState;
    }
    bool GetGpuProfilingState()
    {
        return m_RuntimeOptions.m_EnableGpuProfiling;
    }

    const std::vector<armnn::BackendOptions>& GetBackendOptions() const
    {
        return m_RuntimeOptions.m_BackendOptions;
    }

    /// Appends a backend option to the list of backend options
    void AddBackendOption(const armnn::BackendOptions& option)
    {
        m_RuntimeOptions.m_BackendOptions.push_back(option);
    }

    /// Sets the severity level for logging within ArmNN that will be used on creation of the delegate
    void SetLoggingSeverity(const armnn::LogSeverity& level) { m_LoggingSeverity = level; }
    void SetLoggingSeverity(const std::string& level) { m_LoggingSeverity = armnn::StringToLogLevel(level); }

    /// Returns the severity level for logging within ArmNN
    armnn::LogSeverity GetLoggingSeverity() { return m_LoggingSeverity.value(); }

    bool IsLoggingEnabled() { return m_LoggingSeverity.has_value(); }

    const armnn::OptimizerOptions& GetOptimizerOptions() const { return m_OptimizerOptions; }

    void SetOptimizerOptions(const armnn::OptimizerOptions& optimizerOptions) { m_OptimizerOptions = optimizerOptions; }

    const armnn::Optional<armnn::DebugCallbackFunction>& GetDebugCallbackFunction() const
        { return m_DebugCallbackFunc; }

    void SetInternalProfilingParams(bool internalProfilingState,
                                    const armnn::ProfilingDetailsMethod& internalProfilingDetail)
        { m_InternalProfilingEnabled = internalProfilingState; m_InternalProfilingDetail = internalProfilingDetail; }

    bool GetInternalProfilingState() const { return m_InternalProfilingEnabled; }
    const armnn::ProfilingDetailsMethod& GetInternalProfilingDetail() const { return m_InternalProfilingDetail; }

    void SetExternalProfilingParams(
        const armnn::IRuntime::CreationOptions::ExternalProfilingOptions& externalProfilingParams)
        { m_ProfilingOptions = externalProfilingParams; }

    const armnn::IRuntime::CreationOptions::ExternalProfilingOptions& GetExternalProfilingParams() const
        { return m_ProfilingOptions; }

    void SetSerializeToDot(const std::string& serializeToDotFile) { m_SerializeToDot = serializeToDotFile; }
    const std::string& GetSerializeToDot() const { return m_SerializeToDot; }

    /// @Note: This might overwrite options that were set with other setter functions of DelegateOptions
    void SetRuntimeOptions(const armnn::IRuntime::CreationOptions& runtimeOptions)
    {
        m_RuntimeOptions = runtimeOptions;
    }

    const armnn::IRuntime::CreationOptions& GetRuntimeOptions()
    {
        return m_RuntimeOptions;
    }

private:
    /// Which backend to run Delegate on.
    /// Examples of possible values are: CpuRef, CpuAcc, GpuAcc.
    /// CpuRef as default.
    std::vector<armnn::BackendId> m_Backends = { armnn::Compute::CpuRef };

    /// Creation options for the ArmNN runtime
    /// Contains options for global settings that are valid for the whole lifetime of ArmNN
    /// i.e. BackendOptions, DynamicBackendPath, ExternalProfilingOptions and more
    armnn::IRuntime::CreationOptions m_RuntimeOptions;

    /// Options for the optimization step for the network
    armnn::OptimizerOptions m_OptimizerOptions;

    /// External profiling options.
    armnn::IRuntime::CreationOptions::ExternalProfilingOptions m_ProfilingOptions;

    /// Internal profiling options.
    /// Indicates whether internal profiling is enabled or not.
    bool m_InternalProfilingEnabled = false;
    /// Sets the level of detail output by the profiling. Options are DetailsWithEvents = 1 and DetailsOnly = 2
    armnn::ProfilingDetailsMethod m_InternalProfilingDetail = armnn::ProfilingDetailsMethod::DetailsWithEvents;

    /// Severity level for logging within ArmNN that will be used on creation of the delegate
    armnn::Optional<armnn::LogSeverity> m_LoggingSeverity;

    /// A callback function to debug layers performing custom computations on intermediate tensors.
    /// If a function is not registered, and debug is enabled in OptimizerOptions,
    /// debug will print information of the intermediate tensors.
    armnn::Optional<armnn::DebugCallbackFunction> m_DebugCallbackFunc;

    /// If not empty then the optimized model will be serialized to a file with this file name in "dot" format.
    std::string m_SerializeToDot = "";
};

} // namespace armnnDelegate
