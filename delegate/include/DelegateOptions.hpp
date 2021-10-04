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

    void SetDynamicBackendsPath(const std::string& dynamicBackendsPath) { m_DynamicBackendsPath = dynamicBackendsPath; }
    const std::string& GetDynamicBackendsPath() const { return m_DynamicBackendsPath; }

    void SetGpuProfilingState(bool gpuProfilingState) { m_EnableGpuProfiling = gpuProfilingState; }
    bool GetGpuProfilingState() { return m_EnableGpuProfiling; }

    const std::vector<armnn::BackendOptions>& GetBackendOptions() const { return m_BackendOptions; }

    /// Appends a backend option to the list of backend options
    void AddBackendOption(const armnn::BackendOptions& option) { m_BackendOptions.push_back(option); }

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

private:
    /// Which backend to run Delegate on.
    /// Examples of possible values are: CpuRef, CpuAcc, GpuAcc.
    /// CpuRef as default.
    std::vector<armnn::BackendId> m_Backends = { armnn::Compute::CpuRef };

    /// Pass backend specific options to Delegate
    ///
    /// For example, tuning can be enabled on GpuAcc like below
    /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// m_BackendOptions.emplace_back(
    ///     BackendOptions{"GpuAcc",
    ///       {
    ///         {"TuningLevel", 2},
    ///         {"TuningFile", filename}
    ///       }
    ///     });
    /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// The following backend options are available:
    /// GpuAcc:
    ///   "TuningLevel" : int [0..3] (0=UseOnly(default) | 1=RapidTuning | 2=NormalTuning | 3=ExhaustiveTuning)
    ///   "TuningFile" : string [filenameString]
    ///   "KernelProfilingEnabled" : bool [true | false]
    std::vector<armnn::BackendOptions> m_BackendOptions;

    /// Dynamic backend path.
    /// This is the directory that will be searched for any dynamic backends.
    std::string m_DynamicBackendsPath = "";

    /// Enable Gpu Profiling.
    bool m_EnableGpuProfiling = false;

    /// OptimizerOptions
    /// Reduce Fp32 data to Fp16 for faster processing
    /// bool m_ReduceFp32ToFp16;
    /// Add debug data for easier troubleshooting
    /// bool m_Debug;
    /// Reduce Fp32 data to Bf16 for faster processing
    /// bool m_ReduceFp32ToBf16;
    /// Enable Import
    /// bool m_ImportEnabled;
    /// Enable Model Options
    /// ModelOptions m_ModelOptions;
    armnn::OptimizerOptions m_OptimizerOptions;

    /// External profiling options.
    /// Indicates whether external profiling is enabled or not.
    /// bool m_EnableProfiling
    /// Indicates whether external timeline profiling is enabled or not.
    /// bool m_TimelineEnabled
    /// Path to a file in which outgoing timeline profiling messages will be stored.
    /// std::string m_OutgoingCaptureFile
    /// Path to a file in which incoming timeline profiling messages will be stored.
    /// std::string m_IncomingCaptureFile
    /// Enable profiling output to file only.
    /// bool m_FileOnly
    /// The duration at which captured profiling messages will be flushed.
    /// uint32_t m_CapturePeriod
    /// The format of the file used for outputting profiling data.
    /// std::string m_FileFormat
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
