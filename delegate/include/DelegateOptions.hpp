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


    /**
     * This constructor processes delegate options in form of command line arguments.
     * It works in conjunction with the TfLite external delegate plugin.
     *
     * Available options:
     *
     *    Option key: "backends" \n
     *    Possible values: ["EthosNPU"/"GpuAcc"/"CpuAcc"/"CpuRef"] \n
     *    Descriptions: A comma separated list without whitespaces of
     *                  backends which should be used for execution. Falls
     *                  back to next backend in list if previous doesn't
     *                  provide support for operation. e.g. "GpuAcc,CpuAcc"
     *
     *    Option key: "dynamic-backends-path" \n
     *    Possible values: [filenameString] \n
     *    Descriptions: This is the directory that will be searched for any dynamic backends.
     *
     *    Option key: "logging-severity" \n
     *    Possible values: ["trace"/"debug"/"info"/"warning"/"error"/"fatal"] \n
     *    Description: Sets the logging severity level for ArmNN. Logging
     *                 is turned off if this option is not provided.
     *
     *    Option key: "gpu-tuning-level" \n
     *    Possible values: ["0"/"1"/"2"/"3"] \n
     *    Description: 0=UseOnly(default), 1=RapidTuning, 2=NormalTuning,
     *                 3=ExhaustiveTuning. Requires option gpu-tuning-file.
     *                 1,2 and 3 will create a tuning-file, 0 will apply the
     *                 tunings from an existing file
     *
     *    Option key: "gpu-mlgo-tuning-file" \n
     *    Possible values: [filenameString] \n
     *    Description: File name for the MLGO tuning file
     *
     *    Option key: "gpu-tuning-file" \n
     *    Possible values: [filenameString] \n
     *    Description: File name for the tuning file.
     *
     *    Option key: "gpu-enable-profiling" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Enables GPU profiling
     *
     *    Option key: "gpu-kernel-profiling-enabled" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Enables GPU kernel profiling
     *
     *    Option key: "save-cached-network" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Enables saving of the cached network to a file,
     *                 specified with the cached-network-filepath option
     *
     *    Option key: "cached-network-filepath" \n
     *    Possible values: [filenameString] \n
     *    Description: If non-empty, the given file will be used to load/save the cached network.
     *                 If save-cached-network is given then the cached network will be saved to the given file.
     *                 To save the cached network a file must already exist.
     *                 If save-cached-network is not given then the cached network will be loaded from the given file.
     *                 This will remove initial compilation time of kernels and speed up the first execution.
     *
     *    Option key: "enable-fast-math" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Enables fast_math options in backends that support it
     *
     *    Option key: "number-of-threads" \n
     *    Possible values: ["1"-"64"] \n
     *    Description: Assign the number of threads used by the CpuAcc backend.
     *                 Default is set to 0 (Backend will decide number of threads to use).
     *
     *    Option key: "reduce-fp32-to-fp16" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Reduce Fp32 data to Fp16 for faster processing
     *
     *    Option key: "reduce-fp32-to-bf16" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Reduce Fp32 data to Bf16 for faster processing
     *
     *    Option key: "debug-data" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Add debug data for easier troubleshooting
     *
     *    Option key: "memory-import" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Enable memory import
     *
     *    Option key: "enable-internal-profiling" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Enable the internal profiling feature.
     *
     *    Option key: "internal-profiling-detail" \n
     *    Possible values: [1/2] \n
     *    Description: Set the detail on the internal profiling. 1 = DetailsWithEvents, 2 = DetailsOnly.
     *
     *    Option key: "enable-external-profiling" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Enable the external profiling feature.
     *
     *    Option key: "timeline-profiling" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Indicates whether external timeline profiling is enabled or not.
     *
     *    Option key: "outgoing-capture-file" \n
     *    Possible values: [filenameString] \n
     *    Description: Path to a file in which outgoing timeline profiling messages will be stored.
     *
     *    Option key: "incoming-capture-file" \n
     *    Possible values: [filenameString] \n
     *    Description: Path to a file in which incoming timeline profiling messages will be stored.
     *
     *    Option key: "file-only-external-profiling" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Enable profiling output to file only.
     *
     *    Option key: "counter-capture-period" \n
     *    Possible values: Integer, Default is 10000u
     *    Description: Value in microseconds of the profiling capture period. \n
     *
     *    Option key: "profiling-file-format" \n
     *    Possible values: String of ["binary"] \n
     *    Description: The format of the file used for outputting profiling data. Currently on "binary" is supported.
     *
     *    Option key: "serialize-to-dot" \n
     *    Possible values: [filenameString] \n
     *    Description: Serialize the optimized network to the file specified in "dot" format.
     *
     * @param[in]     option_keys     Delegate option names
     * @param[in]     options_values  Delegate option values
     * @param[in]     num_options     Number of delegate options
     * @param[in,out] report_error    Error callback function
     *
     */
    DelegateOptions(char const* const* options_keys,
                    char const* const* options_values,
                    size_t num_options,
                    void (*report_error)(const char*));

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
