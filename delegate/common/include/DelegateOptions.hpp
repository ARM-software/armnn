//
// Copyright © 2020-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>
#include <armnn/Optional.hpp>

#include <string>
#include <vector>

namespace armnnDelegate
{

struct DelegateOptionsImpl;

class DelegateOptions
{
public:
    ~DelegateOptions();
    DelegateOptions();
    DelegateOptions(const DelegateOptions& other);

    DelegateOptions(armnn::Compute computeDevice,
                    const std::vector<armnn::BackendOptions>& backendOptions = {},
                    armnn::Optional<armnn::LogSeverity> logSeverityLevel = armnn::EmptyOptional());

    DelegateOptions(const std::vector<armnn::BackendId>& backends,
                    const std::vector<armnn::BackendOptions>& backendOptions = {},
                    armnn::Optional<armnn::LogSeverity> logSeverityLevel = armnn::EmptyOptional());

    DelegateOptions(armnn::Compute computeDevice,
                    const armnn::OptimizerOptionsOpaque& optimizerOptions,
                    const armnn::Optional<armnn::LogSeverity>& logSeverityLevel = armnn::EmptyOptional(),
                    const armnn::Optional<armnn::DebugCallbackFunction>& func = armnn::EmptyOptional());

    DelegateOptions(const std::vector<armnn::BackendId>& backends,
                    const armnn::OptimizerOptionsOpaque& optimizerOptions,
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
     *    Description: This option is currently ignored. Please enable Fast Math in the CpuAcc or GpuAcc backends.
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
     *    Option key: "infer-output-shape" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Infers output tensor shape from input tensor shape and validate where applicable.
     *
     *    Option key: "allow-expanded-dims" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: If true will disregard dimensions with a size of 1 when validating tensor shapes but tensor
     *                 sizes must still match. \n
     *                 This is an Experimental parameter that is incompatible with "infer-output-shape". \n
     *                 This parameter may be removed in a later update.
     *
     *    Option key: "disable-tflite-runtime-fallback" \n
     *    Possible values: ["true"/"false"] \n
     *    Description: Disable TfLite Runtime fallback in the Arm NN TfLite delegate.
     *                 An exception will be thrown if unsupported operators are encountered.
     *                 This option is only for testing purposes.
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

    const std::vector<armnn::BackendId>& GetBackends() const;

    void SetBackends(const std::vector<armnn::BackendId>& backends);

    void SetDynamicBackendsPath(const std::string& dynamicBackendsPath);

    const std::string& GetDynamicBackendsPath() const;

    void SetGpuProfilingState(bool gpuProfilingState);

    bool GetGpuProfilingState();

    const std::vector<armnn::BackendOptions>& GetBackendOptions() const;

    /// Appends a backend option to the list of backend options
    void AddBackendOption(const armnn::BackendOptions& option);

    /// Sets the severity level for logging within ArmNN that will be used on creation of the delegate
    void SetLoggingSeverity(const armnn::LogSeverity& level);
    void SetLoggingSeverity(const std::string& level);

    /// Returns the severity level for logging within ArmNN
    armnn::LogSeverity GetLoggingSeverity();

    bool IsLoggingEnabled();

    const armnn::OptimizerOptionsOpaque& GetOptimizerOptions() const;

    void SetOptimizerOptions(const armnn::OptimizerOptionsOpaque& optimizerOptions);

    const armnn::Optional<armnn::DebugCallbackFunction>& GetDebugCallbackFunction() const;

    void SetInternalProfilingParams(bool internalProfilingState,
                                    const armnn::ProfilingDetailsMethod& internalProfilingDetail);

    bool GetInternalProfilingState() const;

    const armnn::ProfilingDetailsMethod& GetInternalProfilingDetail() const;

    void SetSerializeToDot(const std::string& serializeToDotFile);

    const std::string& GetSerializeToDot() const;

    /// @Note: This might overwrite options that were set with other setter functions of DelegateOptions
    void SetRuntimeOptions(const armnn::IRuntime::CreationOptions& runtimeOptions);

    const armnn::IRuntime::CreationOptions& GetRuntimeOptions();

    void DisableTfLiteRuntimeFallback(bool fallbackState);

    bool TfLiteRuntimeFallbackDisabled();

private:
    std::unique_ptr<armnnDelegate::DelegateOptionsImpl> p_DelegateOptionsImpl;

};

} // namespace armnnDelegate
