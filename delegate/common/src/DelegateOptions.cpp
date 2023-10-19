//
// Copyright © 2020-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <DelegateOptions.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/StringUtils.hpp>

namespace armnnDelegate
{

struct DelegateOptionsImpl
{
    ~DelegateOptionsImpl() = default;
    DelegateOptionsImpl() = default;

    explicit DelegateOptionsImpl(armnn::Compute computeDevice,
                                 const std::vector<armnn::BackendOptions>& backendOptions,
                                 const armnn::Optional<armnn::LogSeverity> logSeverityLevel)
            : m_Backends({computeDevice}), m_RuntimeOptions(), m_LoggingSeverity(logSeverityLevel)
    {
        m_RuntimeOptions.m_BackendOptions = backendOptions;
    }

    explicit DelegateOptionsImpl(const std::vector<armnn::BackendId>& backends,
                                 const std::vector<armnn::BackendOptions>& backendOptions,
                                 const armnn::Optional<armnn::LogSeverity> logSeverityLevel)
            : m_Backends(backends), m_RuntimeOptions(), m_LoggingSeverity(logSeverityLevel)
    {
        m_RuntimeOptions.m_BackendOptions = backendOptions;
    }

    explicit DelegateOptionsImpl(armnn::Compute computeDevice,
                                 const armnn::OptimizerOptionsOpaque& optimizerOptions,
                                 const armnn::Optional<armnn::LogSeverity>& logSeverityLevel,
                                 const armnn::Optional<armnn::DebugCallbackFunction>& func)
            : m_Backends({computeDevice}),
              m_RuntimeOptions(),
              m_OptimizerOptions(optimizerOptions),
              m_LoggingSeverity(logSeverityLevel),
              m_DebugCallbackFunc(func)
    {
    }

    explicit DelegateOptionsImpl(const std::vector<armnn::BackendId>& backends,
                                 const armnn::OptimizerOptionsOpaque& optimizerOptions,
                                 const armnn::Optional<armnn::LogSeverity>& logSeverityLevel,
                                 const armnn::Optional<armnn::DebugCallbackFunction>& func)
            : m_Backends(backends),
              m_RuntimeOptions(),
              m_OptimizerOptions(optimizerOptions),
              m_LoggingSeverity(logSeverityLevel),
              m_DebugCallbackFunc(func)
    {
    }

    /// Which backend to run Delegate on.
    /// Examples of possible values are: CpuRef, CpuAcc, GpuAcc.
    /// CpuRef as default.
    std::vector<armnn::BackendId> m_Backends = {armnn::Compute::CpuRef };

    /// Creation options for the ArmNN runtime
    /// Contains options for global settings that are valid for the whole lifetime of ArmNN
    /// i.e. BackendOptions, DynamicBackendPath, ExternalProfilingOptions and more
    armnn::IRuntime::CreationOptions m_RuntimeOptions;

    /// Options for the optimization step for the network
    armnn::OptimizerOptionsOpaque m_OptimizerOptions;

    /// Internal profiling options. Written to INetworkProperties during model load.
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

    /// Option to disable TfLite Runtime fallback for unsupported operators.
    bool m_DisableTfLiteRuntimeFallback = false;

};

DelegateOptions::~DelegateOptions() = default;

DelegateOptions::DelegateOptions()
    : p_DelegateOptionsImpl(std::make_unique<DelegateOptionsImpl>())
{
}

DelegateOptions::DelegateOptions(DelegateOptions const &other)
    : p_DelegateOptionsImpl(std::make_unique<DelegateOptionsImpl>(*other.p_DelegateOptionsImpl))
{
}

DelegateOptions::DelegateOptions(armnn::Compute computeDevice,
                                 const std::vector<armnn::BackendOptions>& backendOptions,
                                 const armnn::Optional<armnn::LogSeverity> logSeverityLevel)
    : p_DelegateOptionsImpl(std::make_unique<DelegateOptionsImpl>(computeDevice, backendOptions, logSeverityLevel))
{
}

DelegateOptions::DelegateOptions(const std::vector<armnn::BackendId>& backends,
                                 const std::vector<armnn::BackendOptions>& backendOptions,
                                 const armnn::Optional<armnn::LogSeverity> logSeverityLevel)
    : p_DelegateOptionsImpl(std::make_unique<DelegateOptionsImpl>(backends, backendOptions, logSeverityLevel))
{
}

DelegateOptions::DelegateOptions(armnn::Compute computeDevice,
                                 const armnn::OptimizerOptionsOpaque& optimizerOptions,
                                 const armnn::Optional<armnn::LogSeverity>& logSeverityLevel,
                                 const armnn::Optional<armnn::DebugCallbackFunction>& func)
    : p_DelegateOptionsImpl(std::make_unique<DelegateOptionsImpl>(computeDevice, optimizerOptions,
                                                                  logSeverityLevel, func))
{
}

DelegateOptions::DelegateOptions(const std::vector<armnn::BackendId>& backends,
                                 const armnn::OptimizerOptionsOpaque& optimizerOptions,
                                 const armnn::Optional<armnn::LogSeverity>& logSeverityLevel,
                                 const armnn::Optional<armnn::DebugCallbackFunction>& func)
    : p_DelegateOptionsImpl(std::make_unique<DelegateOptionsImpl>(backends, optimizerOptions,
                                                                  logSeverityLevel, func))
{
}

DelegateOptions::DelegateOptions(char const* const* options_keys,
                                 char const* const* options_values,
                                 size_t num_options,
                                 void (*report_error)(const char*))
    : p_DelegateOptionsImpl(std::make_unique<DelegateOptionsImpl>())
{
    armnn::IRuntime::CreationOptions runtimeOptions;
    armnn::OptimizerOptionsOpaque optimizerOptions;
    bool internalProfilingState = false;
    armnn::ProfilingDetailsMethod internalProfilingDetail = armnn::ProfilingDetailsMethod::DetailsWithEvents;

    // Process backends
    bool GpuAccFound = false;
    bool CpuAccFound = false;
    for (size_t i = 0; i < num_options; ++i)
    {
        if (std::string(options_keys[i]) == std::string("backends"))
        {
            // The backend option is a comma separated string of backendIDs that needs to be split
            std::vector<armnn::BackendId> backends;
            char *dup = strdup(options_values[i]);
            char *pch = std::strtok(dup, ",");
            while (pch != NULL)
            {
                backends.push_back(pch);
                pch = strtok(NULL, ",");
            }
            SetBackends(backends);
            GpuAccFound = std::count(GetBackends().begin(), GetBackends().end(), "GpuAcc");
            CpuAccFound = std::count(GetBackends().begin(), GetBackends().end(), "CpuAcc");
            break;
        }
    }

    // Rest of options after knowing the backend
    for (size_t i = 0; i < num_options; ++i)
    {
        if (std::string(options_keys[i]) == std::string("backends"))
        {
            continue;
        }
        // Process dynamic-backends-path
        else if (std::string(options_keys[i]) == std::string("dynamic-backends-path"))
        {
            runtimeOptions.m_DynamicBackendsPath = std::string(options_values[i]);
        }
        // Process logging level
        else if (std::string(options_keys[i]) == std::string("logging-severity"))
        {
            SetLoggingSeverity(options_values[i]);
        }
        // Process GPU backend options
        else if (std::string(options_keys[i]) == std::string("gpu-tuning-level"))
        {
            if (GpuAccFound)
            {
                armnn::BackendOptions option("GpuAcc", {{"TuningLevel",
                                                         atoi(options_values[i])}});
                runtimeOptions.m_BackendOptions.push_back(option);
            }
            else
            {
                ARMNN_LOG(warning) <<
                "WARNING: TuningLevel is enabled, but no backends that accept this option are set.";
            }
        }
        else if (std::string(options_keys[i]) == std::string("gpu-mlgo-tuning-file"))
        {
            if (GpuAccFound)
            {
                armnn::BackendOptions option("GpuAcc", {{"MLGOTuningFilePath",
                                                         std::string(options_values[i])}});
                optimizerOptions.AddModelOption(option);
            }
            else
            {
                ARMNN_LOG(warning) <<
                "WARNING: MLGOTuningFilePath is enabled, but no backends that accept this option are set.";
            }
        }
        else if (std::string(options_keys[i]) == std::string("gpu-tuning-file"))
        {
            if (GpuAccFound)
            {
                armnn::BackendOptions option("GpuAcc", {{"TuningFile",
                                                         std::string(options_values[i])}});
                runtimeOptions.m_BackendOptions.push_back(option);
            }
            else
            {
                ARMNN_LOG(warning) <<
                "WARNING: TuningFile is enabled, but no backends that accept this option are set.";
            }
        }
        else if (std::string(options_keys[i]) == std::string("gpu-enable-profiling"))
        {
            runtimeOptions.m_EnableGpuProfiling = (*options_values[i] != '0');
        }
        else if (std::string(options_keys[i]) == std::string("gpu-kernel-profiling-enabled"))
        {
            if (GpuAccFound)
            {
                armnn::BackendOptions option("GpuAcc", {{"KernelProfilingEnabled",
                                                         armnn::stringUtils::StringToBool(options_values[i])}});
                runtimeOptions.m_BackendOptions.push_back(option);
            }
            else
            {
                ARMNN_LOG(warning) <<
                "WARNING: KernelProfilingEnabled is enabled, but no backends that accept this option are set.";
            }
        }
        else if (std::string(options_keys[i]) == std::string("save-cached-network"))
        {
            if (GpuAccFound)
            {
                armnn::BackendOptions option("GpuAcc", {{"SaveCachedNetwork",
                                                         armnn::stringUtils::StringToBool(options_values[i])}});
                optimizerOptions.AddModelOption(option);
            }
            else
            {
                ARMNN_LOG(warning) <<
                "WARNING: SaveCachedNetwork is enabled, but no backends that accept this option are set.";
            }
        }
        else if (std::string(options_keys[i]) == std::string("cached-network-filepath"))
        {
            if (GpuAccFound)
            {
                armnn::BackendOptions option("GpuAcc", {{"CachedNetworkFilePath",
                                                         std::string(options_values[i])}});
                optimizerOptions.AddModelOption(option);
            }
            else
            {
                ARMNN_LOG(warning) <<
                "WARNING: CachedNetworkFilePath is enabled, but no backends that accept this option are set.";
            }
        }
        // Process GPU & CPU backend options
        else if (std::string(options_keys[i]) == std::string("enable-fast-math"))
        {
            if (GpuAccFound)
            {
                armnn::BackendOptions modelOptionGpu("GpuAcc", {{"FastMathEnabled",
                                                                 armnn::stringUtils::StringToBool(options_values[i])}});
                optimizerOptions.AddModelOption(modelOptionGpu);
            }
            if (CpuAccFound)
            {
                armnn::BackendOptions modelOptionCpu("CpuAcc", {{"FastMathEnabled",
                                                                 armnn::stringUtils::StringToBool(options_values[i])}});
                optimizerOptions.AddModelOption(modelOptionCpu);
            }
            if (!GpuAccFound and !CpuAccFound)
            {
                ARMNN_LOG(warning) <<
                "WARNING: Fastmath is enabled, but no backends that accept this option are set.";
            }
        }
        // Process CPU backend options
        else if (std::string(options_keys[i]) == std::string("number-of-threads"))
        {
            if (CpuAccFound)
            {
                unsigned int numberOfThreads = armnn::numeric_cast<unsigned int>(atoi(options_values[i]));
                armnn::BackendOptions modelOption("CpuAcc",
                                                  {{"NumberOfThreads", numberOfThreads}});
                optimizerOptions.AddModelOption(modelOption);
            }
            else
            {
                ARMNN_LOG(warning) <<
                "WARNING: NumberOfThreads is enabled, but no backends that accept this option are set.";
            }
        }
        // Process reduce-fp32-to-fp16 option
        else if (std::string(options_keys[i]) == std::string("reduce-fp32-to-fp16"))
        {
            optimizerOptions.SetReduceFp32ToFp16(armnn::stringUtils::StringToBool(options_values[i]));
        }
        // Process debug-data
        else if (std::string(options_keys[i]) == std::string("debug-data"))
        {
            optimizerOptions.SetDebugEnabled(armnn::stringUtils::StringToBool(options_values[i]));
        }
        // Infer output-shape
        else if (std::string(options_keys[i]) == std::string("infer-output-shape"))
        {
            if (armnn::stringUtils::StringToBool(options_values[i]))
            {
                optimizerOptions.SetShapeInferenceMethod(armnn::ShapeInferenceMethod::InferAndValidate);
            }
            else
            {
                optimizerOptions.SetShapeInferenceMethod(armnn::ShapeInferenceMethod::ValidateOnly);
            }
        }
        // Allow expanded dims
        else if (std::string(options_keys[i]) == std::string("allow-expanded-dims"))
        {
            optimizerOptions.SetAllowExpandedDims(armnn::stringUtils::StringToBool(options_values[i]));
        }
        // Process memory-import
        else if (std::string(options_keys[i]) == std::string("memory-import"))
        {
            optimizerOptions.SetImportEnabled(armnn::stringUtils::StringToBool(options_values[i]));
        }
        // Process enable-internal-profiling
        else if (std::string(options_keys[i]) == std::string("enable-internal-profiling"))
        {
            internalProfilingState = *options_values[i] != '0';
            optimizerOptions.SetProfilingEnabled(internalProfilingState);
        }
        // Process internal-profiling-detail
        else if (std::string(options_keys[i]) == std::string("internal-profiling-detail"))
        {
            uint32_t detailLevel = static_cast<uint32_t>(std::stoul(options_values[i]));
            switch (detailLevel)
            {
                case 1:
                    internalProfilingDetail = armnn::ProfilingDetailsMethod::DetailsWithEvents;
                    break;
                case 2:
                    internalProfilingDetail = armnn::ProfilingDetailsMethod::DetailsOnly;
                    break;
                default:
                    internalProfilingDetail = armnn::ProfilingDetailsMethod::Undefined;
                    break;
            }
        }
        // Process enable-external-profiling
        else if (std::string(options_keys[i]) == std::string("enable-external-profiling"))
        {
            runtimeOptions.m_ProfilingOptions.m_EnableProfiling = armnn::stringUtils::StringToBool(options_values[i]);
        }
        // Process timeline-profiling
        else if (std::string(options_keys[i]) == std::string("timeline-profiling"))
        {
            runtimeOptions.m_ProfilingOptions.m_TimelineEnabled = armnn::stringUtils::StringToBool(options_values[i]);
        }
        // Process outgoing-capture-file
        else if (std::string(options_keys[i]) == std::string("outgoing-capture-file"))
        {
            runtimeOptions.m_ProfilingOptions.m_OutgoingCaptureFile = options_values[i];
        }
        // Process incoming-capture-file
        else if (std::string(options_keys[i]) == std::string("incoming-capture-file"))
        {
            runtimeOptions.m_ProfilingOptions.m_IncomingCaptureFile = options_values[i];
        }
        // Process file-only-external-profiling
        else if (std::string(options_keys[i]) == std::string("file-only-external-profiling"))
        {
            runtimeOptions.m_ProfilingOptions.m_FileOnly = armnn::stringUtils::StringToBool(options_values[i]);
        }
        // Process counter-capture-period
        else if (std::string(options_keys[i]) == std::string("counter-capture-period"))
        {
            runtimeOptions.m_ProfilingOptions.m_CapturePeriod = static_cast<uint32_t>(std::stoul(options_values[i]));
        }
        // Process profiling-file-format
        else if (std::string(options_keys[i]) == std::string("profiling-file-format"))
        {
            runtimeOptions.m_ProfilingOptions.m_FileFormat = options_values[i];
        }
        // Process serialize-to-dot
        else if (std::string(options_keys[i]) == std::string("serialize-to-dot"))
        {
            SetSerializeToDot(options_values[i]);
        }
        // Process disable-tflite-runtime-fallback
        else if (std::string(options_keys[i]) == std::string("disable-tflite-runtime-fallback"))
        {
            this->DisableTfLiteRuntimeFallback(armnn::stringUtils::StringToBool(options_values[i]));
        }
        else
        {
            throw armnn::Exception("Unknown option for the ArmNN Delegate given: " +  std::string(options_keys[i]));
        }
    }

    SetRuntimeOptions(runtimeOptions);
    SetOptimizerOptions(optimizerOptions);
    SetInternalProfilingParams(internalProfilingState, internalProfilingDetail);
}

const std::vector<armnn::BackendId>& DelegateOptions::GetBackends() const
{
    return p_DelegateOptionsImpl->m_Backends;
}

void DelegateOptions::SetBackends(const std::vector<armnn::BackendId>& backends)
{
    p_DelegateOptionsImpl->m_Backends = backends;
}

void DelegateOptions::SetDynamicBackendsPath(const std::string& dynamicBackendsPath)
{
    p_DelegateOptionsImpl->m_RuntimeOptions.m_DynamicBackendsPath = dynamicBackendsPath;
}

const std::string& DelegateOptions::GetDynamicBackendsPath() const
{
    return p_DelegateOptionsImpl->m_RuntimeOptions.m_DynamicBackendsPath;
}

void DelegateOptions::SetGpuProfilingState(bool gpuProfilingState)
{
    p_DelegateOptionsImpl->m_RuntimeOptions.m_EnableGpuProfiling = gpuProfilingState;
}

bool DelegateOptions::GetGpuProfilingState()
{
    return p_DelegateOptionsImpl->m_RuntimeOptions.m_EnableGpuProfiling;
}

const std::vector<armnn::BackendOptions>& DelegateOptions::GetBackendOptions() const
{
    return p_DelegateOptionsImpl->m_RuntimeOptions.m_BackendOptions;
}

void DelegateOptions::AddBackendOption(const armnn::BackendOptions& option)
{
    p_DelegateOptionsImpl->m_RuntimeOptions.m_BackendOptions.push_back(option);
}

void DelegateOptions::SetLoggingSeverity(const armnn::LogSeverity& level)
{
    p_DelegateOptionsImpl->m_LoggingSeverity = level;
}

void DelegateOptions::SetLoggingSeverity(const std::string& level)
{
    p_DelegateOptionsImpl->m_LoggingSeverity = armnn::StringToLogLevel(level);
}

armnn::LogSeverity DelegateOptions::GetLoggingSeverity()
{
    return p_DelegateOptionsImpl->m_LoggingSeverity.value();
}

bool DelegateOptions::IsLoggingEnabled()
{
    return p_DelegateOptionsImpl->m_LoggingSeverity.has_value();
}

const armnn::OptimizerOptionsOpaque& DelegateOptions::GetOptimizerOptions() const
{
    return p_DelegateOptionsImpl->m_OptimizerOptions;
}

void DelegateOptions::SetOptimizerOptions(const armnn::OptimizerOptionsOpaque& optimizerOptions)
{
    p_DelegateOptionsImpl->m_OptimizerOptions = optimizerOptions;
}

const armnn::Optional<armnn::DebugCallbackFunction>& DelegateOptions::GetDebugCallbackFunction() const
{
    return p_DelegateOptionsImpl->m_DebugCallbackFunc;
}

void DelegateOptions::SetInternalProfilingParams(bool internalProfilingState,
                                const armnn::ProfilingDetailsMethod& internalProfilingDetail)
{
    p_DelegateOptionsImpl->m_InternalProfilingEnabled = internalProfilingState;
    p_DelegateOptionsImpl->m_InternalProfilingDetail = internalProfilingDetail;
}

bool DelegateOptions::GetInternalProfilingState() const
{
    return p_DelegateOptionsImpl->m_InternalProfilingEnabled;
}

const armnn::ProfilingDetailsMethod& DelegateOptions::GetInternalProfilingDetail() const
{
    return p_DelegateOptionsImpl->m_InternalProfilingDetail;
}

void DelegateOptions::SetSerializeToDot(const std::string& serializeToDotFile)
{
    p_DelegateOptionsImpl->m_SerializeToDot = serializeToDotFile;
}

const std::string& DelegateOptions::GetSerializeToDot() const
{
    return p_DelegateOptionsImpl->m_SerializeToDot;
}

void DelegateOptions::SetRuntimeOptions(const armnn::IRuntime::CreationOptions& runtimeOptions)
{
    p_DelegateOptionsImpl->m_RuntimeOptions = runtimeOptions;
}

const armnn::IRuntime::CreationOptions& DelegateOptions::GetRuntimeOptions()
{
    return p_DelegateOptionsImpl->m_RuntimeOptions;
}

void DelegateOptions::DisableTfLiteRuntimeFallback(bool fallbackState)
{
    p_DelegateOptionsImpl->m_DisableTfLiteRuntimeFallback = fallbackState;
}

bool DelegateOptions::TfLiteRuntimeFallbackDisabled()
{
    return p_DelegateOptionsImpl->m_DisableTfLiteRuntimeFallback;
}

} // namespace armnnDelegate
