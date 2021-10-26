//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <DelegateOptions.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/StringUtils.hpp>

namespace armnnDelegate
{

DelegateOptions::DelegateOptions(armnn::Compute computeDevice,
                                 const std::vector<armnn::BackendOptions>& backendOptions,
                                 const armnn::Optional<armnn::LogSeverity> logSeverityLevel)
    : m_Backends({computeDevice}), m_RuntimeOptions(), m_LoggingSeverity(logSeverityLevel)
{
    m_RuntimeOptions.m_BackendOptions = backendOptions;
}

DelegateOptions::DelegateOptions(const std::vector<armnn::BackendId>& backends,
                                 const std::vector<armnn::BackendOptions>& backendOptions,
                                 const armnn::Optional<armnn::LogSeverity> logSeverityLevel)
    : m_Backends(backends), m_RuntimeOptions(), m_LoggingSeverity(logSeverityLevel)
{
    m_RuntimeOptions.m_BackendOptions = backendOptions;
}

DelegateOptions::DelegateOptions(armnn::Compute computeDevice,
                                 const armnn::OptimizerOptions& optimizerOptions,
                                 const armnn::Optional<armnn::LogSeverity>& logSeverityLevel,
                                 const armnn::Optional<armnn::DebugCallbackFunction>& func)
    : m_Backends({computeDevice}),
      m_RuntimeOptions(),
      m_OptimizerOptions(optimizerOptions),
      m_LoggingSeverity(logSeverityLevel),
      m_DebugCallbackFunc(func)
{
}

DelegateOptions::DelegateOptions(const std::vector<armnn::BackendId>& backends,
                                 const armnn::OptimizerOptions& optimizerOptions,
                                 const armnn::Optional<armnn::LogSeverity>& logSeverityLevel,
                                 const armnn::Optional<armnn::DebugCallbackFunction>& func)
    : m_Backends(backends),
      m_RuntimeOptions(),
      m_OptimizerOptions(optimizerOptions),
      m_LoggingSeverity(logSeverityLevel),
      m_DebugCallbackFunc(func)
{
}

DelegateOptions::DelegateOptions(char const* const* options_keys,
                                 char const* const* options_values,
                                 size_t num_options,
                                 void (*report_error)(const char*))
{
    armnn::IRuntime::CreationOptions runtimeOptions;
    armnn::OptimizerOptions optimizerOptions;
    bool internalProfilingState = false;
    armnn::ProfilingDetailsMethod internalProfilingDetail = armnn::ProfilingDetailsMethod::DetailsWithEvents;
    armnn::IRuntime::CreationOptions::ExternalProfilingOptions extProfilingParams;
    for (size_t i = 0; i < num_options; ++i)
    {
        // Process backends
        if (std::string(options_keys[i]) == std::string("backends"))
        {
            // The backend option is a comma separated string of backendIDs that needs to be split
            std::vector<armnn::BackendId> backends;
            char* dup = strdup(options_values[i]);
            char* pch = std::strtok(dup, ",");
            while (pch != NULL)
            {
                backends.push_back(pch);
                pch = strtok (NULL, ",");
            }
            this->SetBackends(backends);
        }
            // Process dynamic-backends-path
        else if (std::string(options_keys[i]) == std::string("dynamic-backends-path"))
        {
            runtimeOptions.m_DynamicBackendsPath = std::string(options_values[i]);
        }
            // Process logging level
        else if (std::string(options_keys[i]) == std::string("logging-severity"))
        {
            this->SetLoggingSeverity(options_values[i]);
        }
            // Process GPU backend options
        else if (std::string(options_keys[i]) == std::string("gpu-tuning-level"))
        {
            armnn::BackendOptions option("GpuAcc", {{"TuningLevel", atoi(options_values[i])}});
            runtimeOptions.m_BackendOptions.push_back(option);
        }
        else if (std::string(options_keys[i]) == std::string("gpu-mlgo-tuning-file"))
        {
            armnn::BackendOptions option("GpuAcc", {{"MLGOTuningFilePath", std::string(options_values[i])}});
            optimizerOptions.m_ModelOptions.push_back(option);
        }
        else if (std::string(options_keys[i]) == std::string("gpu-tuning-file"))
        {
            armnn::BackendOptions option("GpuAcc", {{"TuningFile", std::string(options_values[i])}});
            runtimeOptions.m_BackendOptions.push_back(option);
        }
        else if (std::string(options_keys[i]) == std::string("gpu-enable-profiling"))
        {
            runtimeOptions.m_EnableGpuProfiling = (*options_values[i] != '0');
        }
        else if (std::string(options_keys[i]) == std::string("gpu-kernel-profiling-enabled"))
        {
            armnn::BackendOptions option("GpuAcc", {{"KernelProfilingEnabled",
                                                     armnn::stringUtils::StringToBool(options_values[i])}});
            runtimeOptions.m_BackendOptions.push_back(option);
        }
        else if (std::string(options_keys[i]) == std::string("save-cached-network"))
        {
            armnn::BackendOptions option("GpuAcc", {{"SaveCachedNetwork",
                                                     armnn::stringUtils::StringToBool(options_values[i])}});
            optimizerOptions.m_ModelOptions.push_back(option);
        }
        else if (std::string(options_keys[i]) == std::string("cached-network-filepath"))
        {
            armnn::BackendOptions option("GpuAcc", {{"CachedNetworkFilePath", std::string(options_values[i])}});
            optimizerOptions.m_ModelOptions.push_back(option);
        }
            // Process GPU & CPU backend options
        else if (std::string(options_keys[i]) == std::string("enable-fast-math"))
        {
            armnn::BackendOptions modelOptionGpu("GpuAcc", {{"FastMathEnabled",
                                                             armnn::stringUtils::StringToBool(options_values[i])}});
            optimizerOptions.m_ModelOptions.push_back(modelOptionGpu);

            armnn::BackendOptions modelOptionCpu("CpuAcc", {{"FastMathEnabled",
                                                             armnn::stringUtils::StringToBool(options_values[i])}});
            optimizerOptions.m_ModelOptions.push_back(modelOptionCpu);
        }
            // Process CPU backend options
        else if (std::string(options_keys[i]) == std::string("number-of-threads"))
        {
            unsigned int numberOfThreads = armnn::numeric_cast<unsigned int>(atoi(options_values[i]));
            armnn::BackendOptions modelOption("CpuAcc", {{"NumberOfThreads", numberOfThreads}});
            optimizerOptions.m_ModelOptions.push_back(modelOption);
        }
            // Process reduce-fp32-to-fp16 option
        else if (std::string(options_keys[i]) == std::string("reduce-fp32-to-fp16"))
        {
            optimizerOptions.m_ReduceFp32ToFp16 = armnn::stringUtils::StringToBool(options_values[i]);
        }
            // Process reduce-fp32-to-bf16 option
        else if (std::string(options_keys[i]) == std::string("reduce-fp32-to-bf16"))
        {
            optimizerOptions.m_ReduceFp32ToBf16 = armnn::stringUtils::StringToBool(options_values[i]);
        }
            // Process debug-data
        else if (std::string(options_keys[i]) == std::string("debug-data"))
        {
            optimizerOptions.m_Debug = armnn::stringUtils::StringToBool(options_values[i]);
        }
            // Process memory-import
        else if (std::string(options_keys[i]) == std::string("memory-import"))
        {
            optimizerOptions.m_ImportEnabled = armnn::stringUtils::StringToBool(options_values[i]);
        }
            // Process enable-internal-profiling
        else if (std::string(options_keys[i]) == std::string("enable-internal-profiling"))
        {
            internalProfilingState = *options_values[i] != '0';
            optimizerOptions.m_ProfilingEnabled = internalProfilingState;
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
            extProfilingParams.m_EnableProfiling = armnn::stringUtils::StringToBool(options_values[i]);
        }
            // Process timeline-profiling
        else if (std::string(options_keys[i]) == std::string("timeline-profiling"))
        {
            extProfilingParams.m_TimelineEnabled = armnn::stringUtils::StringToBool(options_values[i]);
        }
            // Process outgoing-capture-file
        else if (std::string(options_keys[i]) == std::string("outgoing-capture-file"))
        {
            extProfilingParams.m_OutgoingCaptureFile = options_values[i];
        }
            // Process incoming-capture-file
        else if (std::string(options_keys[i]) == std::string("incoming-capture-file"))
        {
            extProfilingParams.m_IncomingCaptureFile = options_values[i];
        }
            // Process file-only-external-profiling
        else if (std::string(options_keys[i]) == std::string("file-only-external-profiling"))
        {
            extProfilingParams.m_FileOnly = armnn::stringUtils::StringToBool(options_values[i]);
        }
            // Process counter-capture-period
        else if (std::string(options_keys[i]) == std::string("counter-capture-period"))
        {
            extProfilingParams.m_CapturePeriod = static_cast<uint32_t>(std::stoul(options_values[i]));
        }
            // Process profiling-file-format
        else if (std::string(options_keys[i]) == std::string("profiling-file-format"))
        {
            extProfilingParams.m_FileFormat = options_values[i];
        }
            // Process serialize-to-dot
        else if (std::string(options_keys[i]) == std::string("serialize-to-dot"))
        {
            this->SetSerializeToDot(options_values[i]);
        }
        else
        {
            throw armnn::Exception("Unknown option for the ArmNN Delegate given: " + std::string(options_keys[i]));
        }
    }

    this->SetRuntimeOptions(runtimeOptions);
    this->SetOptimizerOptions(optimizerOptions);
    this->SetInternalProfilingParams(internalProfilingState, internalProfilingDetail);
    this->SetExternalProfilingParams(extProfilingParams);
}
} // namespace armnnDelegate
