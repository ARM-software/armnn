//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "armnn_delegate.hpp"
#include <armnn/Logging.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <iostream>
#include <tensorflow/lite/minimal_logging.h>

namespace tflite
{

/**
 * This file defines two symbols that need to be exported to use the TFLite external delegate provider. This is a plugin
 * that can be used for fast integration of delegates into benchmark tests and other tools. It allows loading of
 * a dynamic delegate library at runtime.
 *
 * The external delegate also has Tensorflow Lite Python bindings. Therefore the dynamic external delegate
 * can be directly used with Tensorflow Lite Python APIs.
 *
 * See tensorflow/lite/delegates/external for details or visit the tensorflow guide
 * [here](https://www.tensorflow.org/lite/performance/implementing_delegate#option_2_leverage_external_delegate)
 */

extern "C"
{
std::vector<std::string> gpu_options {"gpu-tuning-level",
                                      "gpu-tuning-file",
                                      "gpu-kernel-profiling-enabled"};


/**
 * Create an ArmNN delegate plugin
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
 *
 * @param[in]     option_keys     Delegate option names
 * @param[in]     options_values  Delegate option values
 * @param[in]     num_options     Number of delegate options
 * @param[in,out] report_error    Error callback function
 *
 * @return An ArmNN delegate if it succeeds else NULL
 */
TfLiteDelegate* tflite_plugin_create_delegate(char** options_keys,
                                              char** options_values,
                                              size_t num_options,
                                              void (*report_error)(const char*))
{
    // Returning null indicates an error during delegate creation so we initialize with that
    TfLiteDelegate* delegate = nullptr;
    try
    {
        // (Initializes with CpuRef backend)
        armnnDelegate::DelegateOptions options = armnnDelegate::TfLiteArmnnDelegateOptionsDefault();
        armnn::OptimizerOptions optimizerOptions;
        for (size_t i = 0; i < num_options; ++i)
        {
            // Process backends
            if (std::string(options_keys[i]) == std::string("backends"))
            {
                // The backend option is a comma separated string of backendIDs that needs to be split
                std::vector<armnn::BackendId> backends;
                char* pch;
                pch = strtok(options_values[i],",");
                while (pch != NULL)
                {
                    backends.push_back(pch);
                    pch = strtok (NULL, ",");
                }
                options.SetBackends(backends);
            }
            // Process logging level
            else if (std::string(options_keys[i]) == std::string("logging-severity"))
            {
                options.SetLoggingSeverity(options_values[i]);
            }
            // Process GPU backend options
            else if (std::string(options_keys[i]) == std::string("gpu-tuning-level"))
            {
                armnn::BackendOptions option("GpuAcc", {{"TuningLevel", atoi(options_values[i])}});
                options.AddBackendOption(option);
            }
            else if (std::string(options_keys[i]) == std::string("gpu-mlgo-tuning-file"))
            {
                armnn::BackendOptions option("GpuAcc", {{"MLGOTuningFilePath", std::string(options_values[i])}});
                options.AddBackendOption(option);
            }
            else if (std::string(options_keys[i]) == std::string("gpu-tuning-file"))
            {
                armnn::BackendOptions option("GpuAcc", {{"TuningFile", std::string(options_values[i])}});
                options.AddBackendOption(option);
            }
            else if (std::string(options_keys[i]) == std::string("gpu-kernel-profiling-enabled"))
            {
                armnn::BackendOptions option("GpuAcc", {{"KernelProfilingEnabled", (*options_values[i] != '0')}});
                options.AddBackendOption(option);
            }
            else if (std::string(options_keys[i]) == std::string("save-cached-network"))
            {
                armnn::BackendOptions option("GpuAcc", {{"SaveCachedNetwork", (*options_values[i] != '0')}});
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
                armnn::BackendOptions modelOptionGpu("GpuAcc", {{"FastMathEnabled", (*options_values[i] != '0')}});
                optimizerOptions.m_ModelOptions.push_back(modelOptionGpu);

                armnn::BackendOptions modelOptionCpu("CpuAcc", {{"FastMathEnabled", (*options_values[i] != '0')}});
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
               optimizerOptions.m_ReduceFp32ToFp16 = *options_values[i] != '0';
            }
            // Process reduce-fp32-to-bf16 option
            else if (std::string(options_keys[i]) == std::string("reduce-fp32-to-bf16"))
            {
               optimizerOptions.m_ReduceFp32ToBf16 = *options_values[i] != '0';
            }
            // Process debug-data
            else if (std::string(options_keys[i]) == std::string("debug-data"))
            {
               optimizerOptions.m_Debug = *options_values[i] != '0';
            }
            // Process memory-import
            else if (std::string(options_keys[i]) == std::string("memory-import"))
            {
               optimizerOptions.m_ImportEnabled = *options_values[i] != '0';
            }
            else
            {
                throw armnn::Exception("Unknown option for the ArmNN Delegate given: " + std::string(options_keys[i]));
            }
        }
        options.SetOptimizerOptions(optimizerOptions);
        delegate = TfLiteArmnnDelegateCreate(options);
    }
    catch (const std::exception& ex)
    {
        if(report_error)
        {
            report_error(ex.what());
        }
    }
    return delegate;
}

/** Destroy a given delegate plugin
 *
 * @param[in] delegate Delegate to destruct
 */
void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate)
{
    armnnDelegate::TfLiteArmnnDelegateDelete(delegate);
}

}  // extern "C"
}  // namespace tflite