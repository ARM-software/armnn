//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "armnn_delegate.hpp"
#include <armnn/Logging.hpp>

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
 *    Option key: "gpu-tuning-file" \n
 *    Possible values: [filenameString] \n
 *    Description: File name for the tuning file.
 *
 *    Option key: "gpu-kernel-profiling-enabled" \n
 *    Possible values: ["true"/"false"] \n
 *    Description: Enables GPU kernel profiling
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
            else
            {
                throw armnn::Exception("Unknown option for the ArmNN Delegate given: " + std::string(options_keys[i]));
            }
        }
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