//
// Copyright © 2020, 2023 Arm Ltd and Contributors. All rights reserved.
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

/**
  * Implementation of the TfLite external delegate plugin
  *
  * For details about what options_keys and option_values are supported please see:
  * armnnDelegate::DelegateOptions::DelegateOptions(char const* const*, char const* const*,size_t,void (*)(const char*))
  */
TfLiteDelegate* tflite_plugin_create_delegate(char** options_keys,
                                              char** options_values,
                                              size_t num_options,
                                              void (*report_error)(const char*))
{
    // Returning null indicates an error during delegate creation, we initialize with that
    TfLiteDelegate* delegate = nullptr;
    try
    {
        armnnDelegate::DelegateOptions options (options_keys, options_values, num_options, (*report_error));
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