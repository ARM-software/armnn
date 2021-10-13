//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//


#include "SharedFunctions.hpp"

#include "DelegateUtils.hpp"

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus ValidateFloorOperator(DelegateData& delegateData,
                                   TfLiteContext* tfLiteContext,
                                   const armnn::TensorInfo& inputTensorInfo,
                                   const armnn::TensorInfo& outputTensorInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsFloorSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputTensorInfo,
                                   outInfo);
    };
    validateFunc(outputTensorInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

} // namespace armnnDelegate

