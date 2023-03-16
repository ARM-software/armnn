//
// Copyright © 2021-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SharedFunctions.hpp"

#include <ClassicDelegateUtils.hpp>

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
        FORWARD_LAYER_SUPPORT_FUNC("FLOOR",
                                   tfLiteContext,
                                   IsFloorSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   armnn::BackendId(),
                                   inputTensorInfo,
                                   outInfo);
    };
    validateFunc(outputTensorInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ValidateFusedActivationOperator(DelegateData& delegateData,
                                             TfLiteContext* tfLiteContext,
                                             const armnn::TensorInfo& inputInfo,
                                             const armnn::TensorInfo& outputInfo,
                                             TfLiteFusedActivation activationType)
{
    armnn::ActivationDescriptor activationDesc;

    switch (activationType)
    {
        case kTfLiteActNone:
        {
            // No Activation
            return kTfLiteOk;
        }
        case kTfLiteActRelu:
        {
            activationDesc.m_Function = armnn::ActivationFunction::ReLu;
            break;
        }
// The name of kTfLiteActRelu1 changed after TF Lite v2.3
#if defined(ARMNN_POST_TFLITE_2_3)
        case kTfLiteActReluN1To1:
#else
            case kTfLiteActRelu1:
#endif
        {
            activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = -1.0f;
            break;
        }
        case kTfLiteActRelu6:
        {
            activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
            activationDesc.m_A = 6.0f;
            activationDesc.m_B = 0.0f;
            break;
        }
        case kTfLiteActSigmoid:
        {
            activationDesc.m_Function = armnn::ActivationFunction::Sigmoid;
            break;
        }
        case kTfLiteActTanh:
        {
            activationDesc.m_Function = armnn::ActivationFunction::TanH;
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = 1.0f;
            break;
        }
        default:
            return kTfLiteError;
    }

    bool isSupported = false;
    armnn::BackendId setBackend;

    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("ACTIVATION",
                                   tfLiteContext,
                                   IsActivationSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   armnn::BackendId(),
                                   inputInfo,
                                   outputInfo,
                                   activationDesc);
    };
    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}


} // namespace armnnDelegate

