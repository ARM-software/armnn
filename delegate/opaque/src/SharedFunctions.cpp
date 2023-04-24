//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SharedFunctions.hpp"

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus ValidateFloorOperator(DelegateData& delegateData,
                                   TfLiteOpaqueContext* tfLiteContext,
                                   const armnn::TensorInfo& inputTensorInfo,
                                   const armnn::TensorInfo& outputTensorInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("FLOOR",
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
                                             TfLiteOpaqueContext* tfLiteContext,
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
        case kTfLiteActReluN1To1:
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
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("ACTIVATION",
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

