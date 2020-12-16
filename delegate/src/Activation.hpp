//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DelegateUtils.hpp"

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus ValidateActivationOperator(DelegateData& delegateData,
                                        TfLiteContext* tfLiteContext,
                                        const armnn::TensorInfo& inputInfo,
                                        const armnn::TensorInfo& outputInfo,
                                        armnn::ActivationDescriptor& activationDesc)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsActivationSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo,
                                   outputInfo,
                                   activationDesc);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus VisitActivationOperator(DelegateData& delegateData,
                                     TfLiteContext* tfLiteContext,
                                     TfLiteNode* tfLiteNode,
                                     int nodeIndex,
                                     int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    armnn::ActivationDescriptor activationDesc;
    switch(operatorCode)
    {
        case kTfLiteBuiltinRelu:
        {
            activationDesc.m_Function = armnn::ActivationFunction::ReLu;
            break;
        }
        case kTfLiteBuiltinRelu6:
        {
            activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
            activationDesc.m_A = 6.0f;
            break;
        }
        case kTfLiteBuiltinLogistic:
        {
            activationDesc.m_Function = armnn::ActivationFunction::Sigmoid;
            break;
        }
        case kTfLiteBuiltinTanh:
        {
            activationDesc.m_Function = armnn::ActivationFunction::TanH;
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = 1.0f;
            break;
        }
        case kTfLiteBuiltinElu:
        {
            activationDesc.m_Function = armnn::ActivationFunction::Elu;
            activationDesc.m_A = 1.0f;
            break;
        }
        case kTfLiteBuiltinHardSwish:
        {
            activationDesc.m_Function = armnn::ActivationFunction::HardSwish;
            break;
        }
        default:
        {
            return kTfLiteError;
        }
    }
    if (!delegateData.m_Network)
    {
        return ValidateActivationOperator(delegateData,
                                          tfLiteContext,
                                          inputTensorInfo,
                                          outputTensorInfo,
                                          activationDesc);
    }
    armnn::IConnectableLayer* activationLayer = delegateData.m_Network->AddActivationLayer(activationDesc);
    ARMNN_ASSERT(activationLayer != nullptr);

    armnn::IOutputSlot& outputSlot = activationLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // Connect
    return Connect(activationLayer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate
