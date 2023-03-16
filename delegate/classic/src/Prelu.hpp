//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ClassicDelegateUtils.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus ValidatePreluOperator(DelegateData& delegateData,
                                   TfLiteContext* tfLiteContext,
                                   const armnn::TensorInfo& inputInfo,
                                   const armnn::TensorInfo& alphaInfo,
                                   const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("PRELU",
                                   tfLiteContext,
                                   IsPreluSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   armnn::BackendId(),
                                   inputInfo,
                                   alphaInfo,
                                   outputInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus VisitPreluOperator(DelegateData& delegateData,
                                TfLiteContext* tfLiteContext,
                                TfLiteNode* tfLiteNode,
                                int nodeIndex,
                                int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;

    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteAlphaTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (!IsValid(tfLiteContext, tfLiteAlphaTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& alphaTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteAlphaTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    if (!delegateData.m_Network)
    {
        return ValidatePreluOperator(delegateData,
                                     tfLiteContext,
                                     inputTensorInfo,
                                     alphaTensorInfo,
                                     outputTensorInfo);
    }

    armnn::IConnectableLayer* preluLayer = delegateData.m_Network->AddPreluLayer();
    ARMNN_ASSERT(preluLayer != nullptr);

    bool isConstantAlpha = tflite::IsConstantTensor(&tfLiteAlphaTensor);

    // Add constant layer for constant alpha
    if (isConstantAlpha)
    {
        auto constAlphaTensor = armnn::ConstTensor(alphaTensorInfo, tfLiteAlphaTensor.data.data);

        armnn::IConnectableLayer* constLayer = delegateData.m_Network->AddConstantLayer(constAlphaTensor);
        ARMNN_ASSERT(constLayer != nullptr);

        constLayer->GetOutputSlot(0).SetTensorInfo(alphaTensorInfo);
        constLayer->GetOutputSlot(0).Connect(preluLayer->GetInputSlot(1));
    }

    armnn::IOutputSlot& outputSlot = preluLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // Connect
    return Connect(preluLayer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate