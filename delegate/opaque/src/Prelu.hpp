//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus ValidatePreluOperator(DelegateData& delegateData,
                                   TfLiteOpaqueContext* tfLiteContext,
                                   const armnn::TensorInfo& inputInfo,
                                   const armnn::TensorInfo& alphaInfo,
                                   const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("PRELU",
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
                                TfLiteOpaqueContext* tfLiteContext,
                                TfLiteOpaqueNode* tfLiteNode,
                                int nodeIndex,
                                int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensor.
    int numInputs = 0;
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteAlphaTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteAlphaTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Gather output indices and use to get output tensors.
    int numOutputs = 0;
    const int* outputTensors;
    if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors, &numOutputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather output tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& alphaTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteAlphaTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

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

    bool isConstantAlpha = IsConstantTensor(tfLiteAlphaTensor);

    // Add constant layer for constant alpha
    if (isConstantAlpha)
    {
        auto constAlphaTensor = armnn::ConstTensor(alphaTensorInfo, TfLiteOpaqueTensorData(tfLiteAlphaTensor));

        armnn::IConnectableLayer* constLayer = delegateData.m_Network->AddConstantLayer(constAlphaTensor);
        ARMNN_ASSERT(constLayer != nullptr);

        constLayer->GetOutputSlot(0).SetTensorInfo(alphaTensorInfo);
        constLayer->GetOutputSlot(0).Connect(preluLayer->GetInputSlot(1));
    }

    armnn::IOutputSlot& outputSlot = preluLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // Connect
    return Connect(preluLayer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnOpaqueDelegate
