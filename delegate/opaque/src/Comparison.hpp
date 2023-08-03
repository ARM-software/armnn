//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

std::string GetOperationName(armnn::ComparisonOperation comparisonOperation)
{
    std::string layerName = "COMPARISON";
    switch (comparisonOperation)
    {
        case armnn::ComparisonOperation::NotEqual:
            layerName += " NOT_EQUAL";
            break;
        case armnn::ComparisonOperation::Equal:
            layerName += " EQUAL";
            break;
        case armnn::ComparisonOperation::Greater:
            layerName += " GREATER";
            break;
        case armnn::ComparisonOperation::GreaterOrEqual:
            layerName += " GREATER_OR_EQUAL";
            break;
        case armnn::ComparisonOperation::Less:
            layerName += " LESS";
            break;
        case armnn::ComparisonOperation::LessOrEqual:
            layerName += " LESS_OR_EQUAL";
            break;
        default:
            layerName += " UNKNOWN";
    }
    return layerName;
}

TfLiteStatus VisitComparisonOperator(DelegateData& delegateData,
                                     TfLiteOpaqueContext* tfLiteContext,
                                     TfLiteOpaqueNode* tfLiteNode,
                                     int nodeIndex,
                                     int32_t tfLiteComparisonOperatorCode,
                                     armnn::ComparisonOperation comparisonOperation)
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

    // Use input indices to get input tensors.
    const TfLiteOpaqueTensor* tfLiteInputTensor0 = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor0, tfLiteComparisonOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteInputTensor1 = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor1, tfLiteComparisonOperatorCode, nodeIndex))
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

    // Use output indices to get output tensor.
    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfLiteComparisonOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    armnn::TensorInfo inputTensorInfo0 = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor0);
    armnn::TensorInfo inputTensorInfo1 = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor1);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    // Check if we need to expand the dims of the input tensor infos.
    // This is required for a few of the backends.
    if(inputTensorInfo0.GetNumDimensions() != inputTensorInfo1.GetNumDimensions())
    {
        ExpandTensorRankToEqual(inputTensorInfo0, inputTensorInfo1);
    }

    armnn::ComparisonDescriptor descriptor(comparisonOperation);
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported, std::string layerName)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC(layerName.c_str(),
                                          tfLiteContext,
                                          IsComparisonSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo0,
                                          inputTensorInfo1,
                                          outputTensorInfo,
                                          descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported, GetOperationName(comparisonOperation));
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetName(descriptor.m_Operation, nodeIndex);
    armnn::IConnectableLayer* comparisonLayer = delegateData.m_Network->AddComparisonLayer(descriptor,
                                                                                           layerName.c_str());
    comparisonLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(comparisonLayer != nullptr);

    armnn::IOutputSlot& outputSlot = comparisonLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(comparisonLayer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    return Connect(comparisonLayer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnOpaqueDelegate
