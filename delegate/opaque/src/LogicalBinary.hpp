//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

std::string GetLayerName(armnn::LogicalBinaryOperation logicalBinaryOperation)
{
    std::string layerName = "LOGICAL_BINARY";
    switch (logicalBinaryOperation)
    {
        case armnn::LogicalBinaryOperation::LogicalAnd:
            layerName += " LOGICAL_AND";
            break;
        case armnn::LogicalBinaryOperation::LogicalOr:
            layerName += " LOGICAL_OR";
            break;
        default:
            layerName += " UNKNOWN";
    }
    return layerName;
}

TfLiteStatus VisitLogicalBinaryOperator(DelegateData& delegateData,
                                        TfLiteOpaqueContext* tfLiteContext,
                                        TfLiteOpaqueNode* tfLiteNode,
                                        int nodeIndex,
                                        int32_t logicalOperatorCode,
                                        armnn::LogicalBinaryOperation binaryOperation)
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
    if (!IsValid(tfLiteContext, tfLiteInputTensor0, logicalOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteInputTensor1 = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor1, logicalOperatorCode, nodeIndex))
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
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, logicalOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    armnn::TensorInfo inputTensorInfo0 = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor0);
    armnn::TensorInfo inputTensorInfo1 = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor1);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    // Check if we need to expand the dims of any input tensor infos.
    // This is required for a few of the backends.
    if(inputTensorInfo0.GetNumDimensions() != inputTensorInfo1.GetNumDimensions())
    {
        ExpandTensorRankToEqual(inputTensorInfo0, inputTensorInfo1);
    }

    // Setup descriptor and assign operation
    armnn::LogicalBinaryDescriptor desc;
    desc.m_Operation = binaryOperation;

    // Check if supported
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported, std::string layerName)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC(layerName.c_str(),
                                          tfLiteContext,
                                          IsLogicalBinarySupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo0,
                                          inputTensorInfo1,
                                          outputTensorInfo,
                                          desc);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported, GetLayerName(binaryOperation));
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetName(desc.m_Operation, nodeIndex);
    armnn::IConnectableLayer* logicalBinaryLayer = delegateData.m_Network->AddLogicalBinaryLayer(desc,
                                                                                                 layerName.c_str());
    logicalBinaryLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(logicalBinaryLayer != nullptr);

    armnn::IOutputSlot& outputSlot = logicalBinaryLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    auto inputsTensorsProcess = ProcessInputs(logicalBinaryLayer,
                                              delegateData,
                                              tfLiteContext,
                                              tfLiteNode,
                                              nodeIndex);
    if (inputsTensorsProcess == kTfLiteError)
    {
        return inputsTensorsProcess;
    }

    return Connect(logicalBinaryLayer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnOpaqueDelegate
