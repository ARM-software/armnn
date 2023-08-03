//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus ValidateReverseV2Operator(DelegateData& delegateData,
                                       TfLiteOpaqueContext* tfLiteContext,
                                       const armnn::TensorInfo& inputInfo0,
                                       const armnn::TensorInfo& inputInfo1,
                                       const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("REVERSEV2",
                                       tfLiteContext,
                                       IsReverseV2Supported,
                                       delegateData.m_Backends,
                                       isSupported,
                                       armnn::BackendId(),
                                       inputInfo0,
                                       inputInfo1,
                                       outputInfo);

    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus VisitReverseV2Operator(DelegateData& delegateData,
                                    TfLiteOpaqueContext* tfLiteContext,
                                    TfLiteOpaqueNode* tfLiteNode,
                                    int nodeIndex,
                                    int32_t reverseV2OperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensor.
    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
            nodeIndex);
        return kTfLiteError;
    }

    // The first input contains the data to be reversed
    const TfLiteOpaqueTensor* tfLiteInputTensor =
            TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            reverseV2OperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // The second input contains the axis tensor
    const TfLiteOpaqueTensor* tfLiteAxisTensor =
            TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (IsDynamicTensor(tfLiteAxisTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            reverseV2OperatorCode, nodeIndex);
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

    // Get the output tensor
    const TfLiteOpaqueTensor* tfLiteOutputTensor =
            TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            reverseV2OperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo0 =
            GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& inputTensorInfo1 =
            GetTensorInfoForTfLiteOpaqueTensor(tfLiteAxisTensor);
    const armnn::TensorInfo& outputTensorInfo =
            GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    if (inputTensorInfo0.GetNumDimensions() != outputTensorInfo.GetNumDimensions())
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: input tensor dimension and output tensor dimension differ #%d node #%d: ",
            reverseV2OperatorCode, nodeIndex);
        return kTfLiteError;
    }

    for (unsigned i=0; i < inputTensorInfo0.GetNumDimensions(); i++)
    {
        if (inputTensorInfo0.GetShape()[i] != outputTensorInfo.GetShape()[i])
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: input tensor dimension and output tensor differ #%d node #%d: ",
                reverseV2OperatorCode, nodeIndex);
            return kTfLiteError;
        }
    }

    // Get axis tensor data
    auto axisTensorNumValues = static_cast<unsigned int>(TfLiteOpaqueTensorDim(tfLiteAxisTensor,0));

    const auto maxDimension = 4;

    if (axisTensorNumValues > maxDimension)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: The Axis-Input-Tensor of the ReverseV2 operation requires a "
            "dimension of <= %d but a tensor with a dimension of %d was given.                      "
            "Operator: #%d node #%d: ",
            maxDimension, axisTensorNumValues, reverseV2OperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // No network pointer indicates that only support for this operator should be checked
    if (!delegateData.m_Network)
    {
        return ValidateReverseV2Operator(delegateData,
                                         tfLiteContext,
                                         inputTensorInfo0,
                                         inputTensorInfo1,
                                         outputTensorInfo);
    }

    auto layerName = GetName(armnn::LayerType::ReverseV2, nodeIndex);
    armnn::IConnectableLayer* reverseV2Layer = delegateData.m_Network->AddReverseV2Layer(layerName.c_str());

    armnn::IOutputSlot& outputSlot = reverseV2Layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(reverseV2Layer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    ARMNN_ASSERT(reverseV2Layer != nullptr);

    return Connect(reverseV2Layer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnOpaqueDelegate
