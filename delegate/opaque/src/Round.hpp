//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>
#include <SharedFunctions.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitFloorOperator(DelegateData& delegateData,
                                TfLiteOpaqueContext* tfLiteContext,
                                TfLiteOpaqueNode* tfLiteNode,
                                int nodeIndex,
                                int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
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
    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
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

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    // NOTE: looks like the outputTensorInfo is the only thing that is required for the case
    //       where we are adding the floor layer so maybe move the other stuff inside the
    //       if !delegateData block for efficiency.
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    // If the m_Network is a nullptr, this signals that a prerequisite TfLite callback is required to clarify the
    // support for the operator
    // If supported, VisitFloorOperator will be called again to add the layer to the network as seen further below
    if (!delegateData.m_Network)
    {
        return ValidateFloorOperator(delegateData, tfLiteContext, inputTensorInfo, outputTensorInfo);
    }

    // Add a Floor layer
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddFloorLayer();
    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if(ProcessInputs(layer,delegateData, tfLiteContext, tfLiteNode) != kTfLiteOk )
    {
        return kTfLiteError;
    }

    // Connect
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnOpaqueDelegate
