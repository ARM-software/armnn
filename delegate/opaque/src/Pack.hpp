//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitPackOperator(DelegateData& delegateData,
                               TfLiteOpaqueContext* tfLiteContext,
                               TfLiteOpaqueNode* tfLiteNode,
                               int nodeIndex,
                               int32_t tfLitePackOperatorCode)
{
    // Check Inputs
    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);
    if (numInputs < 1)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Must have at least one input in (%d != %d) in node #%d",
                1,
                numInputs,
                nodeIndex);
        return kTfLiteError;
    }

    // Gather input indices and use to get input tensors.
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    // Validate all inputs and get TensorInfo
    std::vector<armnn::TensorInfo> inputTensorInfos;
    for (int i = 0; i < numInputs; ++i)
    {
        const TfLiteOpaqueTensor* inputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[i]);
        if (!IsValid(tfLiteContext, inputTensor, tfLitePackOperatorCode, nodeIndex))
        {
            return kTfLiteError;
        }

        armnn::TensorInfo inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(inputTensor);
        inputTensorInfos.emplace_back(inputTensorInfo);
    }

    // Convert inputTensorInfos to const armnn::TensorInfo* type for FORWARD_LAYER_OPAQUE_SUPPORT_FUNC.
    std::vector<const armnn::TensorInfo*> inputConstTensorInfos;
    std::transform(inputTensorInfos.begin(),
                   inputTensorInfos.end(),
                   std::back_inserter(inputConstTensorInfos),
                   [](armnn::TensorInfo& t)->const armnn::TensorInfo*{ return &t; });

    // Check outputs
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather output indices and use to get output tensor.
    const int* outputTensors;
    int numOutputs;
    if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors, &numOutputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather output tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    // Validate the output and get TensorInfo
    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfLitePackOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    armnn::StackDescriptor desc;
    desc.m_NumInputs = static_cast<uint32_t>(numInputs);

    // Get axis from TfLite parameters
    auto* tfLiteNodeParameters = reinterpret_cast<TfLitePackParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
    auto axis = tfLiteNodeParameters->axis;
    desc.m_Axis = NonNegative(axis, nodeIndex);

    // Use the tensor shape of the first input as the "correct" input shape in the descriptor
    desc.m_InputShape = inputTensorInfos[0].GetShape();

    // Check if supported
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("STACK",
                                          tfLiteContext,
                                          IsStackSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputConstTensorInfos,
                                          outputTensorInfo,
                                          desc);
    };

    // If the m_Network is a nullptr, this signals that a prerequisite TfLite callback is required to clarify the
    // support for the operator
    // If supported, VisitPackOperator will be called again to add the layer to the network as seen below
    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // The TfLite Pack operator is equivalent to the ArmNN Stack operator
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddStackLayer(desc);
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    // Connect the Constant Inputs
    auto inputsTensorsProcess = ProcessInputs(layer,
                                              delegateData,
                                              tfLiteContext,
                                              tfLiteNode);
    if (inputsTensorsProcess == kTfLiteError)
    {
        return inputsTensorsProcess;
    }

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // Connect
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnOpaqueDelegate