//
// Copyright © 2021,2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus VisitPackOperator(DelegateData& delegateData,
                               TfLiteContext* tfLiteContext,
                               TfLiteNode* tfLiteNode,
                               int nodeIndex,
                               int32_t operatorCode)
{
    unsigned int numInputs = tfLiteNode->inputs->size;
    if (numInputs < 1)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext, "TfLiteArmnnDelegate: Must have at least one input in (%d != %d) in node #%d",
                1, numInputs, nodeIndex);
        return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;

    // Validate all inputs and get TensorInfo
    std::vector<armnn::TensorInfo> inputTensorInfos;
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[i]];
        if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }

        armnn::TensorInfo inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
        inputTensorInfos.emplace_back(inputTensorInfo);
    }

    // Convert input tensors to const armnn::TensorInfo* type for FORWARD_LAYER_SUPPORT_FUNC.
    std::vector<const armnn::TensorInfo*> inputConstTensorInfos;
    std::transform(inputTensorInfos.begin(),
                   inputTensorInfos.end(),
                   std::back_inserter(inputConstTensorInfos),
                   [](armnn::TensorInfo& t)->const armnn::TensorInfo*{ return &t; });

    // Validate output and get TensorInfo
    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    armnn::StackDescriptor desc;
    desc.m_NumInputs = static_cast<uint32_t>(numInputs);

    // Get axis from TfLite parameters
    auto* params = reinterpret_cast<TfLitePackParams*>(tfLiteNode->builtin_data);
    desc.m_Axis = static_cast<uint32_t>(params->axis);

    // Use the tensor shape of the first input as the "correct" input shape in the descriptor
    desc.m_InputShape = inputTensorInfos[0].GetShape();

    // Check if supported
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("STACK",
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
    return Connect(layer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate
