//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitUnpackOperator(DelegateData& delegateData,
                                 TfLiteOpaqueContext* tfLiteContext,
                                 TfLiteOpaqueNode* tfLiteNode,
                                 int nodeIndex,
                                 int32_t operatorCode)
{
    // Check inputs
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const int* inputTensors;
    int numInputs;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }
    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                     inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    auto* tfLiteNodeParameters = reinterpret_cast<TfLiteUnpackParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);

    // Get Unpack Axis
    const unsigned int unpackAxis = NonNegative(tfLiteNodeParameters->axis, nodeIndex);

    if (unpackAxis >= inputTensorInfo.GetNumDimensions())
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: The unpack axis #%d cannot be greater than or equal to "
                "the number of input dimensions #%d in operator #%d node #%d",
                unpackAxis, inputTensorInfo.GetNumDimensions(), operatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Get Unpack Num
    unsigned int unpackNum = NonNegative(tfLiteNodeParameters->num, nodeIndex);

    // If num is not defined, automatically infer from the length of the dimension axis.
    if(unpackNum == 0)
    {
        unpackNum = inputTensorInfo.GetShape()[unpackAxis];
    }

    // If unpack number cannot be inferred and is still zero, return kTfLiteError.
    if(unpackNum == 0)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Number to unpack must greater than zero in operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Check outputs
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, unpackNum, nodeIndex));

    auto inputDimSize = inputTensorInfo.GetNumDimensions();
    std::vector<unsigned int> unpackDimSizes(inputDimSize);

    // Add current input shape to unpackDimSizes
    for (unsigned int i = 0; i < inputDimSize; ++i)
    {
        unpackDimSizes[i] = inputTensorInfo.GetShape()[i];
    }

    if (unpackDimSizes[unpackAxis] != unpackNum)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Number to unpack must be the same as length "
                "of the dimension to unpack along in operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return kTfLiteError;
    }

    unpackDimSizes[unpackAxis] /= unpackNum;

    // ACL supports up to 4 dimensions
    if(unpackDimSizes.size() > 4)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: Split dimension size greater than 4 is not supported. "
            "Operator: #%d node #%d: ", operatorCode, nodeIndex);
        return kTfLiteError;
    }

    armnn::SplitterDescriptor splitDesc(unpackNum, static_cast<unsigned int>(unpackDimSizes.size()));
    splitDesc.SetAxis(unpackAxis);

    for (unsigned int j = 0; j < unpackNum; ++j)
    {
        // Set the size of the views.
        for (unsigned int dimIdx = 0; dimIdx < unpackDimSizes.size(); ++dimIdx)
        {
            splitDesc.SetViewSize(j, dimIdx, unpackDimSizes[dimIdx]);
        }
        splitDesc.SetViewOriginCoord(j, unpackAxis, unpackDimSizes[unpackAxis] * j);
    }

    // Gather output indices and use to get output tensors.
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

    // Validate all outputs and get TensorInfo
    std::vector<armnn::TensorInfo> outputs;
    for (unsigned int i = 0; i < unpackNum; ++i)
    {
        const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                          outputTensors[i]);
        if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }

        armnn::TensorInfo outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);
        armnn::TensorShape shape = outputTensorInfo.GetShape();
        if (shape.GetDimensionality() == armnn::Dimensionality::NotSpecified)
        {
            shape.SetNumDimensions(1, true);
            shape.SetDimensionSize(0, 1);
            outputTensorInfo.SetShape(shape);
        }

        outputs.push_back(outputTensorInfo);
    }

    const std::vector<std::reference_wrapper<armnn::TensorInfo>> outputTensorInfos(outputs.begin(), outputs.end());

    // Determine the shape of the Splitter layer outputs for validation
    armnn::TensorShape splitOutShape = armnn::TensorShape(static_cast<unsigned int>(unpackDimSizes.size()),
                                                          unpackDimSizes.data());

    std::vector<armnn::TensorInfo> splitterOutputs;
    for (unsigned int outputIndex = 0; outputIndex < outputTensorInfos.size(); ++outputIndex)
    {
        splitterOutputs.push_back(armnn::TensorInfo(splitOutShape,
                                                    outputTensorInfos[outputIndex].get().GetDataType(),
                                                    outputTensorInfos[outputIndex].get().GetQuantizationScale(),
                                                    outputTensorInfos[outputIndex].get().GetQuantizationOffset()));
    }
    std::vector<std::reference_wrapper<armnn::TensorInfo>> splitterOutputTensorInfos(splitterOutputs.begin(),
                                                                                     splitterOutputs.end());

    armnn::BackendId setBackendSplit;
    if (!delegateData.m_Network)
    {
        // Check if splitter is supported
        bool isSupported = false;
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("UNPACK",
                                          tfLiteContext,
                                          IsSplitterSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackendSplit,
                                          inputTensorInfo,
                                          splitterOutputTensorInfos,
                                          splitDesc);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Create Reshape descriptor from the first outputTensorInfo to validate a single Reshape layer
    // Use this descriptor later when creating every ReshapeLayer as all Reshape Layers should be the same
    armnn::ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = outputTensorInfos[0].get().GetShape();

    armnn::BackendId setBackendReshape;
    if (!delegateData.m_Network)
    {
        bool isSupported = false;
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("RESHAPE",
                                          tfLiteContext,
                                          IsReshapeSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackendReshape,
                                          splitterOutputTensorInfos[0],
                                          outputTensorInfos[0],
                                          reshapeDescriptor);
        return isSupported ? kTfLiteOk : kTfLiteError;
    };

    auto layerName = GetName(armnn::LayerType::Splitter, nodeIndex, "Unpack");
    armnn::IConnectableLayer* splitterLayer = delegateData.m_Network->AddSplitterLayer(splitDesc,
                                                                                       layerName.c_str());
    splitterLayer->SetBackendId(setBackendSplit);
    ARMNN_ASSERT(splitterLayer != nullptr);

    for (unsigned int k = 0; k < splitterLayer->GetNumOutputSlots(); ++k)
    {
        splitterLayer->GetOutputSlot(k).SetTensorInfo(outputs[k]);
    }

    // Connect the input slots
    auto inputIndex = static_cast<unsigned int>(inputTensors[0]);
    delegateData.m_OutputSlotForNode[inputIndex]->Connect(splitterLayer->GetInputSlot(0));

    // Create reshape to remove the unpacked dimension for unpack operator of each output from Splitter.
    for (unsigned int outputIndex = 0; outputIndex < splitterLayer->GetNumOutputSlots(); ++outputIndex)
    {
        auto reshapeLayerName = GetName(armnn::LayerType::Reshape, nodeIndex, "Unpack");
        armnn::IConnectableLayer* reshapeLayer = delegateData.m_Network->AddReshapeLayer(reshapeDescriptor,
                                                                                         reshapeLayerName.c_str());
        reshapeLayer->SetBackendId(setBackendReshape);
        ARMNN_ASSERT(reshapeLayer != nullptr);

        splitterLayer->GetOutputSlot(outputIndex).SetTensorInfo(splitterOutputTensorInfos[outputIndex]);
        splitterLayer->GetOutputSlot(outputIndex).Connect(reshapeLayer->GetInputSlot(0));

        armnn::TensorInfo outputTensorInfo  = outputTensorInfos[outputIndex];
        reshapeLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        armnn::IOutputSlot& slot = reshapeLayer->GetOutputSlot(0);

        delegateData.m_OutputSlotForNode[
                static_cast<unsigned long>(static_cast<unsigned int>(outputTensors[outputIndex]))] = &slot;

    }

    return kTfLiteOk;
}

} // namespace armnnOpaqueDelegate