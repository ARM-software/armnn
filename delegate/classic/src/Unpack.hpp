//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/utility/IgnoreUnused.hpp>

#include <ClassicDelegateUtils.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <numeric>

namespace armnnDelegate
{

TfLiteStatus VisitUnpackOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 TfLiteNode* tfLiteNode,
                                 int nodeIndex,
                                 int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];

    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Get Unpack Axis
    const auto params = reinterpret_cast<TfLiteUnpackParams*>(tfLiteNode->builtin_data);

    const unsigned int unpackAxis = NonNegative(params->axis, nodeIndex);

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);

    if (unpackAxis >= inputTensorInfo.GetNumDimensions())
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: The unpack axis #%d cannot be greater than or equal to "
            "the number of input dimensions #%d in operator #%d node #%d",
            unpackAxis, inputTensorInfo.GetNumDimensions(), operatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Get Unpack Num
    unsigned int unpackNum = NonNegative(params->num, nodeIndex);

    // If num is not defined, automatically infer from the length of the dimension axis.
    if(unpackNum == 0)
    {
        unpackNum = inputTensorInfo.GetShape()[unpackAxis];
    }

    // If unpack number cannot be inferred and is still zero, return kTfLiteError.
    if(unpackNum == 0)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Number to unpack must greater than zero in operator #%d node #%d: ",
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
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Number to unpack must be the same as length "
            "of the dimension to unpack along in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }

    unpackDimSizes[unpackAxis] /= unpackNum;

    armnn::SplitterDescriptor splitDesc(unpackNum, static_cast<unsigned int>(unpackDimSizes.size()));
    for (unsigned int j = 0; j < unpackNum; ++j)
    {
        // Set the size of the views.
        for (unsigned int dimIdx = 0; dimIdx < unpackDimSizes.size(); ++dimIdx)
        {
            splitDesc.SetViewSize(j, dimIdx, unpackDimSizes[dimIdx]);
        }
        splitDesc.SetViewOriginCoord(j, unpackAxis, unpackDimSizes[unpackAxis] * j);
    }

    std::vector<armnn::TensorInfo> outputs;
    for (unsigned int i = 0; i < unpackNum; ++i)
    {
        const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[i]];
        if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
        outputs.push_back(GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true));
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
        FORWARD_LAYER_SUPPORT_FUNC("UNPACK",
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
        FORWARD_LAYER_SUPPORT_FUNC("RESHAPE",
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

    std::string splitterLayerName("Unpack Splitter");

    armnn::IConnectableLayer* splitterLayer = delegateData.m_Network->AddSplitterLayer(splitDesc,
                                                                                       splitterLayerName.c_str());
    splitterLayer->SetBackendId(setBackendSplit);
    ARMNN_ASSERT(splitterLayer != nullptr);

    for (unsigned int k = 0; k < splitterLayer->GetNumOutputSlots(); ++k)
    {
        splitterLayer->GetOutputSlot(k).SetTensorInfo(outputs[k]);
    }

    // Connect the input slots
    delegateData.m_OutputSlotForNode[tfLiteNode->inputs->data[0]]->Connect(splitterLayer->GetInputSlot(0));

    // Create reshape to remove the unpacked dimension for unpack operator of each output from Splitter.
    for (unsigned int outputIndex = 0; outputIndex < splitterLayer->GetNumOutputSlots(); ++outputIndex)
    {
        std::string reshapeLayerName("Unpack Reshape");
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
            static_cast<unsigned long>(tfLiteNode->outputs->data[outputIndex])] = &slot;

    }

    return kTfLiteOk;
}

} // namespace armnnDelegate
