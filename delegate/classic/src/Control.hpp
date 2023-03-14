//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/utility/IgnoreUnused.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/kernels/internal/tensor_ctypes.h>
#include <tensorflow/lite/minimal_logging.h>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

namespace armnnDelegate
{

void SetupConcatViewOrigin(const armnn::TensorInfo& inputTensorInfo,
                           armnn::OriginsDescriptor& concatDescriptor,
                           const unsigned int concatAxis,
                           unsigned int inputIndex,
                           unsigned int& mergeDimOrigin)
{
    const uint32_t inputRank = concatDescriptor.GetNumDimensions();

    // double check dimensions of the tensors
    if (inputTensorInfo.GetNumDimensions() != inputRank)
    {
        throw armnn::ParseException("The number of dimensions for input tensors "
                                    "of the concatenation operator should be: " + std::to_string(inputRank));
    }

    for (unsigned int j = 0; j < concatAxis; ++j)
    {
        concatDescriptor.SetViewOriginCoord(inputIndex, j, 0);
    }

    concatDescriptor.SetViewOriginCoord(inputIndex, concatAxis, mergeDimOrigin);
    mergeDimOrigin += inputTensorInfo.GetShape()[concatAxis];

    for (unsigned int j = concatAxis + 1; j < inputRank; ++j)
    {
        concatDescriptor.SetViewOriginCoord(inputIndex, j, 0);
    }
}

TfLiteStatus VisitConcatenationOperator(DelegateData& delegateData,
                                        TfLiteContext* tfLiteContext,
                                        TfLiteNode* tfLiteNode,
                                        int nodeIndex,
                                        int32_t tfLiteConcatOperatorCode)
{
    unsigned int numInputs = tfLiteNode->inputs->size;
    if (numInputs < 2)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext, "TfLiteArmnnDelegate: Minimum number of inputs (%d != %d) in node #%d",
            2, numInputs, nodeIndex);
        return kTfLiteError;
    }
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;

    std::vector<armnn::TensorInfo> inputTensorInfos;
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[i]];
        if (!IsValid(tfLiteContext, tfLiteInputTensor, tfLiteConcatOperatorCode, nodeIndex))
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

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfLiteConcatOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Setup OriginsDescriptor, axis and view origin
    unsigned int numConcatView = static_cast<unsigned int>(numInputs);
    uint32_t inputRank = tfLiteTensors[tfLiteNode->inputs->data[0]].dims->size;

    auto* concatenationParameters = reinterpret_cast<TfLiteConcatenationParams*>(tfLiteNode->builtin_data);

    if(!concatenationParameters)
    {
        throw armnn::Exception(&"TfLiteArmnnDelegate: Concat parameters are null in: " [ nodeIndex]);
    }

    const unsigned int concatDimInput = static_cast<unsigned int>(
            (static_cast<int>(inputRank) + concatenationParameters->axis) % static_cast<int>(inputRank));

    armnn::OriginsDescriptor concatDescriptor(static_cast<uint32_t>(numConcatView), inputRank);
    concatDescriptor.SetConcatAxis(concatDimInput);

    unsigned int mergeDimOrigin = 0;
    for (unsigned int viewIndex = 0; viewIndex < numConcatView; ++viewIndex)
    {
        armnn::TensorInfo inputTensorInfo = GetTensorInfoForTfLiteTensor(
                tfLiteTensors[tfLiteNode->inputs->data[viewIndex]]);

        // Sets up concatDescriptor view origin
        SetupConcatViewOrigin(inputTensorInfo, concatDescriptor, concatDimInput, viewIndex, mergeDimOrigin);
    }

    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    // Verify we support the fused activation before attempting to create a layer
    TfLiteFusedActivation activationType = concatenationParameters->activation;

    TfLiteStatus activationStatus = ValidateFusedActivationOperator(delegateData, tfLiteContext, outputTensorInfo,
                                                                    outputTensorInfo, activationType);
    if(activationStatus != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // Check if supported
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("CONCATENATION",
                                   tfLiteContext,
                                   IsConcatSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputConstTensorInfos,
                                   outputTensorInfo,
                                   concatDescriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Setup layer and connect.
    armnn::IConnectableLayer* concatenationLayer = delegateData.m_Network->AddConcatLayer(concatDescriptor);
    concatenationLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(concatenationLayer != nullptr);

    // Connect the Constant Inputs
    auto inputsTensorsProcess = ProcessInputs(concatenationLayer,
                                              delegateData,
                                              tfLiteContext,
                                              tfLiteNode);
    if (inputsTensorsProcess == kTfLiteError)
    {
        return inputsTensorsProcess;
    }

    armnn::IOutputSlot& outputSlot = concatenationLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);
    if(Connect(concatenationLayer, tfLiteNode, delegateData) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    if (activationType == kTfLiteActNone)
    {
        // No Activation
        return kTfLiteOk;
    }

    // Check and Create activation
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, concatenationLayer, 0, delegateData);
}

TfLiteStatus VisitMeanOperator(DelegateData& delegateData,
                               TfLiteContext* tfLiteContext,
                               TfLiteNode* tfLiteNode,
                               int nodeIndex,
                               int32_t tfLiteMeanOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if(!IsValid(&tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid input tensor in operator #%d node #%d: ",
            tfLiteMeanOperatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            tfLiteMeanOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteAxisTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if(!IsValid(&tfLiteAxisTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid axis tensor in operator #%d node #%d: ",
            tfLiteMeanOperatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteAxisTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic axis tensors are not supported in operator #%d node #%d: ",
            tfLiteMeanOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if(!IsValid(&tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid output tensor in operator #%d node #%d: ",
            tfLiteAxisTensor, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            tfLiteMeanOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo =  GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& axisTensorInfo =   GetTensorInfoForTfLiteTensor(tfLiteAxisTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    auto* axisTensorData = tflite::GetTensorData<int32_t>(&tfLiteAxisTensor);

    std::vector<int32_t> axis;
    // Add axis data to vector to be converter to unsigned int and assigned to descriptor axis.
    for (unsigned int i = 0; i < axisTensorInfo.GetNumElements(); ++i)
    {
        axis.emplace_back(axisTensorData[i]);
    }

    // Convert the axis to unsigned int and remove duplicates.
    unsigned int rank = inputTensorInfo.GetNumDimensions();
    std::set<unsigned int> uniqueAxis;
    std::transform(axis.begin(),
                   axis.end(),
                   std::inserter(uniqueAxis, uniqueAxis.begin()),
                   [rank](int i)->unsigned int{ return (i + rank) % rank; });

    // Setup MeanDescriptor and assign axis and keepDims
    armnn::MeanDescriptor desc;
    desc.m_Axis.assign(uniqueAxis.begin(), uniqueAxis.end());
    desc.m_KeepDims = inputTensorInfo.GetNumDimensions() == outputTensorInfo.GetNumDimensions() ? true : false;

    // Check if supported
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("MEAN",
                                   tfLiteContext,
                                   IsMeanSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outputTensorInfo,
                                   desc);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Setup layer and connect.
    armnn::IConnectableLayer* meanLayer = delegateData.m_Network->AddMeanLayer(desc);
    meanLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(meanLayer != nullptr);

    armnn::IOutputSlot& outputSlot = meanLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if(ProcessInputs(meanLayer,delegateData, tfLiteContext, tfLiteNode) != kTfLiteOk )
    {
        return kTfLiteError;
    }

    return Connect(meanLayer, tfLiteNode, delegateData);
}

TfLiteStatus VisitControlOperator(DelegateData& delegateData,
                                  TfLiteContext* tfLiteContext,
                                  TfLiteNode* tfLiteNode,
                                  int nodeIndex,
                                  int32_t operatorCode)
{
    armnn::IgnoreUnused(delegateData,
                        tfLiteContext,
                        tfLiteNode,
                        nodeIndex,
                        operatorCode);
                        
    switch(operatorCode)
    {
        case kTfLiteBuiltinConcatenation:
            return VisitConcatenationOperator(delegateData, tfLiteContext, tfLiteNode, nodeIndex, operatorCode);
        case kTfLiteBuiltinMean:
            return VisitMeanOperator(delegateData, tfLiteContext, tfLiteNode, nodeIndex, operatorCode);
        default:
            return kTfLiteError;
    }
}

} // namespace armnnDelegate
