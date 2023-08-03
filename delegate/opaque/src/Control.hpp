//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <DelegateUtils.hpp>
#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitConcatenationOperator(DelegateData& delegateData,
                                        TfLiteOpaqueContext* tfLiteContext,
                                        TfLiteOpaqueNode* tfLiteNode,
                                        int nodeIndex,
                                        int32_t tfLiteConcatOperatorCode)
{
    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);
    if (numInputs < 2)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Minimum number of inputs (%d != %d) in node #%d",
                2, numInputs, nodeIndex);
        return kTfLiteError;
    }
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensor.
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    std::vector<armnn::TensorInfo> inputTensorInfos;
    for (int i = 0; i < numInputs; ++i)
    {
        const TfLiteOpaqueTensor* inputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[i]);
        if (!IsValid(tfLiteContext, inputTensor, tfLiteConcatOperatorCode, nodeIndex))
        {
            return kTfLiteError;
        }

        armnn::TensorInfo inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(inputTensor);
        inputTensorInfos.emplace_back(inputTensorInfo);
    }

    // Convert input tensors to const armnn::TensorInfo* type for FORWARD_LAYER_SUPPORT_FUNC.
    std::vector<const armnn::TensorInfo*> inputConstTensorInfos;
    std::transform(inputTensorInfos.begin(),
                   inputTensorInfos.end(),
                   std::back_inserter(inputConstTensorInfos),
                   [](armnn::TensorInfo& t)->const armnn::TensorInfo*{ return &t; });

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
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfLiteConcatOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Setup OriginsDescriptor, axis and view origin
    auto numConcatView = static_cast<unsigned int>(numInputs);
    uint32_t inputRank = TfLiteOpaqueTensorNumDims(TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]));

    auto* concatenationParameters =
            reinterpret_cast<TfLiteConcatenationParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));

    if(!concatenationParameters)
    {
        throw armnn::Exception(&"TfLiteArmnnOpaqueDelegate: Concat parameters are null in: " [ nodeIndex ]);
    }

    const auto concatDimInput = static_cast<unsigned int>(
            (static_cast<int>(inputRank) + concatenationParameters->axis) % static_cast<int>(inputRank));

    armnn::OriginsDescriptor concatDescriptor(static_cast<uint32_t>(numConcatView), inputRank);
    concatDescriptor.SetConcatAxis(concatDimInput);

    unsigned int mergeDimOrigin = 0;
    for (unsigned int viewIndex = 0; viewIndex < numConcatView; ++viewIndex)
    {
        armnn::TensorInfo inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(
                TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[viewIndex]));

        // Sets up concatDescriptor view origin
        SetupConcatViewOrigin(inputTensorInfo, concatDescriptor, concatDimInput, viewIndex, mergeDimOrigin);
    }

    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

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
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("CONCATENATION",
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
    auto layerName = GetName(armnn::LayerType::Concat, nodeIndex);
    armnn::IConnectableLayer* concatenationLayer = delegateData.m_Network->AddConcatLayer(concatDescriptor,
                                                                                          layerName.c_str());
    concatenationLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(concatenationLayer != nullptr);

    // Connect the Constant Inputs
    auto inputsTensorsProcess = ProcessInputs(concatenationLayer,
                                              delegateData,
                                              tfLiteContext,
                                              tfLiteNode,
                                              nodeIndex);
    if (inputsTensorsProcess == kTfLiteError)
    {
        return inputsTensorsProcess;
    }

    armnn::IOutputSlot& outputSlot = concatenationLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);
    if(Connect(concatenationLayer, tfLiteContext, tfLiteNode, delegateData) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    if (activationType == kTfLiteActNone)
    {
        // No Activation
        return kTfLiteOk;
    }

    // Check and Create activation
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, concatenationLayer, 0, delegateData, nodeIndex);
}

TfLiteStatus VisitMeanOperator(DelegateData& delegateData,
                               TfLiteOpaqueContext* tfLiteContext,
                               TfLiteOpaqueNode* tfLiteNode,
                               int nodeIndex,
                               int32_t tfLiteMeanOperatorCode)
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
    if (!IsValid(tfLiteContext, tfLiteInputTensor, tfLiteMeanOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Use input indices to get axis tensor.
    const TfLiteOpaqueTensor* tfLiteAxisTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteAxisTensor, tfLiteMeanOperatorCode, nodeIndex))
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
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfLiteMeanOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo =  GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& axisTensorInfo =   GetTensorInfoForTfLiteOpaqueTensor(tfLiteAxisTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    auto* axisTensorData = static_cast<int32_t*>(TfLiteOpaqueTensorData(tfLiteAxisTensor));

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
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("MEAN",
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
    auto layerName = GetName(armnn::LayerType::Mean, nodeIndex);
    armnn::IConnectableLayer* meanLayer = delegateData.m_Network->AddMeanLayer(desc, layerName.c_str());
    meanLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(meanLayer != nullptr);

    armnn::IOutputSlot& outputSlot = meanLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(meanLayer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    return Connect(meanLayer, tfLiteContext, tfLiteNode, delegateData);
}

TfLiteStatus VisitControlOperator(DelegateData& delegateData,
                                  TfLiteOpaqueContext* tfLiteContext,
                                  TfLiteOpaqueNode* tfLiteNode,
                                  int nodeIndex,
                                  int32_t operatorCode)
{
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

