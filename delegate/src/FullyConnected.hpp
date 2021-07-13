//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DelegateUtils.hpp"
#include <armnn/utility/IgnoreUnused.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus VisitFullyConnectedOperator(DelegateData& delegateData,
                                         TfLiteContext* tfLiteContext,
                                         TfLiteNode* tfLiteNode,
                                         int nodeIndex,
                                         int32_t operatorCode)
{
    auto numInputs = tfLiteNode->inputs->size;
    if (numInputs < 2)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext, "TfLiteArmnnDelegate: Minimum number of inputs (%d != %d) in node #%d",
            2, numInputs, nodeIndex);
        return kTfLiteError;
    }
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    bool biasEnabled = (numInputs == 3);

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteWeightsTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (!IsValid(tfLiteContext, tfLiteWeightsTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo   = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    armnn::TensorInfo weightsTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteWeightsTensor);
    const armnn::TensorInfo& outputTensorInfo  = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    // Fully Connected Layer accepts two dimensional weights input
    int32_t weightsDimension = static_cast<int32_t>(weightsTensorInfo.GetNumDimensions());
    if (weightsDimension != 2)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dimension #$d for Fully Connected weights is not supported by Armnn"
            " in operator #%d node #%d: ", weightsDimension, operatorCode, nodeIndex);
        return kTfLiteError;
    }

    bool isConstantWeights = tflite::IsConstantTensor(&tfLiteWeightsTensor);

    armnn::TensorInfo biasTensorInfo;
    if (biasEnabled)
    {
        const TfLiteTensor& tfLiteBiasTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
        if (!IsValid(tfLiteContext, tfLiteBiasTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
        biasTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteBiasTensor);
    }
    else
    {
        biasTensorInfo = armnn::TensorInfo(armnn::TensorShape({1}), GetDataType(tfLiteInputTensor));
    }

    armnn::TensorInfo reshapedTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    if (inputTensorInfo.GetNumDimensions() > 2)
    {
        // Calculate reshape to flatten to 2D [batch_size, input_size]
        std::vector<unsigned int> reshapedDimensions(2);
        reshapedDimensions[1] = weightsTensorInfo.GetShape()[1];
        reshapedDimensions[0] = inputTensorInfo.GetNumElements() / reshapedDimensions[1];

        if (inputTensorInfo.GetNumElements() % reshapedDimensions[1] != 0)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Failed to deduce input tensor shape from filter size #%d #%d node #%d: ",
                reshapedDimensions[1], operatorCode, nodeIndex);
            return kTfLiteError;
        }

        reshapedTensorInfo.SetShape(armnn::TensorShape{ 2, reshapedDimensions.data() });
    }

    armnn::FullyConnectedDescriptor descriptor;
    descriptor.m_TransposeWeightMatrix = true;
    descriptor.m_BiasEnabled           = biasEnabled;
    descriptor.m_ConstantWeights       = isConstantWeights;

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsFullyConnectedSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   reshapedTensorInfo,
                                   outputTensorInfo,
                                   weightsTensorInfo,
                                   biasTensorInfo,
                                   descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* layer = delegateData.m_Network->AddFullyConnectedLayer(descriptor);
    ARMNN_ASSERT(layer != nullptr);

    // Add a constant layer for weights and biases if inputs are constant.
    if (isConstantWeights)
    {
        auto weightsTensor = CreateConstTensor(&tfLiteWeightsTensor,
                                               weightsTensorInfo,
                                               armnn::Optional<armnn::PermutationVector&>());

        armnn::IConnectableLayer* weightsLayer = delegateData.m_Network->AddConstantLayer(weightsTensor);

        weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
        weightsLayer->GetOutputSlot(0).SetTensorInfo(weightsTensorInfo);
    }

    if (biasEnabled)
    {
        const TfLiteTensor& tfLiteBiasTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
        if(tflite::IsConstantTensor(&tfLiteBiasTensor))
        {
            auto biasTensor = CreateConstTensor(&tfLiteBiasTensor,
                                                biasTensorInfo,
                                                armnn::Optional<armnn::PermutationVector&>());

            armnn::IConnectableLayer* biasLayer = delegateData.m_Network->AddConstantLayer(biasTensor);
            ARMNN_ASSERT(biasLayer != nullptr);

            biasLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
            biasLayer->GetOutputSlot(0).SetTensorInfo(biasTensorInfo);
        }
    }

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    armnn::IConnectableLayer* reshapeLayer = nullptr;
    if (inputTensorInfo.GetNumDimensions() > 2)
    {
        // Add reshape to flatten to 2D [batch_size, input_size]
        armnn::ReshapeDescriptor reshapeDescriptor;
        reshapeDescriptor.m_TargetShape = reshapedTensorInfo.GetShape();
        reshapeLayer = delegateData.m_Network->AddReshapeLayer(reshapeDescriptor);
        ARMNN_ASSERT(reshapeLayer != nullptr);

        reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedTensorInfo);

        // Connect
        delegateData.m_OutputSlotForNode[tfLiteNode->inputs->data[0]]->Connect(reshapeLayer->GetInputSlot(0));
        reshapeLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));

        if (!descriptor.m_ConstantWeights)
        {
            delegateData.m_OutputSlotForNode[tfLiteNode->inputs->data[1]]->Connect(layer->GetInputSlot(1));
        }

        if (biasEnabled && !tflite::IsConstantTensor(&tfLiteTensors[tfLiteNode->inputs->data[2]]))
        {
            delegateData.m_OutputSlotForNode[tfLiteNode->inputs->data[2]]->Connect(layer->GetInputSlot(2));
        }
        delegateData.m_OutputSlotForNode[tfLiteNode->outputs->data[0]] = &outputSlot;
    }

    if (reshapeLayer == nullptr)
    {
        Connect(layer, tfLiteNode, delegateData);
    }

    auto* tfLiteNodeParameters = reinterpret_cast<TfLiteFullyConnectedParams*>(tfLiteNode->builtin_data);
    if (!tfLiteNodeParameters)
    {
        // No Activation
        return kTfLiteOk;
    }

    // Check Activation
    TfLiteFusedActivation activationType = tfLiteNodeParameters->activation;
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, layer, 0, delegateData);
}

} // namespace armnnDelegate