//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>
#include <SharedFunctions.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitFullyConnectedOperator(DelegateData& delegateData,
                                         TfLiteOpaqueContext* tfLiteContext,
                                         TfLiteOpaqueNode* tfLiteNode,
                                         int nodeIndex,
                                         int32_t operatorCode)
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

    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteWeightsTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteWeightsTensor, operatorCode, nodeIndex))
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

    const armnn::TensorInfo& inputTensorInfo   = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& weightsTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteWeightsTensor);
    const armnn::TensorInfo& outputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    // Check for zero dimension in input and output tensors
    if(ZeroDimPresent({inputTensorInfo, outputTensorInfo}))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: Zero dimension tensors are not supported in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Check that we support fused activation before we attempt to create a layer
    auto* tfLiteNodeParameters =
            reinterpret_cast<TfLiteFullyConnectedParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
    TfLiteFusedActivation activationType=kTfLiteActNone;
    if (tfLiteNodeParameters)
    {
        activationType = tfLiteNodeParameters->activation;
        TfLiteStatus activationStatus = ValidateFusedActivationOperator(delegateData, tfLiteContext, outputTensorInfo,
                                                                        outputTensorInfo, activationType);
        if(activationStatus != kTfLiteOk)
        {
            return kTfLiteError;
        }
    }

    // Fully Connected Layer accepts two dimensional weights input
    int32_t weightsDimension = static_cast<int32_t>(weightsTensorInfo.GetNumDimensions());
    if (weightsDimension != 2)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dimension #$d for Fully Connected weights is not supported by Armnn"
                " in operator #%d node #%d: ", weightsDimension, operatorCode, nodeIndex);
        return kTfLiteError;
    }

    armnn::TensorInfo biasTensorInfo;
    const TfLiteOpaqueTensor* tfLiteBiasTensor = nullptr;

    bool biasEnabled = IsOptionalOperandPresent(tfLiteNode, 2);
    if (biasEnabled)
    {
        // Use input indices to get bias tensor.
        tfLiteBiasTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[2]);
        if (!IsValid(tfLiteContext, tfLiteBiasTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
        biasTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteBiasTensor);
    }
    else
    {
        biasTensorInfo = armnn::TensorInfo(armnn::TensorShape({1}), GetDataType(tfLiteInputTensor));
    }

    armnn::TensorInfo reshapedTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    if (inputTensorInfo.GetNumDimensions() > 2)
    {
        // Calculate reshape to flatten to 2D [batch_size, input_size]
        std::vector<unsigned int> reshapedDimensions(2);
        reshapedDimensions[1] = weightsTensorInfo.GetShape()[1];
        reshapedDimensions[0] = inputTensorInfo.GetNumElements() / reshapedDimensions[1];

        if (inputTensorInfo.GetNumElements() % reshapedDimensions[1] != 0)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Failed to deduce input tensor shape from filter size #%d #%d node #%d: ",
                reshapedDimensions[1], operatorCode, nodeIndex);
            return kTfLiteError;
        }

        reshapedTensorInfo.SetShape(armnn::TensorShape{ 2, reshapedDimensions.data() });
    }
    armnn::TensorInfo reshapedOutputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor);

    if (outputTensorInfo.GetNumDimensions() > 2)
    {
        // Calculate reshape to flatten to 2D [batch_size, input_size]
        std::vector<unsigned int> reshapedDimensions(2);
        reshapedDimensions[1] = weightsTensorInfo.GetShape()[0];
        reshapedDimensions[0] = outputTensorInfo.GetNumElements() / reshapedDimensions[1];

        if (outputTensorInfo.GetNumElements() % reshapedDimensions[1] != 0)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Failed to deduce output tensor shape from filter size #%d #%d node #%d: ",
                reshapedDimensions[1], operatorCode, nodeIndex);
            return kTfLiteError;
        }
        reshapedOutputTensorInfo.SetShape(armnn::TensorShape{ 2, reshapedDimensions.data() });
    }

    armnn::FullyConnectedDescriptor descriptor;
    descriptor.m_TransposeWeightMatrix = true;
    descriptor.m_BiasEnabled           = biasEnabled;
    descriptor.m_ConstantWeights       = weightsTensorInfo.IsConstant();

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {

        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("FULLY_CONNECTED",
                                          tfLiteContext,
                                          IsFullyConnectedSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          reshapedTensorInfo,
                                          outputTensorInfo,
                                          weightsTensorInfo,
                                          biasTensorInfo,
                                          descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(reshapedOutputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetName(armnn::LayerType::FullyConnected, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddFullyConnectedLayer(descriptor, layerName.c_str());
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    // Add a constant layer for weights and biases if inputs are constant.
    if (weightsTensorInfo.IsConstant())
    {
        auto weightsTensor = CreateConstTensor(tfLiteWeightsTensor, weightsTensorInfo);

        armnn::IConnectableLayer* weightsLayer = delegateData.m_Network->AddConstantLayer(weightsTensor);

        weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
        weightsLayer->GetOutputSlot(0).SetTensorInfo(weightsTensorInfo);
    }

    if (biasEnabled)
    {
        if(biasTensorInfo.IsConstant())
        {
            auto biasTensor = CreateConstTensor(tfLiteBiasTensor, biasTensorInfo);

            armnn::IConnectableLayer* biasLayer = delegateData.m_Network->AddConstantLayer(biasTensor);
            ARMNN_ASSERT(biasLayer != nullptr);

            biasLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
            biasLayer->GetOutputSlot(0).SetTensorInfo(biasTensorInfo);
        }
    }

    // The data input can also be constant, so we must check that this is also allocated to an input slot
    if(inputTensorInfo.IsConstant())
    {
        auto input = CreateConstTensor(tfLiteInputTensor, inputTensorInfo);

        armnn::IConnectableLayer* inputLayer = delegateData.m_Network->AddConstantLayer(input);
        inputLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0u));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
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
        delegateData.m_OutputSlotForNode[inputTensors[0]]->Connect(reshapeLayer->GetInputSlot(0));
        reshapeLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));

        if (!descriptor.m_ConstantWeights)
        {
            delegateData.m_OutputSlotForNode[inputTensors[1]]->Connect(layer->GetInputSlot(1));
        }

        if (biasEnabled && !biasTensorInfo.IsConstant())
        {
            delegateData.m_OutputSlotForNode[inputTensors[2]]->Connect(layer->GetInputSlot(2));
        }
        delegateData.m_OutputSlotForNode[outputTensors[0]] = &outputSlot;
    }

    if (reshapeLayer == nullptr)
    {
        if(Connect(layer, tfLiteContext, tfLiteNode, delegateData) != kTfLiteOk)
        {
            return kTfLiteError;
        }
    }

    if (outputTensorInfo.GetNumDimensions() > 2)
    {
        layer = AddReshapeLayer(tfLiteContext,
                                tfLiteNode,
                                layer,
                                reshapedOutputTensorInfo,
                                outputTensorInfo,
                                delegateData,
                                nodeIndex);
        if (!layer)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: Failed to add reshape for FullyConnected #%d node #%d: ",
                    operatorCode,
                    nodeIndex);
            return kTfLiteError;
        }
    }

    if (!tfLiteNodeParameters)
    {
        // No Activation
        return kTfLiteOk;
    }

    // Check and Create Activation
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, layer, 0, delegateData, nodeIndex);
}

} // namespace armnnOpaqueDelegate
