//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

std::string GetLayerName(armnn::ActivationFunction activationFunction)
{
    std::string layerName = "ACTIVATION";
    switch (activationFunction)
    {
        case armnn::ActivationFunction::Abs:
            layerName += " ABS";
            break;
        case armnn::ActivationFunction::BoundedReLu:
            layerName += " BOUNDED_RELU";
            break;
        case armnn::ActivationFunction::Elu:
            layerName += " ELU";
            break;
        case armnn::ActivationFunction::Gelu:
            layerName += " GELU";
            break;
        case armnn::ActivationFunction::HardSwish:
            layerName += " HARD_SWISH";
            break;
        case armnn::ActivationFunction::LeakyReLu:
            layerName += " LEAKY_RELU";
            break;
        case armnn::ActivationFunction::Linear:
            layerName += " LINEAR";
            break;
        case armnn::ActivationFunction::ReLu:
            layerName += " RELU";
            break;
        case armnn::ActivationFunction::Sigmoid:
            layerName += " SIGMOID";
            break;
        case armnn::ActivationFunction::SoftReLu:
            layerName += " SOFT_RELU";
            break;
        case armnn::ActivationFunction::Square:
            layerName += " SQUARE";
            break;
        case armnn::ActivationFunction::Sqrt:
            layerName += " SQRT";
            break;
        case armnn::ActivationFunction::TanH:
            layerName += " TANH";
            break;
        default:
            layerName += " UNKNOWN";
    }
    return layerName;
}

TfLiteStatus ValidateActivationOperator(DelegateData& delegateData,
                                        TfLiteOpaqueContext* tfLiteContext,
                                        const armnn::TensorInfo& inputInfo,
                                        const armnn::TensorInfo& outputInfo,
                                        armnn::ActivationDescriptor& activationDesc)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported, std::string layerName)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC(layerName.c_str(),
                                          tfLiteContext,
                                          IsActivationSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          armnn::BackendId(),
                                          inputInfo,
                                          outputInfo,
                                          activationDesc);
    };

    validateFunc(outputInfo, isSupported, GetLayerName(activationDesc.m_Function));
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus VisitActivationOperator(DelegateData& delegateData,
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
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    armnn::ActivationDescriptor activationDesc;
    switch(operatorCode)
    {
        case kTfLiteBuiltinRelu:
        {
            activationDesc.m_Function = armnn::ActivationFunction::ReLu;
            break;
        }
        case kTfLiteBuiltinRelu6:
        {
            activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
            activationDesc.m_A = 6.0f;
            break;
        }
        case kTfLiteBuiltinLogistic:
        {
            activationDesc.m_Function = armnn::ActivationFunction::Sigmoid;
            break;
        }
        case kTfLiteBuiltinTanh:
        {
            activationDesc.m_Function = armnn::ActivationFunction::TanH;
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = 1.0f;
            break;
        }
        case kTfLiteBuiltinElu:
        {
            activationDesc.m_Function = armnn::ActivationFunction::Elu;
            activationDesc.m_A = 1.0f;
            break;
        }
        case kTfLiteBuiltinHardSwish:
        {
            activationDesc.m_Function = armnn::ActivationFunction::HardSwish;
            break;
        }
        case kTfLiteBuiltinLeakyRelu:
        {
            // Get alpha param from builtin data
            auto* leakyReluParameters =
                reinterpret_cast<TfLiteLeakyReluParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
            activationDesc.m_Function = armnn::ActivationFunction::LeakyReLu;
            activationDesc.m_A = leakyReluParameters->alpha;
            break;
        }
        case kTfLiteBuiltinGelu:
        {
            activationDesc.m_Function = armnn::ActivationFunction::Gelu;
            break;
        }
        default:
        {
            return kTfLiteError;
        }
    }
    if (!delegateData.m_Network)
    {
        return ValidateActivationOperator(delegateData,
                                          tfLiteContext,
                                          inputTensorInfo,
                                          outputTensorInfo,
                                          activationDesc);
    }
    auto layerName = GetName(activationDesc.m_Function, nodeIndex);
    armnn::IConnectableLayer* activationLayer = delegateData.m_Network->AddActivationLayer(activationDesc,
                                                                                           layerName.c_str());
    ARMNN_ASSERT(activationLayer != nullptr);

    armnn::IOutputSlot& outputSlot = activationLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(activationLayer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // Connect
    return Connect(activationLayer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnDelegate
