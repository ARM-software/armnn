//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SharedFunctions.hpp"

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus ValidateFloorOperator(DelegateData& delegateData,
                                   TfLiteOpaqueContext* tfLiteContext,
                                   const armnn::TensorInfo& inputTensorInfo,
                                   const armnn::TensorInfo& outputTensorInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("FLOOR",
                                          tfLiteContext,
                                          IsFloorSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          armnn::BackendId(),
                                          inputTensorInfo,
                                          outInfo);
    };
    validateFunc(outputTensorInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ValidateFusedActivationOperator(DelegateData& delegateData,
                                             TfLiteOpaqueContext* tfLiteContext,
                                             const armnn::TensorInfo& inputInfo,
                                             const armnn::TensorInfo& outputInfo,
                                             TfLiteFusedActivation activationType)
{
    armnn::ActivationDescriptor activationDesc;

    switch (activationType)
    {
        case kTfLiteActNone:
        {
            // No Activation
            return kTfLiteOk;
        }
        case kTfLiteActRelu:
        {
            activationDesc.m_Function = armnn::ActivationFunction::ReLu;
            break;
        }
        case kTfLiteActReluN1To1:
        {
            activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = -1.0f;
            break;
        }
        case kTfLiteActRelu6:
        {
            activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
            activationDesc.m_A = 6.0f;
            activationDesc.m_B = 0.0f;
            break;
        }
        case kTfLiteActSigmoid:
        {
            activationDesc.m_Function = armnn::ActivationFunction::Sigmoid;
            break;
        }
        case kTfLiteActTanh:
        {
            activationDesc.m_Function = armnn::ActivationFunction::TanH;
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = 1.0f;
            break;
        }
        default:
            return kTfLiteError;
    }

    bool isSupported = false;
    armnn::BackendId setBackend;

    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("ACTIVATION",
                                          tfLiteContext,
                                          IsActivationSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          armnn::BackendId(),
                                          inputInfo,
                                          outputInfo,
                                          activationDesc);
    };
    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteOpaqueNode* GetNodeConnectedToInput(TfLiteOpaqueContext* tfLiteContext,
                                          int32_t& connectedIndex,
                                          int32_t inputIdx)
{
    TfLiteIntArray* executionPlan = nullptr;
    if (TfLiteOpaqueContextGetExecutionPlan(tfLiteContext, &executionPlan) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(tfLiteContext, "TfLiteArmnnDelegate: Unable to get graph execution plan.");
        return nullptr;
    }

    for (int i = 0; i < executionPlan->size; ++i)
    {
        connectedIndex = executionPlan->data[i];

        // If TfLite nodes can be delegated to ArmNN
        TfLiteOpaqueNode* connectedNode = nullptr;
        TfLiteRegistrationExternal* tfLiteRegistration = nullptr;
        if (TfLiteOpaqueContextGetNodeAndRegistration(
                tfLiteContext, connectedIndex, &connectedNode, &tfLiteRegistration) != kTfLiteOk)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(tfLiteContext,
                                            "TfLiteArmnnOpaqueDelegate: Unable to get node and registration for node "
                                            "%d.", connectedIndex);
            continue;
        }
        int numOutputs = 0;
        const int* outputTensors;

        if (TfLiteOpaqueNodeOutputs(connectedNode, &outputTensors, &numOutputs) != kTfLiteOk)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: Unable to gather output tensor indices from node #%d: ",
                    connectedIndex);
            continue;
        }

        for (int j= 0; j < numOutputs; ++j)
        {
            if (outputTensors[j] == inputIdx)
            {
                return connectedNode;
            }
        }
    }
    // No node found so set connectedIndex to -1
    connectedIndex = -1;
    return nullptr;
}

bool WillInputBeOptimizedToConst(TfLiteOpaqueContext* tfLiteContext, int32_t inputIdx)
{
    int32_t connectedIndex;
    TfLiteOpaqueNode* connectedNode = GetNodeConnectedToInput(tfLiteContext, connectedIndex, inputIdx);

    if (connectedNode)
    {
        TfLiteRegistrationExternal* tfLiteRegistration = nullptr;

        if (TfLiteOpaqueContextGetNodeAndRegistration(tfLiteContext, connectedIndex, &connectedNode,
                                                      &tfLiteRegistration) == kTfLiteOk)
        {
            switch (TfLiteRegistrationExternalGetBuiltInCode(tfLiteRegistration))
            {
                case kTfLiteBuiltinDequantize:
                {
                    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(connectedNode);
                    if (numInputs >= 1)
                    {
                        const int* inputTensors;
                        if (TfLiteOpaqueNodeInputs(connectedNode, &inputTensors, &numInputs) != kTfLiteOk)
                        {
                            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                                    tfLiteContext,
                                    "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                                    connectedIndex);
                            return kTfLiteError;
                        }
                        const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                inputTensors[0]);

                        // If the input to the Dequantize is a Constant then both that Constant layer and the Dequantize
                        // layer will be replaced by a single Constant layer containing the dequantized values.
                        if (IsConstantTensor(tfLiteInputTensor))
                        {
                            return true;
                        }
                    }
                    break;
                }
                default:
                {
                }
            }
        }
    }
    return false;
}

} // namespace armnnDelegate

