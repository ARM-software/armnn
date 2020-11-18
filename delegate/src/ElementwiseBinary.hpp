//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DelegateUtils.hpp"

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include "tensorflow/lite/delegates/utils.h"

namespace armnnDelegate
{

TfLiteStatus ValidateAddOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 const armnn::TensorInfo& inputInfo1,
                                 const armnn::TensorInfo& inputInfo2,
                                 const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsAdditionSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo1,
                                   inputInfo2,
                                   outputTensorInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ValidateDivOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 const armnn::TensorInfo& inputInfo1,
                                 const armnn::TensorInfo& inputInfo2,
                                 const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsDivisionSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo1,
                                   inputInfo2,
                                   outputTensorInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ValidateMaximumOperator(DelegateData& delegateData,
                                     TfLiteContext* tfLiteContext,
                                     const armnn::TensorInfo& inputInfo1,
                                     const armnn::TensorInfo& inputInfo2,
                                     const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsMaximumSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo1,
                                   inputInfo2,
                                   outputTensorInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ValidateMinimumOperator(DelegateData& delegateData,
                                     TfLiteContext* tfLiteContext,
                                     const armnn::TensorInfo& inputInfo1,
                                     const armnn::TensorInfo& inputInfo2,
                                     const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsMinimumSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo1,
                                   inputInfo2,
                                   outputTensorInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ValidateMulOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 const armnn::TensorInfo& inputInfo1,
                                 const armnn::TensorInfo& inputInfo2,
                                 const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsMultiplicationSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo1,
                                   inputInfo2,
                                   outputTensorInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ValidateSubOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 const armnn::TensorInfo& inputInfo1,
                                 const armnn::TensorInfo& inputInfo2,
                                 const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsSubtractionSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo1,
                                   inputInfo2,
                                   outputTensorInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus VisitElementwiseBinaryOperator(DelegateData& delegateData,
                                            TfLiteContext* tfLiteContext,
                                            TfLiteNode* tfLiteNode,
                                            int nodeIndex,
                                            int32_t elementwiseBinaryOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor0 = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteInputTensor0))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            elementwiseBinaryOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteInputTensor1 = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (IsDynamicTensor(tfLiteInputTensor1))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            elementwiseBinaryOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            elementwiseBinaryOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    armnn::TensorInfo inputTensorInfo0 = GetTensorInfoForTfLiteTensor(tfLiteInputTensor0);
    armnn::TensorInfo inputTensorInfo1 = GetTensorInfoForTfLiteTensor(tfLiteInputTensor1);

    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    if (!delegateData.m_Network)
    {
        switch(elementwiseBinaryOperatorCode)
        {
            case kTfLiteBuiltinAdd:
                return ValidateAddOperator(delegateData,
                                           tfLiteContext,
                                           inputTensorInfo0,
                                           inputTensorInfo1,
                                           outputTensorInfo);
            case kTfLiteBuiltinDiv:
                return ValidateDivOperator(delegateData,
                                           tfLiteContext,
                                           inputTensorInfo0,
                                           inputTensorInfo1,
                                           outputTensorInfo);
            case kTfLiteBuiltinMaximum:
                return ValidateMaximumOperator(delegateData,
                                               tfLiteContext,
                                               inputTensorInfo0,
                                               inputTensorInfo1,
                                               outputTensorInfo);
            case kTfLiteBuiltinMinimum:
                return ValidateMinimumOperator(delegateData,
                                               tfLiteContext,
                                               inputTensorInfo0,
                                               inputTensorInfo1,
                                               outputTensorInfo);
            case kTfLiteBuiltinMul:
                return ValidateMulOperator(delegateData,
                                           tfLiteContext,
                                           inputTensorInfo0,
                                           inputTensorInfo1,
                                           outputTensorInfo);
            case kTfLiteBuiltinSub:
                return ValidateSubOperator(delegateData,
                                           tfLiteContext,
                                           inputTensorInfo0,
                                           inputTensorInfo1,
                                           outputTensorInfo);
            default:
                return kTfLiteError;
        }
    }

    armnn::IConnectableLayer* elementwiseBinaryLayer = nullptr;

    switch(elementwiseBinaryOperatorCode)
    {
        case kTfLiteBuiltinAdd:
            elementwiseBinaryLayer = delegateData.m_Network->AddAdditionLayer();
            break;
        case kTfLiteBuiltinDiv:
            elementwiseBinaryLayer = delegateData.m_Network->AddDivisionLayer();
            break;
        case kTfLiteBuiltinMaximum:
            elementwiseBinaryLayer = delegateData.m_Network->AddMaximumLayer();
            break;
        case kTfLiteBuiltinMinimum:
            elementwiseBinaryLayer = delegateData.m_Network->AddMinimumLayer();
            break;
        case kTfLiteBuiltinMul:
            elementwiseBinaryLayer = delegateData.m_Network->AddMultiplicationLayer();
            break;
        case kTfLiteBuiltinSub:
            elementwiseBinaryLayer = delegateData.m_Network->AddSubtractionLayer();
            break;
        default:
            return kTfLiteError;
    }
    ARMNN_ASSERT(elementwiseBinaryLayer != nullptr);
    armnn::IOutputSlot& outputSlot = elementwiseBinaryLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    if(tflite::IsConstantTensor(&tfLiteInputTensor0))
    {
        auto status = ConnectConstant(elementwiseBinaryLayer,
                                      inputTensorInfo0,
                                      tfLiteContext,
                                      tfLiteInputTensor0,
                                      delegateData,
                                      tfLiteNode->inputs->data[0]);
        if (status == kTfLiteError)
        {
            return status;
        }
    }

    if(tflite::IsConstantTensor(&tfLiteInputTensor1))
    {
        auto status = ConnectConstant(elementwiseBinaryLayer,
                                      inputTensorInfo1,
                                      tfLiteContext,
                                      tfLiteInputTensor1,
                                      delegateData,
                                      tfLiteNode->inputs->data[1]);
        if (status == kTfLiteError)
        {
            return status;
        }
    }

    auto reshapeLayer = BroadcastTensor(inputTensorInfo0,
                                        inputTensorInfo1,
                                        elementwiseBinaryLayer,
                                        tfLiteContext,
                                        tfLiteNode,
                                        delegateData);
    if (!reshapeLayer)
    {
        return kTfLiteError;
    }

    auto* tfLiteNodeParameters = reinterpret_cast<TfLiteAddParams*>(tfLiteNode->builtin_data);
    if (!tfLiteNodeParameters)
    {
        // No Activation
        return kTfLiteOk;
    }
    // Check activation
    TfLiteFusedActivation activationType = tfLiteNodeParameters->activation;
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, elementwiseBinaryLayer, 0, delegateData);
}

} // namespace armnnDelegate
